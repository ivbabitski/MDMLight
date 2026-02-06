import json
import sqlite3
import uuid
import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from db.sqlite_db import (
    get_user_by_username,
    init_mdm_models,
    create_mdm_model,
    list_mdm_models,
    get_mdm_model_by_id,
    get_mdm_model_by_name,
    update_mdm_model,
    soft_delete_mdm_model,
)

bp = Blueprint("mdm_model_api", __name__)

def _require_app_user_id():
    v = request.headers.get("X-User-Id") or request.args.get("app_user_id") or ""
    v = str(v).strip()

    if not v:
        actor = (request.headers.get("X-Actor") or request.args.get("actor") or "").strip()
        if actor:
            user = get_user_by_username(actor)
            if not user:
                return None, (jsonify({"error": "unknown actor"}), 401)
            v = str(user.get("id") or "").strip()

        if not v:
            dv = str(os.environ.get("DEFAULT_APP_USER_ID", "")).strip()
            if dv:
                v = dv
            else:
                return None, (jsonify({"error": "app_user_id is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "app_user_id must be an integer"}), 400)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _json():
    return request.get_json(silent=True) or {}

def _qs_bool(name: str, default: bool = False) -> bool:
    v = request.args.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _slugify(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    k = "".join(out)
    while "__" in k:
        k = k.replace("__", "_")
    return k.strip("_")

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def _require_actor(data: dict):
    actor = (data.get("actor") or request.headers.get("X-Actor") or "").strip()
    if not actor:
        return None, None, (jsonify({"error": "actor is required"}), 400)

    user = get_user_by_username(actor)
    if not user:
        return None, None, (jsonify({"error": "unknown actor"}), 401)

    return actor, int(user["id"]), None

def _normalize_config(cfg: dict, model_name: str) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")

    out = dict(cfg)

    out.setdefault("domainModelName", model_name)

    out.setdefault("sourceType", "csv")
    out.setdefault("csvFileName", "")
    out.setdefault("recordCount", 12500)

    out.setdefault("apiEndpoint", "http://localhost:5000/api/ingest")
    out.setdefault("apiToken", "")

    out.setdefault("aiExceptions", False)

    try:
        mt = float(out.get("matchThreshold", 0.85))
    except Exception:
        mt = 0.85
    try:
        pt = float(out.get("possibleThreshold", 0.7))
    except Exception:
        pt = 0.7

    mt = _clamp(mt, 0.5, 0.99)
    pt = _clamp(pt, 0.3, 0.95)
    if pt >= mt:
        pt = max(0.3, mt - 0.05)

    out["matchThreshold"] = mt
    out["possibleThreshold"] = pt

    out.setdefault("advanced", False)
    out.setdefault("globalRule", "recency_updated_date")
    out.setdefault("systemPriority", [])
    out.setdefault("userPriority", [])

    st = str(out.get("sourceType") or "").strip().lower()
    if st not in ("csv", "api"):
        raise ValueError("config.sourceType must be 'csv' or 'api'")
    out["sourceType"] = st

    if st == "csv":
        if not str(out.get("csvFileName") or "").strip():
            raise ValueError("config.csvFileName is required when sourceType='csv'")
    else:
        if not str(out.get("apiEndpoint") or "").strip():
            raise ValueError("config.apiEndpoint is required when sourceType='api'")
        if not str(out.get("apiToken") or "").strip():
            raise ValueError("config.apiToken is required when sourceType='api'")

    try:
        out["recordCount"] = int(out.get("recordCount") or 0)
    except Exception:
        out["recordCount"] = 0

    fields = out.get("fields")
    if not isinstance(fields, list) or len(fields) == 0:
        raise ValueError("config.fields is required (non-empty array)")

    match_field_codes = out.get("matchFieldCodes")
    match_code_set = None
    if match_field_codes is not None:
        if not isinstance(match_field_codes, list) or len(match_field_codes) == 0:
            raise ValueError("config.matchFieldCodes is required (non-empty array)")
        match_code_set = {str(c).strip() for c in match_field_codes if str(c).strip()}
        if len(match_code_set) == 0:
            raise ValueError("config.matchFieldCodes is required (non-empty array)")

    norm_fields = []
    key_count = 0
    match_fields = []

    for i, f in enumerate(fields):
        if not isinstance(f, dict):
            raise ValueError(f"config.fields[{i}] must be an object")

        ff = dict(f)

        if not str(ff.get("id") or "").strip():
            ff["id"] = f"f-{uuid.uuid4()}"

        ff["code"] = str(ff.get("code") or "").strip()
        ff["label"] = str(ff.get("label") or "")
        ff["kind"] = str(ff.get("kind") or "").strip()
        ff["type"] = str(ff.get("type") or "text").strip()
        ff["include"] = bool(ff.get("include"))
        ff["key"] = bool(ff.get("key"))

        if ff["key"]:
            key_count += 1

        try:
            ff["weight"] = float(ff.get("weight", 0.0))
        except Exception:
            ff["weight"] = 0.0
        ff["weight"] = _clamp(ff["weight"], 0.0, 1.0)

        is_match = ff["kind"] == "flex" and ff["include"] and (
            match_code_set is None or ff["code"] in match_code_set
        )
        if is_match:
            try:
                t = float(ff.get("matchThreshold", 0.8))
            except Exception:
                t = 0.8
            ff["matchThreshold"] = _clamp(t, 0.0, 1.0)
            match_fields.append(ff)

        if ff.get("rule") is None:
            ff["rule"] = str(out.get("globalRule") or "recency_updated_date")

        norm_fields.append(ff)

    if key_count != 1:
        raise ValueError("config.fields must contain exactly one key=true field")
    if len(match_fields) < 1:
        if match_code_set is None:
            raise ValueError("config.fields must include at least one flex/include=true match field")
        raise ValueError("config.matchFieldCodes must reference at least one flex/include=true field")


    if not bool(out.get("advanced")):
        gr = str(out.get("globalRule") or "recency_updated_date")
        for ff in norm_fields:
            if ff.get("include"):
                ff["rule"] = gr

    total_weight_pct = 0
    for ff in match_fields:
        total_weight_pct += int(round(float(ff.get("weight") or 0.0) * 100.0))
    if total_weight_pct < 95 or total_weight_pct > 105:
        raise ValueError(f"match field weights must total ~100% (currently {total_weight_pct}%)")

    out["fields"] = norm_fields
    return out

@bp.get("/mdm/models")
def mdm_list_models():
    init_mdm_models()
    include_deleted = _qs_bool("include_deleted", False)

    app_user_id, err = _require_app_user_id()
    if err:
        return err


    rows = list_mdm_models(include_deleted=include_deleted)
    out = []
    for r in rows:
        d = dict(r)
        # Multi-user scoping: only return models owned by this app_user_id
        if d.get("owner_user_id") is not None and int(d.get("owner_user_id")) != int(app_user_id):
            continue
        if d.get("app_user_id") is not None and int(d.get("app_user_id")) != int(app_user_id):
            continue

        try:
            d["config"] = json.loads(d.get("config_json") or "{}")
        except Exception:
            d["config"] = None

        # Normalize keys expected by the UI.
        cfg = d.get("config")
        if not str(d.get("model_name") or "").strip():
            alt = str(d.get("name") or "").strip()
            if not alt and isinstance(cfg, dict):
                alt = str(cfg.get("domainModelName") or "").strip()
            d["model_name"] = alt

        if d.get("app_user_id") is None and d.get("owner_user_id") is not None:
            d["app_user_id"] = d.get("owner_user_id")

        out.append(d)

    return jsonify({"ok": True, "models": out})


@bp.get("/mdm/models/<model_id>")
def mdm_get_model_by_id(model_id: str):
    init_mdm_models()
    include_deleted = _qs_bool("include_deleted", False)

    app_user_id, err = _require_app_user_id()
    if err:
        return err


    row = get_mdm_model_by_id(model_id, include_deleted=include_deleted)
    if not row:
        return jsonify({"error": "mdm model not found"}), 404

    d = dict(row)
    if d.get("owner_user_id") is not None and int(d.get("owner_user_id")) != int(app_user_id):
        return jsonify({"error": "mdm model not found"}), 404
    if d.get("app_user_id") is not None and int(d.get("app_user_id")) != int(app_user_id):
        return jsonify({"error": "mdm model not found"}), 404

    try:
        d["config"] = json.loads(d.get("config_json") or "{}")
    except Exception:
        d["config"] = None

    # Normalize keys expected by the UI.
    cfg = d.get("config")
    if not str(d.get("model_name") or "").strip():
        alt = str(d.get("name") or "").strip()
        if not alt and isinstance(cfg, dict):
            alt = str(cfg.get("domainModelName") or "").strip()
        d["model_name"] = alt

    if d.get("app_user_id") is None and d.get("owner_user_id") is not None:
        d["app_user_id"] = d.get("owner_user_id")

    return jsonify({"ok": True, "model": d})


@bp.get("/mdm/models/by-name/<model_name>")
def mdm_get_model_by_name(model_name: str):
    init_mdm_models()
    include_deleted = _qs_bool("include_deleted", False)

    app_user_id, err = _require_app_user_id()
    if err:
        return err


    row = get_mdm_model_by_name(model_name, include_deleted=include_deleted)
    if not row:
        return jsonify({"error": "mdm model not found"}), 404

    d = dict(row)
    if d.get("owner_user_id") is not None and int(d.get("owner_user_id")) != int(app_user_id):
        return jsonify({"error": "mdm model not found"}), 404
    if d.get("app_user_id") is not None and int(d.get("app_user_id")) != int(app_user_id):
        return jsonify({"error": "mdm model not found"}), 404

    try:
        d["config"] = json.loads(d.get("config_json") or "{}")
    except Exception:
        d["config"] = None

    return jsonify({"ok": True, "model": d})

@bp.post("/mdm/models")
def mdm_create_model():
    init_mdm_models()
    data = _json()
    actor, owner_user_id, err = _require_actor(data)
    if err:
        return err

    cfg = data.get("config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            return jsonify({"error": "config (string) must be valid JSON"}), 400

    model_name = str(data.get("model_name") or data.get("name") or (cfg.get("domainModelName") if isinstance(cfg, dict) else "") or "").strip()
    if not model_name:
        return jsonify({"error": "model_name (or config.domainModelName) is required"}), 400

    model_key = str(data.get("model_key") or "").strip()
    if not model_key:
        model_key = _slugify(model_name)
    if not model_key:
        return jsonify({"error": "model_key is required (or derivable from model_name)"}), 400

    if not isinstance(cfg, dict):
        return jsonify({"error": "config is required (object)"}), 400

    try:
        cfg_norm = _normalize_config(cfg, model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    config_json = json.dumps(cfg_norm, ensure_ascii=False, separators=(",", ":"))

    try:
        model_id = create_mdm_model(
            model_key=model_key,
            model_name=model_name,
            config_json=config_json,
            owner_user_id=owner_user_id,
            actor=actor,
            now=_utc_now_iso(),
        )
    except sqlite3.IntegrityError:
        return jsonify({"error": "active model_name or model_key already exists"}), 409

    return jsonify({"ok": True, "id": model_id, "model_key": model_key, "model_name": model_name}), 201

@bp.put("/mdm/models/<model_id>")
def mdm_update_model(model_id: str):
    init_mdm_models()
    data = _json()
    actor, actor_user_id, err = _require_actor(data)
    if err:
        return err

    # Ownership check: only owner can update
    existing = get_mdm_model_by_id(model_id, include_deleted=True)
    if not existing:
        return jsonify({"error": "mdm model not found"}), 404
    ex = dict(existing)
    if ex.get("owner_user_id") is not None and int(ex.get("owner_user_id")) != int(actor_user_id):
        return jsonify({"error": "forbidden"}), 403
    if ex.get("app_user_id") is not None and int(ex.get("app_user_id")) != int(actor_user_id):
        return jsonify({"error": "forbidden"}), 403


    model_name = data.get("model_name")
    if model_name is not None:
        model_name = str(model_name).strip()
        if not model_name:
            model_name = None

    cfg = data.get("config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            return jsonify({"error": "config (string) must be valid JSON"}), 400

    config_json = None
    if cfg is not None:
        if not isinstance(cfg, dict):
            return jsonify({"error": "config must be an object"}), 400
        use_name = model_name or str(cfg.get("domainModelName") or "").strip() or ""
        if not use_name:
            return jsonify({"error": "model_name (or config.domainModelName) is required"}), 400
        try:
            cfg_norm = _normalize_config(cfg, use_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        config_json = json.dumps(cfg_norm, ensure_ascii=False, separators=(",", ":"))

    ok = update_mdm_model(
        model_id=model_id,
        model_name=model_name,
        config_json=config_json,
        actor=actor,
        now=_utc_now_iso(),
    )
    if not ok:
        return jsonify({"error": "mdm model not found"}), 404

    return jsonify({"ok": True})

@bp.delete("/mdm/models/<model_id>")
def mdm_delete_model(model_id: str):
    init_mdm_models()
    data = _json()
    actor, actor_user_id, err = _require_actor(data)
    if err:
        return err

    # Ownership check: only owner can delete
    existing = get_mdm_model_by_id(model_id, include_deleted=True)
    if not existing:
        return jsonify({"error": "mdm model not found"}), 404
    ex = dict(existing)
    if ex.get("owner_user_id") is not None and int(ex.get("owner_user_id")) != int(actor_user_id):
        return jsonify({"error": "forbidden"}), 403
    if ex.get("app_user_id") is not None and int(ex.get("app_user_id")) != int(actor_user_id):
        return jsonify({"error": "forbidden"}), 403


    ok = soft_delete_mdm_model(model_id=model_id, actor=actor, now=_utc_now_iso())
    if not ok:
        return jsonify({"error": "mdm model not found"}), 404

    return jsonify({"ok": True})
