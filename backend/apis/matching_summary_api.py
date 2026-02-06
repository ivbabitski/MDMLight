import os
import json
from flask import Blueprint, jsonify, request

try:
    from db.sqlite_db import (
        get_conn,
        init_mdm_models,
        init_match_job,
        init_cluster_map,
        init_golden_record,
        init_match_exception,
    )
except Exception:
    from sqlite_db import (
        get_conn,
        init_mdm_models,
        init_match_job,
        init_cluster_map,
        init_golden_record,
        init_match_exception,
    )

bp = Blueprint("matching_summary_api", __name__, url_prefix="/api")

SURVIVORSHIP_LABELS = {
    "recency_created_date": "Recency (most recently created)",
    "recency_updated_date": "Recency (most recently updated)",
    "first_created_date": "First Created",
    "first_updated_date": "First Updated",
    "system": "By System Priority",
    "specific_value_priority": "By Specific Value",
}

def _require_app_user_id():
    v = request.headers.get("X-User-Id") or ""
    v = str(v).strip()
    if not v:
        return None, (jsonify({"error": "X-User-Id is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "X-User-Id must be an integer"}), 400)

def _rule_label(code: str) -> str:
    c = str(code or "").strip()
    if not c:
        return ""
    return SURVIVORSHIP_LABELS.get(c, c)

def _is_real_label(code: str, label: str) -> bool:
    c = str(code or "").strip().lower()
    l = str(label or "").strip().lower()
    if not c or not l:
        return False
    if l == c:
        return False
    if l.startswith("f") and l[1:].isdigit():
        return False
    return True

@bp.get("/matching/summary")
def matching_summary():
    init_mdm_models()
    init_match_job()
    init_cluster_map()
    init_golden_record()
    init_match_exception()

    app_user_id, err = _require_app_user_id()
    if err:
        return err

    model_id = str(request.args.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    with get_conn() as conn:
        # db file (debug)
        db_file = ""
        try:
            db_list = conn.execute("PRAGMA database_list").fetchall()
            for r in db_list:
                try:
                    name = r["name"]
                    file = r["file"]
                except Exception:
                    name = r[1]
                    file = r[2]
                if name == "main":
                    db_file = str(file or "")
                    break
        except Exception:
            db_file = ""

        # Load model config (and enforce ownership)
        row = conn.execute(
            """
            SELECT id, model_name, config_json, owner_user_id, app_user_id
            FROM mdm_models
            WHERE id = ?
            LIMIT 1
            """,
            (model_id,),
        ).fetchone()


        if not row:
            return jsonify({"error": "model not found"}), 404

        try:
            owner_id = row["owner_user_id"]
        except Exception:
            owner_id = None
        try:
            model_app_user_id = row["app_user_id"]
        except Exception:
            model_app_user_id = None
        try:
            model_name = str(row["model_name"] or "").strip()
        except Exception:
            model_name = ""


        if owner_id is not None and int(owner_id) != int(app_user_id):
            return jsonify({"error": "model does not belong to user"}), 403
        if model_app_user_id is not None and int(model_app_user_id) != int(app_user_id):
            return jsonify({"error": "model does not belong to user"}), 403

        try:
            cfg_raw = row["config_json"]
            model_cfg = json.loads(cfg_raw or "{}") if cfg_raw else {}
        except Exception:
            model_cfg = {}

        cfg_obj = model_cfg.get("config") if isinstance(model_cfg, dict) else None
        cfg = cfg_obj if isinstance(cfg_obj, dict) else (model_cfg if isinstance(model_cfg, dict) else {})
        fields_arr = cfg.get("fields") if isinstance(cfg, dict) else None
        if not isinstance(fields_arr, list):
            fields_arr = []

        advanced = bool(cfg.get("advanced"))
        global_rule = str(cfg.get("globalRule") or "").strip()

        def _as_float(v):
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                return None

        def _field_match_threshold(fd: dict):
            if not isinstance(fd, dict):
                return None

            v = fd.get("match_threshold")
            if v is None:
                v = fd.get("matchThreshold")
            if v is None:
                v = fd.get("threshold")
            if v is None:
                m = fd.get("match")
                if isinstance(m, dict):
                    v = m.get("match_threshold")
                    if v is None:
                        v = m.get("matchThreshold")
                    if v is None:
                        v = m.get("threshold")

            return _as_float(v)

        matching_fields = []
        rules_to_fields = {}

        for f in fields_arr:
            if not isinstance(f, dict):
                continue
            kind = str(f.get("kind") or "").strip()
            include = bool(f.get("include"))
            if kind != "flex" or not include:
                continue

            code = str(f.get("code") or "").strip()
            label = str(f.get("label") or "").strip()
            if not _is_real_label(code, label):
                label = code

            try:
                weight = float(f.get("weight") or 0.0)
            except Exception:
                weight = 0.0

            rule_code = str(f.get("rule") or global_rule or "").strip()

            if rule_code:
                rules_to_fields.setdefault(rule_code, []).append(code)

            match_threshold = _field_match_threshold(f)

            if weight > 0.0 and match_threshold is not None:
                matching_fields.append(
                    {
                        "code": code,
                        "label": label,
                        "weight": weight,
                        "weight_pct": int(round(weight * 100.0)),
                        "match_threshold": match_threshold,
                        "rule": rule_code,
                        "rule_label": _rule_label(rule_code),
                    }
                )

        match_field_pills = [x["label"] for x in matching_fields if str(x.get("label") or "").strip()]

        distinct_rules = [r for r in rules_to_fields.keys() if str(r or "").strip()]
        distinct_rules.sort()

        survivorship_mode = "global" if not advanced else "per_field"
        survivorship_label = ""

        if not advanced:
            survivorship_label = _rule_label(global_rule) if global_rule else ""
        else:
            if len(distinct_rules) == 0 and global_rule:
                survivorship_label = _rule_label(global_rule)
            elif len(distinct_rules) == 1:
                survivorship_label = _rule_label(distinct_rules[0])
            elif len(distinct_rules) > 1:
                survivorship_mode = "per_field_mixed"
                survivorship_label = "Per-field (mixed)"

        survivorship_rules = []
        for r in distinct_rules:
            survivorship_rules.append(
                {
                    "rule": r,
                    "label": _rule_label(r),
                    "field_codes": rules_to_fields.get(r, []),
                }
            )

        # Latest COMPLETED job (per your requirement)
        job_row = conn.execute(
            """
            SELECT job_id, status, created_at, updated_at, started_at, finished_at
            FROM match_job
            WHERE app_user_id = ?
              AND model_id = ?
              AND status = 'completed'
            ORDER BY COALESCE(finished_at, updated_at, created_at) DESC
            LIMIT 1
            """,
            (app_user_id, model_id),
        ).fetchone()

        job = None
        job_id = ""

        if job_row:
            try:
                job_id = str(job_row["job_id"] or "").strip()
            except Exception:
                job_id = ""
            job = {
                "job_id": job_id,
                "status": str(job_row["status"] or ""),
                "created_at": str(job_row["created_at"] or ""),
                "updated_at": str(job_row["updated_at"] or ""),
                "started_at": str(job_row["started_at"] or ""),
                "finished_at": str(job_row["finished_at"] or ""),
            }

        golden_records = 0
        exceptions = 0

        if job_id:
            gr = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM golden_record
                WHERE app_user_id = ?
                  AND model_id = ?
                  AND job_id = ?
                """,
                (app_user_id, model_id, job_id),
            ).fetchone()
            try:
                golden_records = int(gr["n"] or 0)
            except Exception:
                golden_records = 0

            ex = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM match_exception
                WHERE app_user_id = ?
                  AND model_id = ?
                  AND job_id = ?
                  AND resolved_at IS NULL
                """,
                (app_user_id, model_id, job_id),
            ).fetchone()
            try:
                exceptions = int(ex["n"] or 0)
            except Exception:
                exceptions = 0

        cl = conn.execute(
            """
            SELECT COUNT(DISTINCT cluster_id) AS n
            FROM cluster_map
            WHERE app_user_id = ?
              AND model_id = ?
            """,
            (app_user_id, model_id),
        ).fetchone()
        try:
            match_clusters = int(cl["n"] or 0)
        except Exception:
            match_clusters = 0

    return jsonify(
        {
            "db_file": db_file,
            "model_id": model_id,
            "model_name": model_name,
            "job": job,
            "golden_records": golden_records,
            "exceptions": exceptions,
            "match_clusters": match_clusters,
            "matching_fields": matching_fields,
            "match_field_pills": match_field_pills,
            "survivorship_mode": survivorship_mode,
            "survivorship_label": survivorship_label,
            "survivorship_global_rule": global_rule,
            "survivorship_rules": survivorship_rules,
        }
    )
