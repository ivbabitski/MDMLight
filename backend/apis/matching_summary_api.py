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

from services.services import ServiceError, matching_summary_get


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
    app_user_id, err = _require_app_user_id()
    if err:
        return err

    model_id = str(request.args.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    try:
        payload = matching_summary_get(app_user_id=app_user_id, model_id=model_id)
        return jsonify(payload)
    except ServiceError as se:
        out = {"error": str(se)}
        try:
            if getattr(se, "payload", None):
                out.update(se.payload)
        except Exception:
            pass
        return jsonify(out), int(getattr(se, "status_code", 500))
    except Exception:
        return jsonify({"error": "internal error"}), 500
