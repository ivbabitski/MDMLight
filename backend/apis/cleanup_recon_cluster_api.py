import json
from flask import Blueprint, request, jsonify

from db.sqlite_db import get_conn, init_all_tables

bp = Blueprint("cleanup_recon_cluster_api", __name__)

init_all_tables()


def _require_user_id():
    v = request.headers.get("X-User-Id") or ""
    v = str(v).strip()
    if not v:
        return None, (jsonify({"error": "X-User-Id is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "X-User-Id must be an integer"}), 400)


def _require_mdm_model_id(payload: dict):
    v = payload.get("mdm_model_id", None)
    v = "" if v is None else str(v).strip()
    if not v:
        return None, (jsonify({"error": "mdm_model_id is required"}), 400)
    return v, None


def _require_model_id(payload: dict):
    v = payload.get("model_id", None)
    v = "" if v is None else str(v).strip()
    if not v:
        return None, (jsonify({"error": "model_id is required"}), 400)
    return v, None


@bp.delete("/api/cleanup/recon-cluster")
def cleanup_recon_cluster():
    payload = request.get_json(silent=True) or {}

    app_user_id, err = _require_user_id()
    if err:
        return err

    model_id, err = _require_model_id(payload)
    if err:
        return err

    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM recon_cluster WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, model_id),
        )
        deleted = cur.rowcount

        remaining_row = conn.execute(
            "SELECT COUNT(1) AS n FROM recon_cluster WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, model_id),
        ).fetchone()
        remaining = int(remaining_row["n"] or 0) if remaining_row else 0

    return jsonify(
        {
            "ok": True,
            "table": "recon_cluster",
            "app_user_id": app_user_id,
            "model_id": model_id,
            "deleted": deleted,
            "remaining": remaining,
        }
    )
