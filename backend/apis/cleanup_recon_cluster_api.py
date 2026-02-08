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
        model_row = conn.execute(
            "SELECT model_name FROM mdm_models WHERE id = ? AND app_user_id = ? AND deleted_at IS NULL LIMIT 1",
            (model_id, app_user_id),
        ).fetchone()
        model_name = str(model_row["model_name"] or "").strip() if model_row else ""

        if model_name:
            cur = conn.execute(
                "DELETE FROM recon_cluster WHERE app_user_id = ? AND (model_id = ? OR model_name = ?)",
                (app_user_id, model_id, model_name),
            )
        else:
            cur = conn.execute(
                "DELETE FROM recon_cluster WHERE app_user_id = ? AND model_id = ?",
                (app_user_id, model_id),
            )
        deleted = cur.rowcount

        cur_cm = conn.execute(
            "DELETE FROM cluster_map WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, model_id),
        )
        cluster_map_deleted = cur_cm.rowcount

        cur_me = conn.execute(
            "DELETE FROM match_exception WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, model_id),
        )
        match_exception_deleted = cur_me.rowcount

        if model_name:
            remaining_row = conn.execute(
                "SELECT COUNT(1) AS n FROM recon_cluster WHERE app_user_id = ? AND (model_id = ? OR model_name = ?)",
                (app_user_id, model_id, model_name),
            ).fetchone()
        else:
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
            "model_name": model_name,
            "deleted": deleted,
            "remaining": remaining,
            "cluster_map_deleted": cluster_map_deleted,
            "match_exception_deleted": match_exception_deleted,
        }
    )
