import os
import json
import uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from db.sqlite_db import get_conn, init_all_tables

bp = Blueprint("match_api", __name__)

init_all_tables()

def _require_app_user_id():
    v = request.headers.get("X-User-Id") or ""
    v = str(v).strip()
    if not v:
        return None, (jsonify({"error": "X-User-Id is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "X-User-Id must be an integer"}), 400)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@bp.post("/match/run")
def match_run():
    payload = request.get_json(silent=True) or {}

    app_user_id, err = _require_app_user_id()
    if err:
        return err


    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    job_id = str(uuid.uuid4())
    now = _utc_now_iso()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO match_job (
              job_id, status, model_id, model_json,
              message, exceptions_json,
              total_records, total_buckets, total_pairs_scored, total_matches, total_clusters,
              created_at, updated_at, started_at, finished_at, app_user_id
            ) VALUES (
              ?, 'queued', ?, ?,
              NULL, '[]',
              NULL, NULL, NULL, NULL, NULL,
              ?, ?, NULL, NULL, ?
            )
            """,
            (job_id, model_id, "{}", now, now, app_user_id),
        )

    return jsonify({"ok": True, "job_id": job_id, "status": "queued", "model_id": model_id})



@bp.get("/match/status/<job_id>")
def match_status(job_id: str):
    app_user_id, err = _require_app_user_id()
    if err:
        return err

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM match_job WHERE job_id=? AND app_user_id=?", (job_id, app_user_id)).fetchone()
        if not row:
            return jsonify({"error": "job_id not found", "job_id": job_id}), 404
        return jsonify(dict(row))


@bp.post("/match/recon_cluster/clear")
def clear_recon_cluster():
    payload = request.get_json(silent=True) or {}

    app_user_id, err = _require_app_user_id()
    if err:
        return err

    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    with get_conn() as conn:
        cur1 = conn.execute(
            "DELETE FROM recon_cluster WHERE app_user_id=? AND model_id=?",
            (app_user_id, model_id),
        )
        cur2 = conn.execute(
            "DELETE FROM cluster_map WHERE app_user_id=? AND model_id=?",
            (app_user_id, model_id),
        )

    return jsonify(
        {
            "ok": True,
            "app_user_id": app_user_id,
            "model_id": model_id,
            "recon_cluster_deleted": cur1.rowcount,
            "cluster_map_deleted": cur2.rowcount,
        }
    )


@bp.post("/match/golden_record/clear")
def clear_golden_record():
    payload = request.get_json(silent=True) or {}

    app_user_id, err = _require_app_user_id()
    if err:
        return err

    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM golden_record WHERE app_user_id=? AND model_id=?",
            (app_user_id, model_id),
        )

    return jsonify(
        {
            "ok": True,
            "app_user_id": app_user_id,
            "model_id": model_id,
            "golden_record_deleted": cur.rowcount,
        }
    )
