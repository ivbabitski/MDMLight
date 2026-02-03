import os
import json
import uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from db.sqlite_db import get_conn

bp = Blueprint("match_api", __name__)

def _require_app_user_id(payload: dict):
    v = ""
    if isinstance(payload, dict):
        v = payload.get("app_user_id") or ""
    v = v or request.headers.get("X-User-Id") or request.headers.get("X-App-User-Id") or request.args.get("app_user_id") or ""
    v = str(v).strip()
    if not v:
        dv = str(os.environ.get("DEFAULT_APP_USER_ID", "")).strip()
        if dv:
            v = dv
        else:
            return None, (jsonify({"error": "X-User-Id (app_user_id) is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "X-User-Id (app_user_id) must be an integer"}), 400)

@bp.post("/match/run")
def match_run():
    payload = request.get_json(force=True)

    app_user_id, err = _require_app_user_id(payload)
    if err:
        return err

    model_id = str(payload.get("model_id") or payload.get("mdm_model_id") or "").strip()

    if model_id:
        model = payload.get("model") or {}
        if not isinstance(model, dict):
            return jsonify({"error": "model must be an object if provided"}), 400

        job_id = str(uuid.uuid4())
        now = _utc_now_iso()

        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO match_job (
                  job_id, status, model_id, model_json, message, exceptions_json,
                  total_records, total_buckets, total_pairs_scored, total_matches, total_clusters,
                  created_at, updated_at, started_at, finished_at, app_user_id
                ) VALUES (?, 'queued', ?, ?, NULL, '[]', NULL, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, ?)
                """,
                (job_id, model_id, json.dumps(model), now, now, app_user_id),
            )

        return jsonify({"ok": True, "job_id": job_id, "status": "queued", "model_id": model_id})

    model = payload.get("model")

    if not isinstance(model, dict):
        return jsonify({"error": "model is required (object)"}), 400

    # Minimal required pieces
    selected_fields = model.get("selected_fields")
    if not isinstance(selected_fields, list) or len(selected_fields) == 0:
        return jsonify({"error": "model.selected_fields must be a non-empty array like ['f01','f02']"}), 400

    if "model_threshold" not in model:
        return jsonify({"error": "model.model_threshold is required"}), 400

    job_id = str(uuid.uuid4())
    now = _utc_now_iso()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO match_job (
              job_id, status, model_json, message, exceptions_json,
              total_records, total_buckets, total_pairs_scored, total_matches, total_clusters,
              created_at, updated_at, started_at, finished_at, app_user_id
            ) VALUES (?, 'queued', ?, NULL, '[]', NULL, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, ?)
            """,
            (job_id, json.dumps(model), now, now, app_user_id),
        )

    return jsonify({"ok": True, "job_id": job_id, "status": "queued"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



@bp.get("/match/status/<job_id>")
def match_status(job_id: str):
    app_user_id, err = _require_app_user_id({})
    if err:
        return err

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM match_job WHERE job_id=? AND app_user_id=?", (job_id, app_user_id)).fetchone()
        if not row:
            return jsonify({"error": "job_id not found", "job_id": job_id}), 404
        return jsonify(dict(row))