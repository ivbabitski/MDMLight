from __future__ import annotations

from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from db.sqlite_db import get_conn, init_golden_record


bp = Blueprint("golden_record_records_api", __name__, url_prefix="/api/golden-record")


def _require_user_id() -> int:
    raw = str(request.headers.get("X-User-Id", "") or "").strip()
    if not raw:
        raise ValueError("X-User-Id header is required")


    try:
        user_id = int(raw)
    except ValueError as e:
        raise ValueError("X-User-Id must be an integer") from e

    if user_id <= 0:
        raise ValueError("X-User-Id must be a positive integer")

    return user_id


@bp.get("/records")
def list_golden_record_records():
    """
    List golden_record rows for a user + model.

    Query params:
      - model_id (required)
      - limit (optional, default: 5000, max: 5000)
      - offset (optional, default: 0)

    Returns:
      { "records": [ ... ], "limit": int, "offset": int }
    """

    try:
        app_user_id = _require_user_id()
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    model_id = str(request.args.get("model_id", "") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    try:
        limit = int(str(request.args.get("limit", "5000") or "5000").strip())
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    try:
        offset = int(str(request.args.get("offset", "0") or "0").strip())
    except ValueError:
        return jsonify({"error": "offset must be an integer"}), 400

    if limit <= 0:
        return jsonify({"error": "limit must be >= 1"}), 400
    if limit > 5000:
        return jsonify({"error": "limit must be <= 5000"}), 400
    if offset < 0:
        return jsonify({"error": "offset must be >= 0"}), 400

    try:
        init_golden_record()
    except Exception as e:
        return (
            jsonify({"error": f"init_golden_record failed: {e.__class__.__name__}: {e}"}),
            500,
        )

    f_cols = [f"f{str(i).zfill(2)}" for i in range(1, 21)]
    cols = [
        "master_id",
        "job_id",
        "model_id",
        "source_name",
        "match_threshold",
        "survivorship_json",
        "representative_record_id",
        "representative_source_name",
        "representative_source_id",
        "lineage_json",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        *f_cols,
    ]

    sql = (
        f"SELECT {', '.join(cols)} "
        "FROM golden_record "
        "WHERE app_user_id = ? AND model_id = ? "
        "ORDER BY created_at DESC, master_id ASC "
        "LIMIT ? OFFSET ?"
    )

    try:
        with get_conn() as conn:
            cur = conn.execute(sql, (app_user_id, model_id, limit, offset))
            rows = cur.fetchall()
    except Exception as e:
        return (
            jsonify({"error": f"golden_record query failed: {e.__class__.__name__}: {e}"}),
            500,
        )

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})

    return jsonify({"records": out, "limit": limit, "offset": offset})
