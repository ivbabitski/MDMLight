from __future__ import annotations

from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from db.sqlite_db import get_conn, init_recon_cluster


bp = Blueprint("recon_cluster_records_api", __name__, url_prefix="/api/recon-cluster")


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
def list_recon_cluster_records():
    """
    List recon_cluster records for a user + model.

    Query params:
      - model_id (required)
      - status: match | exception(s) (optional, default: match)
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

    status = str(request.args.get("status", "match") or "match").strip().lower()
    if status in {"match", "matches"}:
        status_mode = "match"
    elif status in {"exception", "exceptions"}:
        status_mode = "exceptions"
    else:
        return jsonify({"error": "status must be one of: match | exceptions"}), 400

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

    init_recon_cluster()

    f_cols = [f"f{str(i).zfill(2)}" for i in range(1, 21)]
    cols = [
        "id",
        "job_id",
        "cluster_id",
        "record_id",
        "model_id",
        "model_name",
        "source_name",
        "source_id",
        "match_status",
        "match_score",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        *f_cols,
    ]

    where_parts = ["app_user_id = ?", "model_id = ?"]
    params: List[Any] = [app_user_id, model_id]


    # "match" = full recon_cluster (no match_status filtering).
    # "exceptions" = only rows flagged as exceptions.
    if status_mode == "exceptions":
        where_parts.append("lower(coalesce(match_status, 'match')) <> 'match'")

    where_sql = " AND ".join(where_parts)


    sql = (
        f"SELECT {', '.join(cols)} "
        "FROM recon_cluster "
        f"WHERE {where_sql} "
        "ORDER BY cluster_id ASC, (match_score IS NULL) ASC, match_score DESC, id ASC "
        "LIMIT ? OFFSET ?"
    )

    params.extend([limit, offset])

    with get_conn() as conn:
        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})

    return jsonify({"records": out, "limit": limit, "offset": offset})
