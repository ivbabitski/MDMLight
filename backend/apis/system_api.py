import os
from flask import Blueprint, jsonify, request

try:
    from db.sqlite_db import get_conn, init_source_input
except Exception:
    # Fallback for local runs where module path is different
    from sqlite_db import get_conn, init_source_input


bp = Blueprint("source_systems_api", __name__, url_prefix="/api")


def _require_user_id():
    v = request.headers.get("X-User-Id") or ""
    v = str(v).strip()
    if not v:
        return None, (jsonify({"error": "X-User-Id header is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "X-User-Id must be an integer"}), 400)


@bp.get("/source-systems/health")
def source_systems_health():
    return jsonify({"ok": True})


@bp.get("/source-systems")
def list_source_systems():
    """Return distinct (source_id, source_name) pairs from source_input for the current user."""
    init_source_input()

    user_id, err = _require_user_id()
    if err:
        return err

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT source_id, source_name
              FROM source_input
             WHERE app_user_id = ?
             ORDER BY source_name, source_id
            """,
            (user_id,),
        ).fetchall()

    items = [{"source_id": r["source_id"], "source_name": r["source_name"]} for r in rows]
    return jsonify({"items": items, "count": len(items)})
