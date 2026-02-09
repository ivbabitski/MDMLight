import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from db.sqlite_db import (
    get_user_by_username,
    init_source_input,
    get_conn,
)

bp = Blueprint("mdm_source_systems_api", __name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _qs_bool(name: str, default: bool = False) -> bool:
    v = request.args.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _require_app_user_id():
    # Keep same behavior as mdm_model_api.py
    v = (
        request.headers.get("X-User-Id")
        or request.args.get("app_user_id")
        or ""
    )
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


@bp.get("/mdm/source-systems")
def mdm_list_source_systems():
    """
    Returns distinct source systems from source_input for the current app_user_id.

    Assumption (for now): source_input.source_name == source system name.
    """
    init_source_input()

    app_user_id, err = _require_app_user_id()
    if err:
        return err

    include_stats = _qs_bool("include_stats", True)
    since = str(request.args.get("since") or "").strip()   # optional created_at lower bound
    until = str(request.args.get("until") or "").strip()   # optional created_at upper bound

    where = "WHERE app_user_id = ?"
    params = [app_user_id]

    if since:
        where += " AND created_at >= ?"
        params.append(since)

    if until:
        where += " AND created_at <= ?"
        params.append(until)

    with get_conn() as conn:
        if include_stats:
            rows = conn.execute(
                f"""
                SELECT
                    source_name AS source_system,
                    COUNT(*)    AS record_count,
                    MIN(created_at) AS first_seen_at,
                    MAX(created_at) AS last_seen_at
                FROM source_input
                {where}
                GROUP BY source_name
                ORDER BY source_name ASC
                """,
                tuple(params),
            ).fetchall()

            out = [dict(r) for r in rows]
        else:
            rows = conn.execute(
                f"""
                SELECT DISTINCT
                    source_name AS source_system
                FROM source_input
                {where}
                ORDER BY source_name ASC
                """,
                tuple(params),
            ).fetchall()

            out = [{"source_system": r["source_system"]} for r in rows]

    return jsonify(
        {
            "ok": True,
            "app_user_id": app_user_id,
            "as_of": _utc_now_iso(),
            "source_systems": out,
        }
    )
