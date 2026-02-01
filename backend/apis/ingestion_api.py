import os
import sqlite3
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from db.sqlite_db import get_conn, init_source_input

bp = Blueprint("ingestion_api", __name__)

MAX_ROWS = int(os.environ.get("INGEST_MAX_ROWS", "5000"))

def _require_app_user_id(payload: dict):
    v = payload.get("app_user_id") or request.headers.get("X-App-User-Id") or ""
    v = str(v).strip()
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


def _looks_like_email(s: str) -> bool:
    if " " in s:
        return False
    if "@" not in s:
        return False
    left, right = s.split("@", 1)
    if not left or not right:
        return False
    return "." in right


def _looks_like_phone(s: str) -> bool:
    if any(ch.isalpha() for ch in s):
        return False
    digits = sum(1 for ch in s if ch.isdigit())
    return 7 <= digits <= 15


def _looks_like_number(s: str) -> bool:
    if any(ch.isalpha() for ch in s):
        return False
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False


def _looks_like_date(s: str) -> bool:
    x = s
    if x.endswith("Z"):
        x = x[:-1] + "+00:00"
    try:
        datetime.fromisoformat(x)
        return True
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            datetime.strptime(s, fmt)
            return True
        except Exception:
            continue
    return False


def _guess_type(v: str) -> str:
    s = str(v or "").strip()
    if not s:
        return "text"
    if _looks_like_email(s):
        return "email"
    if _looks_like_phone(s):
        return "phone"
    if _looks_like_date(s):
        return "date"
    if _looks_like_number(s):
        return "number"
    return "text"


def _best_type(samples) -> str:
    counts = {"email": 0, "phone": 0, "date": 0, "number": 0, "text": 0}
    for v in samples:
        counts[_guess_type(v)] += 1

    best = max(counts.values()) if counts else 0
    if best <= 0:
        return "text"

    for t in ("email", "phone", "date", "number", "text"):
        if counts.get(t, 0) == best:
            return t
    return "text"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@bp.post("/ingest/batch")
def ingest_batch():
    payload = request.get_json(force=True)

    app_user_id, err = _require_app_user_id(payload)
    if err:
        return err


    source_name = payload.get("source_name")
    actor = payload.get("actor")
    rows = payload.get("rows")

    if not source_name or not isinstance(source_name, str):
        return jsonify({"error": "source_name is required (string)"}), 400
    if not actor or not isinstance(actor, str):
        return jsonify({"error": "actor is required (string)"}), 400
    if not isinstance(rows, list) or len(rows) == 0:
        return jsonify({"error": "rows must be a non-empty array"}), 400
    if len(rows) > MAX_ROWS:
        return jsonify({"error": f"too many rows: {len(rows)} (max {MAX_ROWS})"}), 400

    init_source_input()

    init_source_input()

    headers = payload.get("headers") or payload.get("field_names")
    if headers is not None and not isinstance(headers, list):
        return jsonify({"error": "headers must be an array (optional)"}), 400

    samples = [[] for _ in range(20)]
    max_field_len = 0


    now = _utc_now_iso()

    cols = [
        "source_id", "source_name",
        "f01","f02","f03","f04","f05","f06","f07","f08","f09","f10",
        "f11","f12","f13","f14","f15","f16","f17","f18","f19","f20",
        "created_at", "created_by", "updated_at", "updated_by"
        "app_user_id",
    ]

    insert_sql = f"""
    INSERT INTO source_input (
      {",".join(cols)}
    ) VALUES (
      ?, ?,
      ?,?,?,?,?,?,?,?,?,?,
      ?,?,?,?,?,?,?,?,?,?,
      ?, ?,
      NULL, NULL, ?
    )
    ON CONFLICT(source_id) DO UPDATE SET
      source_name=excluded.source_name,
      f01=excluded.f01, f02=excluded.f02, f03=excluded.f03, f04=excluded.f04, f05=excluded.f05,
      f06=excluded.f06, f07=excluded.f07, f08=excluded.f08, f09=excluded.f09, f10=excluded.f10,
      f11=excluded.f11, f12=excluded.f12, f13=excluded.f13, f14=excluded.f14, f15=excluded.f15,
      f16=excluded.f16, f17=excluded.f17, f18=excluded.f18, f19=excluded.f19, f20=excluded.f20,
      app_user_id=COALESCE(source_input.app_user_id, excluded.app_user_id),
      updated_at=?,
      updated_by=?
    WHERE source_input.app_user_id IS NULL OR source_input.app_user_id = excluded.app_user_id;
    """

    params = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            return jsonify({"error": f"row {i} must be an object"}), 400

        source_id = r.get("source_id")
        fields = r.get("fields")

        if not source_id or not isinstance(source_id, str):
            return jsonify({"error": f"row {i}: source_id is required (string)"}), 400
        if not isinstance(fields, list):
            return jsonify({"error": f"row {i}: fields is required (array length <= 20)"}), 400
        if len(fields) > 20:
            return jsonify({"error": f"row {i}: fields length is {len(fields)} (max 20)"}), 400

        padded = (fields + [None] * 20)[:20]
        padded = [None if v is None else str(v) for v in padded]

        # Insert uses created_at/created_by. updated_* are NULL on insert.
        # Update path sets updated_at/updated_by via the final 2 placeholders.
        tup = (
            source_id, source_name,
            *padded,
            now, actor,   # created_at, created_by
            app_user_id,
            now, actor    # updated_at, updated_by (used only on UPDATE)
        )
        params.append(tup)



    # Block cross-user overwrite: if a source_id exists with a different non-null app_user_id, fail fast.
    source_ids = [r.get("source_id") for r in rows if isinstance(r, dict) and r.get("source_id")]
    conflicts = []
    if source_ids:
        # SQLite default variable limit is often 999. Chunk to stay safe.
        chunk_size = 900
        with get_conn() as conn:
            for off in range(0, len(source_ids), chunk_size):
                chunk = source_ids[off:off + chunk_size]
                ph = ",".join(["?"] * len(chunk))
                q = (
                    f"SELECT source_id, app_user_id FROM source_input "
                    f"WHERE source_id IN ({ph}) AND app_user_id IS NOT NULL AND app_user_id <> ? "
                    f"LIMIT 25"
                )
                cur = conn.execute(q, (*chunk, app_user_id))
                for rr in cur.fetchall():
                    try:
                        conflicts.append({"source_id": rr["source_id"], "app_user_id": rr["app_user_id"]})
                    except Exception:
                        conflicts.append({"source_id": rr[0], "app_user_id": rr[1]})
                if conflicts:
                    break

    if conflicts:
        return jsonify({
            "error": "source_id already exists for another user",
            "conflicts": conflicts,
            "detail": "Some source_id values are already owned by a different app_user_id.",
        }), 409

    try:
        with get_conn() as conn:
            conn.executemany(insert_sql, params)
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "no such table" in msg.lower():
            return jsonify({
                "error": "source_input table not found. Run your SQL script to create it first.",
                "detail": msg
            }), 500
        return jsonify({"error": "sqlite error", "detail": msg}), 500

    field_count = min(20, max_field_len)

    header_list = []
    if isinstance(headers, list) and headers:
        header_list = [str(h or "").strip() for h in headers][:field_count]
        while len(header_list) < field_count:
            header_list.append(f"f{len(header_list) + 1:02d}")
    else:
        header_list = [f"f{j + 1:02d}" for j in range(field_count)]

    schema_fields = []
    for j in range(field_count):
        schema_fields.append(
            {
                "code": f"f{j + 1:02d}",
                "header": header_list[j],
                "type": _best_type(samples[j]),
            }
        )

    return jsonify({
        "ok": True,
        "source_name": source_name,
        "received_rows": len(rows),
        "upserted_rows": len(rows),
        "max_rows_per_call": MAX_ROWS,
        "schema": {
            "fields": schema_fields
        }
    })