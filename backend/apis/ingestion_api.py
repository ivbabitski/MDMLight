import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from services.services import ServiceError, source_input_ingest_batch_from_fields_array

DB_PATH = os.environ.get("DB_PATH", "/data/app.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

bp = Blueprint("ingestion_api", __name__)

MAX_ROWS = int(os.environ.get("INGEST_MAX_ROWS", "5000"))

def _require_app_user_id(payload: dict):
    v = payload.get("app_user_id") or request.headers.get("X-User-Id") or ""
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

    if not isinstance(payload, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    app_user_id, err = _require_app_user_id(payload)
    if err:
        return err

    headers = payload.get("headers") or payload.get("field_names")

    try:
        out = source_input_ingest_batch_from_fields_array(
            app_user_id=app_user_id,
            source_name=payload.get("source_name"),
            actor=payload.get("actor"),
            rows=payload.get("rows"),
            headers=headers,
            max_rows_per_call=MAX_ROWS,
        )
        return jsonify(out)
    except ServiceError as e:
        resp = {"error": str(e)}
        try:
            for k, v in (e.payload or {}).items():
                if k in resp:
                    continue
                resp[k] = v
        except Exception:
            pass
        return jsonify(resp), getattr(e, "status_code", 500)
    except Exception:
        return jsonify({"error": "internal error"}), 500
