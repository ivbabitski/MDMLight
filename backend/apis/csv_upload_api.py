import os
import csv
import io
import sqlite3
from datetime import datetime

from flask import Blueprint, request, jsonify

from db.sqlite_db import get_conn, init_source_input


bp = Blueprint("csv_upload_api", __name__)

MAX_UPLOAD_BYTES = int(os.environ.get("INGEST_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_ROWS = int(os.environ.get("INGEST_CSV_MAX_ROWS", "200000"))

def _require_user_id():
    v = request.headers.get("X-User-Id") or ""
    v = str(v).strip()
    if not v:
        return None, (jsonify({"error": "X-User-Id header is required"}), 400)
    try:
        return int(v), None
    except Exception:
        return None, (jsonify({"error": "X-User-Id must be an integer"}), 400)


def _norm(h: str) -> str:
    s = str(h or "").strip().lower()
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    k = "".join(out)
    while "__" in k:
        k = k.replace("__", "_")
    return k.strip("_")


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


@bp.post("/ingest/csv")
def ingest_csv():
    if request.content_length and request.content_length > (MAX_UPLOAD_BYTES + 1024 * 1024):
        return jsonify({"error": f"file too large (max {MAX_UPLOAD_BYTES} bytes)"}), 413

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "file is required (multipart/form-data field: file)"}), 400

    try:
        file.stream.seek(0, os.SEEK_END)
        size = file.stream.tell()
        file.stream.seek(0)
    except Exception:
        size = None

    if size is not None and size > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"file too large (max {MAX_UPLOAD_BYTES} bytes)"}), 413


    app_user_id, err = _require_user_id()
    if err:
        return err

    init_source_input()


    text = io.TextIOWrapper(file.stream, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text)

    headers = reader.fieldnames or []
    if not headers:
        return jsonify({"error": "CSV header row is required"}), 400

    norm_map = {h: _norm(h) for h in headers}
    norm_to_orig = {}
    for h in headers:
        nh = norm_map.get(h) or ""
        if nh and nh not in norm_to_orig:
            norm_to_orig[nh] = h

    required = ["source_id", "source_name", "created_at", "created_by"]
    missing = [k for k in required if k not in norm_to_orig]
    if missing:
        return jsonify({"error": "missing required columns", "missing": missing}), 400

    reserved = {"source_id", "source_name", "created_at", "created_by", "updated_at", "updated_by", "app_user_id"}

    flex_headers = [h for h in headers if (norm_map.get(h) or "") not in reserved]
    if len(flex_headers) > 20:
        return jsonify({"error": f"too many data columns: {len(flex_headers)} (max 20 flex fields)"}), 400

    flex_codes = [f"f{i + 1:02d}" for i in range(len(flex_headers))]
    flex_samples = {c: [] for c in flex_codes}

    def _clean(v):
        s = "" if v is None else str(v)
        s = s.strip()
        return s if s else None

    src_id_col = norm_to_orig["source_id"]
    src_name_col = norm_to_orig["source_name"]
    created_at_col = norm_to_orig["created_at"]
    created_by_col = norm_to_orig["created_by"]
    updated_at_col = norm_to_orig.get("updated_at")
    updated_by_col = norm_to_orig.get("updated_by")

    cols = [
        "source_id",
        "source_name",
        *[f"f{i:02d}" for i in range(1, 21)],
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "app_user_id",
    ]

    placeholders = ",".join(["?"] * (2 + 20 + 4 + 1))
    update_set = ", ".join(
        [
            "source_name=excluded.source_name",
            *[f"f{i:02d}=excluded.f{i:02d}" for i in range(1, 21)],
            "created_at=excluded.created_at",
            "created_by=excluded.created_by",
            "updated_at=excluded.updated_at",
            "updated_by=excluded.updated_by",
            "app_user_id=excluded.app_user_id",
        ]
    )


    insert_sql = f"""
    INSERT INTO source_input (
      {",".join(cols)}
    ) VALUES (
      {placeholders}
    )
    ON CONFLICT(source_id) DO UPDATE SET
      {update_set}
    WHERE source_input.app_user_id IS NULL OR source_input.app_user_id = excluded.app_user_id;
    """

    params = []
    row_count = 0
    seen_source_ids = set()
    unique_source_ids = 0
    for i, r in enumerate(reader):
        if r is None:
            continue
        row_count += 1
        if row_count > MAX_ROWS:
            return jsonify({"error": f"too many rows: {row_count} (max {MAX_ROWS})"}), 400

        source_id = _clean(r.get(src_id_col))
        source_name = _clean(r.get(src_name_col))
        created_at = _clean(r.get(created_at_col))
        created_by = _clean(r.get(created_by_col))
        updated_at = _clean(r.get(updated_at_col)) if updated_at_col else None
        updated_by = _clean(r.get(updated_by_col)) if updated_by_col else None

        if not source_id:
            return jsonify({"error": f"row {i + 2}: source_id is required"}), 400

        if source_id not in seen_source_ids:
            seen_source_ids.add(source_id)
            unique_source_ids += 1

        if not source_name:
            return jsonify({"error": f"row {i + 2}: source_name is required"}), 400
        if not created_at:
            return jsonify({"error": f"row {i + 2}: created_at is required"}), 400
        if not created_by:
            return jsonify({"error": f"row {i + 2}: created_by is required"}), 400

        flex_values = []
        for j, h in enumerate(flex_headers):
            v = _clean(r.get(h))
            flex_values.append(v)
            if v is not None and j < len(flex_codes) and len(flex_samples[flex_codes[j]]) < 50:
                flex_samples[flex_codes[j]].append(v)

        padded = (flex_values + [None] * 20)[:20]

        params.append(
            (
                source_id,
                source_name,
                *padded,
                created_at,
                created_by,
                updated_at,
                updated_by,
                app_user_id,
            )
        )

    try:
        with get_conn() as conn:
            conn.execute("DELETE FROM source_input WHERE app_user_id=?", (app_user_id,))
            conn.executemany(insert_sql, params)
    except sqlite3.OperationalError as e:
        return jsonify({"error": "sqlite error", "detail": str(e)}), 500



    # Detect cross-user conflicts or duplicate source_id rows (unique_source_ids is per file)
    try:
        with get_conn() as conn:
            nrow = conn.execute(
                "SELECT COUNT(*) AS n FROM source_input WHERE app_user_id=?",
                (app_user_id,),
            ).fetchone()
            try:
                inserted_n = int(nrow["n"] or 0)
            except Exception:
                inserted_n = int(nrow[0] or 0)
    except Exception:
        inserted_n = None

    if inserted_n is not None and inserted_n != unique_source_ids:
        return jsonify({
            "error": "source_id conflict across users or duplicate ids",
            "detail": "Some rows could not be written because source_id already exists for another user (or duplicated in the file).",
            "expected_unique_source_ids": unique_source_ids,
            "written_rows_for_user": inserted_n,
            "app_user_id": app_user_id,
        }), 409

    schema_fields = []
    for j, h in enumerate(flex_headers):
        code = f"f{j + 1:02d}"
        schema_fields.append({"code": code, "header": h, "type": _best_type(flex_samples[code])})

    return jsonify(
        {
            "ok": True,
            "received_rows": row_count,
            "upserted_rows": row_count,
            "headers": headers,
            "schema": {
                "required": [
                    {"code": "source_id", "header": src_id_col, "type": "id"},
                    {"code": "source_name", "header": src_name_col, "type": "text"},
                ],
                "reserved": [
                    {"code": "created_at", "header": created_at_col, "type": "date"},
                    {"code": "created_by", "header": created_by_col, "type": "text"},
                    {"code": "updated_at", "header": updated_at_col or "updated_at", "type": "date"},
                    {"code": "updated_by", "header": updated_by_col or "updated_by", "type": "text"},
                ],
                "fields": schema_fields,
            },
            "mapping": [
                {"header": h, "code": f"f{idx + 1:02d}"} for idx, h in enumerate(flex_headers)
            ],
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_rows": MAX_ROWS,
        }
    )