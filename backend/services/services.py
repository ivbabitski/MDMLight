"""
Service-layer functions for the app.

Goal:
- Keep framework (Flask) route files as thin "doorways" (parse inputs + call services + return response).
- Put business logic and DB logic in here so it is reusable across APIs, CLI jobs, tests, etc.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from db.sqlite_db import get_conn, init_source_input
except Exception:
    from sqlite_db import get_conn, init_source_input


__all__ = [
    "ServiceError",
    "ValidationError",
    "NotFoundError",
    "ForbiddenError",
    "UnauthorizedError",
    "ConflictError",
    "PayloadTooLargeError",
    "source_input_get_summary",
    "match_enqueue_job",
    "match_get_status",
    "match_clear_recon_cluster",
    "match_clear_golden_record",
    "source_input_ingest_csv",
    "auth_register",
    "auth_login",
    "auth_recover",
    "cleanup_recon_cluster",
    "cleanup_golden_record",
    "recon_cluster_list_cluster_records",
    "source_systems_health",
    "source_systems_list_pairs",
    "mdm_source_systems_list",
    "recon_cluster_list_records",
    "golden_record_list_records",
    "source_input_ingest_batch_from_fields_array",
    "source_input_ingest_batch",
    "mdm_model_list",
    "mdm_model_get",
    "mdm_model_create",
    "mdm_model_update",
    "mdm_model_delete",
    "matching_summary_get",
    "api_key_rotate",
    "api_key_revoke",
    "api_ingest_api",
    "api_batch_get",
]


class ServiceError(Exception):
    """
    Base class for service-layer exceptions.
    API routes can catch these later and convert into HTTP responses.
    """

    status_code: int = 500

    def __init__(self, message: str, *, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload: Dict[str, Any] = payload or {}


class ValidationError(ServiceError):
    status_code = 400


class NotFoundError(ServiceError):
    status_code = 404


class ForbiddenError(ServiceError):
    status_code = 403


class UnauthorizedError(ServiceError):
    status_code = 401


class ConflictError(ServiceError):
    status_code = 409


class PayloadTooLargeError(ServiceError):
    status_code = 413


def _row_get(row: Any, key: str, idx: int) -> Any:
    """
    Safely read sqlite rows that might be dict-like (row["col"]) or tuple-like (row[i]).
    """
    if row is None:
        return None

    try:
        if hasattr(row, "keys"):
            return row[key]
    except Exception:
        pass

    try:
        return row[idx]
    except Exception:
        return None


def _build_code_to_label_map(model_cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping like {"f01": "Customer Name", ...} from model config json.
    Supports both shapes:
      - { "config": { "fields": [ ... ] } }
      - { "fields": [ ... ] }
    """
    code_to_label: Dict[str, str] = {}

    cfg_obj = model_cfg.get("config") if isinstance(model_cfg, dict) else None
    if isinstance(cfg_obj, dict):
        fields_arr = cfg_obj.get("fields")
    else:
        fields_arr = model_cfg.get("fields") if isinstance(model_cfg, dict) else None

    if not isinstance(fields_arr, list):
        return code_to_label

    for f in fields_arr:
        if not isinstance(f, dict):
            continue

        code = str(f.get("code") or "").strip()
        label = str(f.get("label") or "").strip()
        if not code or not label:
            continue

        c_lo = code.lower()
        l_lo = label.lower()

        if l_lo == c_lo:
            continue
        if l_lo.startswith("f") and l_lo[1:].isdigit():
            continue

        code_to_label[code] = label

    return code_to_label


def _get_db_file(conn: Any) -> str:
    """
    Returns the sqlite main database file path (best-effort).
    """
    db_file = ""
    try:
        db_list = conn.execute("PRAGMA database_list").fetchall()
        for r in db_list:
            try:
                name = r["name"]
                file = r["file"]
            except Exception:
                name = r[1]
                file = r[2]
            if name == "main":
                db_file = str(file or "")
                break
    except Exception:
        db_file = ""
    return db_file


def _get_table_cols(conn: Any, table_name: str) -> List[str]:
    """
    Returns a list of columns for table_name (best-effort).
    """
    table_cols: List[str] = []
    try:
        info_rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        for r in info_rows:
            try:
                col = r["name"]
            except Exception:
                col = r[1]
            if col:
                table_cols.append(str(col))
    except Exception:
        table_cols = []
    return table_cols


def _detect_f_cols(table_cols: List[str]) -> List[str]:
    """
    Detect real f* columns in the source_input table (handles f1 vs f01, etc).
    """
    f_cols: List[str] = []
    seen = set()

    for c in table_cols:
        if not c.startswith("f"):
            continue
        tail = c[1:]
        if not tail.isdigit():
            continue
        if c in seen:
            continue
        seen.add(c)
        f_cols.append(c)

    f_cols.sort(key=lambda x: int(x[1:]))
    return f_cols


def _compute_field_stats(
    conn: Any,
    *,
    app_user_id: int,
    field_keys: List[str],
    table_cols: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute non-empty counts per field for the given user.
    Only computes for columns that actually exist.
    """
    existing = set(table_cols)
    stat_keys = [k for k in field_keys if k in existing]

    if not stat_keys:
        return [{"key": k, "non_empty": 0} for k in field_keys]

    exprs = ", ".join(
        [
            f'SUM(CASE WHEN TRIM(COALESCE("{k}", \'\')) <> \'\' THEN 1 ELSE 0 END) AS "{k}"'
            for k in stat_keys
        ]
    )

    stats_row = conn.execute(
        f"SELECT {exprs} FROM source_input WHERE app_user_id=?",
        (app_user_id,),
    ).fetchone()

    out: List[Dict[str, Any]] = []
    for k in field_keys:
        if k not in existing:
            out.append({"key": k, "non_empty": 0})
            continue

        try:
            out.append({"key": k, "non_empty": int(stats_row[k] or 0)})
        except Exception:
            out.append({"key": k, "non_empty": 0})

    return out


def source_input_get_summary(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    """
    Service equivalent of the current /source-input/summary endpoint logic.

    Inputs:
      - app_user_id: int
      - model_id: str (required)

    Returns:
      A dict ready to jsonify().
    Raises:
      ValidationError / NotFoundError / ForbiddenError
    """
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValidationError("model_id is required")

    init_source_input()

    with get_conn() as conn:
        row = None
        try:
            row = conn.execute(
                """
                SELECT id, config_json, owner_user_id, app_user_id
                FROM mdm_models
                WHERE id = ?
                LIMIT 1
                """,
                (model_id,),
            ).fetchone()
        except Exception:
            row = None

        if not row:
            raise NotFoundError("model not found")

        owner_id = _row_get(row, "owner_user_id", 2)
        model_app_user_id = _row_get(row, "app_user_id", 3)

        if owner_id is not None and int(owner_id) != int(app_user_id):
            raise ForbiddenError("model does not belong to user")
        if model_app_user_id is not None and int(model_app_user_id) != int(app_user_id):
            raise ForbiddenError("model does not belong to user")

        cfg_raw = _row_get(row, "config_json", 1)
        try:
            model_cfg = json.loads(cfg_raw or "{}") if cfg_raw else {}
        except Exception:
            model_cfg = {}

        code_to_label = _build_code_to_label_map(model_cfg)

        db_file = _get_db_file(conn)
        table_cols = _get_table_cols(conn, "source_input")

        f_cols = _detect_f_cols(table_cols)

        field_keys = f_cols if f_cols else [f"f{str(i).zfill(2)}" for i in range(1, 21)]

        total_records = 0
        total_row = conn.execute(
            "SELECT COUNT(*) AS n FROM source_input WHERE app_user_id=?",
            (app_user_id,),
        ).fetchone()
        try:
            total_records = int(total_row["n"] or 0)
        except Exception:
            total_records = int(total_row[0] or 0)

        sources: List[Dict[str, Any]] = []
        src_rows = conn.execute(
            """
            SELECT source_name AS source, COUNT(*) AS count
            FROM source_input
            WHERE app_user_id = ?
            GROUP BY source_name
            ORDER BY count DESC, source ASC
            """,
            (app_user_id,),
        ).fetchall()

        for r in src_rows:
            try:
                src = r["source"]
                cnt = r["count"]
            except Exception:
                src = r[0]
                cnt = r[1]
            sources.append({"source": str(src or ""), "count": int(cnt or 0)})

        field_stats = _compute_field_stats(
            conn,
            app_user_id=app_user_id,
            field_keys=field_keys,
            table_cols=table_cols,
        )

    fields_with_data = sum(1 for x in field_stats if int(x.get("non_empty") or 0) > 0)

    labeled_fields: List[Dict[str, Any]] = []
    for fs in field_stats:
        k = str(fs.get("key") or "").strip()
        if not k:
            continue
        lbl = code_to_label.get(k)
        if not lbl:
            continue
        labeled_fields.append(
            {"code": k, "label": lbl, "non_empty": int(fs.get("non_empty") or 0)}
        )

    field_pills = [x["label"] for x in labeled_fields]

    return {
        "total_records": total_records,
        "sources": sources,
        "field_keys": field_keys,
        "fields_total": len(field_keys),
        "fields_with_data": fields_with_data,
        "field_stats": field_stats,
        "labeled_fields": labeled_fields,
        "field_pills": field_pills,
        "model_id": model_id,
        "db_file": db_file,
        "table_cols": table_cols,
    }


# ---------------------------------------------------------------------------
# Match services (moved from match_api.py)
# ---------------------------------------------------------------------------

def _init_all_tables() -> None:
    try:
        from db.sqlite_db import init_all_tables
    except Exception:
        from sqlite_db import init_all_tables
    init_all_tables()


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def match_enqueue_job(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValidationError("model_id is required")

    import uuid

    _init_all_tables()

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

    return {"ok": True, "job_id": job_id, "status": "queued", "model_id": model_id}


def match_get_status(*, app_user_id: int, job_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValidationError("job_id is required")

    _init_all_tables()

    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM match_job WHERE job_id=? AND app_user_id=?",
            (job_id, app_user_id),
        ).fetchone()
        if not row:
            raise NotFoundError("job_id not found", payload={"job_id": job_id})
        return dict(row)


def match_clear_recon_cluster(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValidationError("model_id is required")

    _init_all_tables()

    with get_conn() as conn:
        cur1 = conn.execute(
            "DELETE FROM recon_cluster WHERE app_user_id=? AND model_id=?",
            (app_user_id, model_id),
        )
        cur2 = conn.execute(
            "DELETE FROM cluster_map WHERE app_user_id=? AND model_id=?",
            (app_user_id, model_id),
        )

    return {
        "ok": True,
        "app_user_id": app_user_id,
        "model_id": model_id,
        "recon_cluster_deleted": cur1.rowcount,
        "cluster_map_deleted": cur2.rowcount,
    }


def match_clear_golden_record(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValidationError("model_id is required")

    _init_all_tables()

    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM golden_record WHERE app_user_id=? AND model_id=?",
            (app_user_id, model_id),
        )

    return {
        "ok": True,
        "app_user_id": app_user_id,
        "model_id": model_id,
        "golden_record_deleted": cur.rowcount,
    }


# ---------------------------------------------------------------------------
# CSV ingest services (moved from csv_upload_api.py)
# ---------------------------------------------------------------------------

def _csv_norm(h: str) -> str:
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


def _csv_looks_like_email(s: str) -> bool:
    if " " in s:
        return False
    if "@" not in s:
        return False
    left, right = s.split("@", 1)
    if not left or not right:
        return False
    return "." in right


def _csv_looks_like_phone(s: str) -> bool:
    if any(ch.isalpha() for ch in s):
        return False
    digits = sum(1 for ch in s if ch.isdigit())
    return 7 <= digits <= 15


def _csv_looks_like_number(s: str) -> bool:
    if any(ch.isalpha() for ch in s):
        return False
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False


def _csv_looks_like_date(s: str) -> bool:
    from datetime import datetime

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


def _csv_guess_type(v: str) -> str:
    s = str(v or "").strip()
    if not s:
        return "text"
    if _csv_looks_like_email(s):
        return "email"
    if _csv_looks_like_phone(s):
        return "phone"
    if _csv_looks_like_date(s):
        return "date"
    if _csv_looks_like_number(s):
        return "number"
    return "text"


def _csv_best_type(samples) -> str:
    counts = {"email": 0, "phone": 0, "date": 0, "number": 0, "text": 0}
    for v in samples:
        counts[_csv_guess_type(v)] += 1

    best = max(counts.values()) if counts else 0
    if best <= 0:
        return "text"

    for t in ("email", "phone", "date", "number", "text"):
        if counts.get(t, 0) == best:
            return t
    return "text"


def source_input_ingest_csv(
    *,
    app_user_id: int,
    file: Any,
    content_length: Optional[int] = None,
    max_upload_bytes: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    import os
    import csv
    import io
    import sqlite3

    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    if max_upload_bytes is None:
        max_upload_bytes = int(
            os.environ.get("INGEST_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))
        )
    if max_rows is None:
        max_rows = int(os.environ.get("INGEST_CSV_MAX_ROWS", "200000"))

    if content_length and content_length > (max_upload_bytes + 1024 * 1024):
        raise PayloadTooLargeError(
            f"file too large (max {max_upload_bytes} bytes)",
            payload={"max_upload_bytes": max_upload_bytes},
        )

    if not file:
        raise ValidationError("file is required (multipart/form-data field: file)")

    stream = file.stream if hasattr(file, "stream") else file

    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
    except Exception:
        size = None

    if size is not None and size > max_upload_bytes:
        raise PayloadTooLargeError(
            f"file too large (max {max_upload_bytes} bytes)",
            payload={"max_upload_bytes": max_upload_bytes},
        )

    init_source_input()

    text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text)

    headers = reader.fieldnames or []
    if not headers:
        raise ValidationError("CSV header row is required")

    norm_map = {h: _csv_norm(h) for h in headers}
    norm_to_orig = {}
    for h in headers:
        nh = norm_map.get(h) or ""
        if nh and nh not in norm_to_orig:
            norm_to_orig[nh] = h

    required = ["source_id", "source_name", "created_at", "created_by"]
    missing = [k for k in required if k not in norm_to_orig]
    if missing:
        raise ValidationError("missing required columns", payload={"missing": missing})

    reserved = {
        "source_id",
        "source_name",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "app_user_id",
    }

    flex_headers = [h for h in headers if (norm_map.get(h) or "") not in reserved]
    if len(flex_headers) > 20:
        raise ValidationError(
            f"too many data columns: {len(flex_headers)} (max 20 flex fields)"
        )

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
    ON CONFLICT(app_user_id, source_name, source_id) DO UPDATE SET
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
        if row_count > max_rows:
            raise ValidationError(f"too many rows: {row_count} (max {max_rows})")

        source_id = _clean(r.get(src_id_col))
        source_name = _clean(r.get(src_name_col))
        created_at = _clean(r.get(created_at_col))
        created_by = _clean(r.get(created_by_col))
        updated_at = _clean(r.get(updated_at_col)) if updated_at_col else None
        updated_by = _clean(r.get(updated_by_col)) if updated_by_col else None

        if not source_id:
            raise ValidationError(f"row {i + 2}: source_id is required")

        if source_id not in seen_source_ids:
            seen_source_ids.add(source_id)
            unique_source_ids += 1

        if not source_name:
            raise ValidationError(f"row {i + 2}: source_name is required")
        if not created_at:
            raise ValidationError(f"row {i + 2}: created_at is required")
        if not created_by:
            raise ValidationError(f"row {i + 2}: created_by is required")

        flex_values = []
        for j, h in enumerate(flex_headers):
            v = _clean(r.get(h))
            flex_values.append(v)
            if (
                v is not None
                and j < len(flex_codes)
                and len(flex_samples[flex_codes[j]]) < 50
            ):
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
        raise ServiceError("sqlite error", payload={"detail": str(e)})

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
        raise ConflictError(
            "source_id conflict across users or duplicate ids",
            payload={
                "detail": "Some rows could not be written because source_id already exists for another user (or duplicated in the file).",
                "expected_unique_source_ids": unique_source_ids,
                "written_rows_for_user": inserted_n,
                "app_user_id": app_user_id,
            },
        )

    schema_fields = []
    for j, h in enumerate(flex_headers):
        code = f"f{j + 1:02d}"
        schema_fields.append(
            {"code": code, "header": h, "type": _csv_best_type(flex_samples[code])}
        )

    return {
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
                {
                    "code": "updated_at",
                    "header": updated_at_col or "updated_at",
                    "type": "date",
                },
                {
                    "code": "updated_by",
                    "header": updated_by_col or "updated_by",
                    "type": "text",
                },
            ],
            "fields": schema_fields,
        },
        "mapping": [
            {"header": h, "code": f"f{idx + 1:02d}"} for idx, h in enumerate(flex_headers)
        ],
        "max_upload_bytes": max_upload_bytes,
        "max_rows": max_rows,
    }


def _auth_imports():
    try:
        from db.sqlite_db import (
            create_user,
            get_user_by_username,
            init_users,
            update_user_password,
        )
    except Exception:
        from sqlite_db import (
            create_user,
            get_user_by_username,
            init_users,
            update_user_password,
        )

    return create_user, get_user_by_username, init_users, update_user_password


def _auth_send_email(to_email: str, subject: str, body: str) -> None:
    import os
    import smtplib
    from email.message import EmailMessage

    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = os.environ.get("SMTP_PORT")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_from = os.environ.get("SMTP_FROM")

    if not smtp_host or not smtp_port or not smtp_user or not smtp_pass or not smtp_from:
        raise RuntimeError(
            "SMTP not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM."
        )

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def auth_register(*, username: str, email: str, password: str) -> Dict[str, Any]:
    from werkzeug.security import generate_password_hash

    create_user, get_user_by_username, init_users, _ = _auth_imports()
    init_users()

    u = str(username or "").strip()
    e = str(email or "").strip()
    p = str(password or "")

    if not u or not e or not p:
        raise ValidationError("username, email, password are required")

    password_hash = generate_password_hash(p)

    try:
        create_user(u, e, password_hash)
    except Exception as ex:
        msg = str(ex).lower()
        if "unique" in msg or "constraint" in msg:
            raise ConflictError("username or email already exists")
        raise ServiceError("register failed")

    user_row = get_user_by_username(u)
    user_id_val = _row_get(user_row, "id", 0) if user_row else None
    user_id = int(user_id_val) if user_id_val is not None else None

    return {"ok": True, "user_id": user_id}


def auth_login(*, username: str, password: str) -> Dict[str, Any]:
    from werkzeug.security import check_password_hash

    _, get_user_by_username, init_users, _ = _auth_imports()
    init_users()

    u = str(username or "").strip()
    p = str(password or "")

    if not u or not p:
        raise ValidationError("username and password are required")

    user = get_user_by_username(u)
    if not user:
        raise UnauthorizedError("invalid credentials")

    pw_hash = _row_get(user, "password_hash", 3)
    if not pw_hash or not check_password_hash(str(pw_hash), p):
        raise UnauthorizedError("invalid credentials")

    user_id = int(_row_get(user, "id", 0))
    uname = str(_row_get(user, "username", 1) or "")

    return {"ok": True, "user_id": user_id, "username": uname}


def auth_recover(*, username: str, email: str) -> Dict[str, Any]:
    import secrets
    from werkzeug.security import generate_password_hash

    _, get_user_by_username, init_users, update_user_password = _auth_imports()
    init_users()

    u = str(username or "").strip()
    e = str(email or "").strip()

    if not u or not e:
        raise ValidationError("username and email are required")

    user = get_user_by_username(u)
    if not user:
        raise UnauthorizedError("invalid credentials")

    user_email = str(_row_get(user, "email", 2) or "").strip()
    if user_email.lower() != e.lower():
        raise UnauthorizedError("invalid credentials")

    temp_password = secrets.token_urlsafe(9)
    password_hash = generate_password_hash(temp_password)

    try:
        if not update_user_password(u, password_hash):
            raise ServiceError("recovery failed")

        _auth_send_email(
            user_email,
            "MDM Light password reset",
            f"Your temporary password is: {temp_password}\n\nLog in and change it.",
        )
    except ServiceError:
        raise
    except Exception as ex:
        raise ServiceError(str(ex))

    return {"ok": True}


# ---------------------------------------------------------------------------
# Cleanup services (moved from cleanup_recon_cluster_api.py / cleanup_golden_record_api.py)
# ---------------------------------------------------------------------------

def cleanup_recon_cluster(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    _init_all_tables()

    with get_conn() as conn:
        model_row = conn.execute(
            "SELECT model_name FROM mdm_models WHERE id = ? AND app_user_id = ? AND deleted_at IS NULL LIMIT 1",
            (mid, app_user_id),
        ).fetchone()
        model_name = str(_row_get(model_row, "model_name", 0) or "").strip() if model_row else ""

        if model_name:
            cur = conn.execute(
                "DELETE FROM recon_cluster WHERE app_user_id = ? AND (model_id = ? OR model_name = ?)",
                (app_user_id, mid, model_name),
            )
        else:
            cur = conn.execute(
                "DELETE FROM recon_cluster WHERE app_user_id = ? AND model_id = ?",
                (app_user_id, mid),
            )
        deleted = cur.rowcount

        cur_cm = conn.execute(
            "DELETE FROM cluster_map WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, mid),
        )
        cluster_map_deleted = cur_cm.rowcount

        cur_me = conn.execute(
            "DELETE FROM match_exception WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, mid),
        )
        match_exception_deleted = cur_me.rowcount

        if model_name:
            remaining_row = conn.execute(
                "SELECT COUNT(1) AS n FROM recon_cluster WHERE app_user_id = ? AND (model_id = ? OR model_name = ?)",
                (app_user_id, mid, model_name),
            ).fetchone()
        else:
            remaining_row = conn.execute(
                "SELECT COUNT(1) AS n FROM recon_cluster WHERE app_user_id = ? AND model_id = ?",
                (app_user_id, mid),
            ).fetchone()

        remaining = int(_row_get(remaining_row, "n", 0) or 0) if remaining_row else 0

    return {
        "ok": True,
        "table": "recon_cluster",
        "app_user_id": app_user_id,
        "model_id": mid,
        "model_name": model_name,
        "deleted": deleted,
        "remaining": remaining,
        "cluster_map_deleted": cluster_map_deleted,
        "match_exception_deleted": match_exception_deleted,
    }


def cleanup_golden_record(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    _init_all_tables()

    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM golden_record WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, mid),
        )
        deleted = cur.rowcount

        remaining_row = conn.execute(
            "SELECT COUNT(1) AS n FROM golden_record WHERE app_user_id = ? AND model_id = ?",
            (app_user_id, mid),
        ).fetchone()
        remaining = int(_row_get(remaining_row, "n", 0) or 0) if remaining_row else 0

    return {
        "ok": True,
        "table": "golden_record",
        "app_user_id": app_user_id,
        "model_id": mid,
        "deleted": deleted,
        "remaining": remaining,
    }


def golden_record_list_records(
    *,
    app_user_id: int,
    model_id: str,
    limit: int = 5000,
    offset: int = 0,
) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    if not isinstance(limit, int):
        raise ValidationError("limit must be an integer")
    if not isinstance(offset, int):
        raise ValidationError("offset must be an integer")

    if limit <= 0:
        raise ValidationError("limit must be >= 1")
    if limit > 5000:
        raise ValidationError("limit must be <= 5000")
    if offset < 0:
        raise ValidationError("offset must be >= 0")

    try:
        try:
            from db.sqlite_db import init_golden_record
        except Exception:
            from sqlite_db import init_golden_record
        init_golden_record()
    except Exception as ex:
        raise ServiceError(f"init_golden_record failed: {ex.__class__.__name__}: {ex}")

    f_cols: List[str] = []
    cols: List[str] = []
    col_names: List[str] = []
    rows: List[Any] = []

    try:
        with get_conn() as conn:
            model_row = None
            try:
                model_row = conn.execute(
                    "SELECT config_json FROM mdm_models WHERE id = ? LIMIT 1",
                    (mid,),
                ).fetchone()
            except Exception:
                model_row = None

            cfg_raw = _row_get(model_row, "config_json", 0) if model_row else None
            try:
                model_cfg = json.loads(cfg_raw or "{}") if cfg_raw else {}
            except Exception:
                model_cfg = {}

            code_to_label = _build_code_to_label_map(model_cfg)

            used_f_cols: List[str] = []
            for code in code_to_label.keys():
                c = str(code or "").strip().lower()
                if not c.startswith("f"):
                    continue
                tail = c[1:]
                if not tail.isdigit():
                    continue
                n = int(tail)
                if n < 1 or n > 20:
                    continue
                used_f_cols.append(f"f{n:02d}")

            f_cols = sorted({c for c in used_f_cols}, key=lambda x: int(x[1:]))

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

            col_names = [
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
            params: List[Any] = [app_user_id, mid, limit, offset]

            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
    except Exception as ex:
        raise ServiceError(f"golden_record query failed: {ex.__class__.__name__}: {ex}")

    out: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            out.append({k: r[k] for k in r.keys()})
        else:
            d = {}
            n = min(len(col_names), len(r))
            for i in range(n):
                d[col_names[i]] = r[i]
            out.append(d)

    return {"records": out, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Recon cluster: cluster records service (moved from cluster_records_by_cluster_id_api.py)
# ---------------------------------------------------------------------------

def _init_recon_cluster() -> None:
    try:
        from db.sqlite_db import init_recon_cluster
    except Exception:
        from sqlite_db import init_recon_cluster
    init_recon_cluster()


def recon_cluster_list_cluster_records(
    *,
    app_user_id: int,
    model_id: str,
    cluster_id: str = "",
    status_mode: str = "match",
    limit: int = 5000,
    offset: int = 0,
) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    cid = str(cluster_id or "").strip()
    cluster_filter = bool(cid)

    status = str(status_mode or "match").strip().lower()
    if status in {"match", "matches"}:
        status_mode_norm = "match"
    elif status in {"exception", "exceptions"}:
        status_mode_norm = "exceptions"
    else:
        raise ValidationError("status_mode must be one of: match | exceptions")

    if not isinstance(limit, int):
        raise ValidationError("limit must be an integer")
    if not isinstance(offset, int):
        raise ValidationError("offset must be an integer")

    if limit <= 0:
        raise ValidationError("limit must be >= 1")
    if limit > 5000:
        raise ValidationError("limit must be <= 5000")
    if offset < 0:
        raise ValidationError("offset must be >= 0")

    try:
        _init_recon_cluster()
    except Exception as ex:
        raise ServiceError(f"init_recon_cluster failed: {ex.__class__.__name__}: {ex}")

    f_cols: List[str] = []
    cols: List[str] = []
    col_names: List[str] = []
    rows: List[Any] = []

    try:
        with get_conn() as conn:
            f_cols = [f"f{str(i).zfill(2)}" for i in range(1, 21)]

            if status_mode_norm == "exceptions":
                model_row = None
                try:
                    model_row = conn.execute(
                        "SELECT config_json FROM mdm_models WHERE id = ? LIMIT 1",
                        (mid,),
                    ).fetchone()
                except Exception:
                    model_row = None

                cfg_raw = _row_get(model_row, "config_json", 0) if model_row else None
                try:
                    model_cfg = json.loads(cfg_raw or "{}") if cfg_raw else {}
                except Exception:
                    model_cfg = {}

                code_to_label = _build_code_to_label_map(model_cfg)

                used_f_cols: List[str] = []
                for code in code_to_label.keys():
                    c = str(code or "").strip().lower()
                    if not c.startswith("f"):
                        continue
                    tail = c[1:]
                    if not tail.isdigit():
                        continue
                    n = int(tail)
                    if n < 1 or n > 20:
                        continue
                    used_f_cols.append(f"f{n:02d}")

                f_cols = sorted({c for c in used_f_cols}, key=lambda x: int(x[1:]))

            cols = [
                "rowid AS id",
                "job_id",
                "cluster_id",
                "record_id",
                "model_id",
                "model_name",
                "source_name",
                "source_id",
                "match_status",
                "match_score",
                "match_threshold",
                "exceptions_threshold",
                "created_at",
                "created_by",
                "updated_at",
                "updated_by",
                *f_cols,
            ]

            col_names = [
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
                "match_threshold",
                "exceptions_threshold",
                "created_at",
                "created_by",
                "updated_at",
                "updated_by",
                *f_cols,
            ]

            if cluster_filter:
                sql = (
                    f"SELECT {', '.join(cols)} "
                    "FROM recon_cluster "
                    "WHERE app_user_id = ? AND model_id = ? AND cluster_id = ? "
                    "ORDER BY (match_score IS NULL) ASC, match_score DESC, id ASC "
                    "LIMIT ? OFFSET ?"
                )
                params: List[Any] = [app_user_id, mid, cid, limit, offset]
            else:
                where_parts = ["app_user_id = ?", "model_id = ?"]
                params = [app_user_id, mid]

                if status_mode_norm == "exceptions":
                    where_parts.append("lower(trim(coalesce(match_status, ''))) = 'exception'")

                where_sql = " AND ".join(where_parts)


                sql = (
                    f"SELECT {', '.join(cols)} "
                    "FROM recon_cluster "
                    f"WHERE {where_sql} "
                    "ORDER BY cluster_id ASC, (match_score IS NULL) ASC, match_score DESC, id ASC "
                    "LIMIT ? OFFSET ?"
                )
                params.extend([limit, offset])

            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
    except Exception as ex:
        raise ServiceError(f"recon_cluster query failed: {ex.__class__.__name__}: {ex}")

    out: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            out.append({k: r[k] for k in r.keys()})
        else:
            d = {}
            n = min(len(col_names), len(r))
            for i in range(n):
                d[col_names[i]] = r[i]
            out.append(d)

    return {"records": out, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# System / Source systems services (moved from system_api.py / source_systems_api.py)
# ---------------------------------------------------------------------------

def source_systems_health() -> Dict[str, Any]:
    return {"ok": True}


def source_systems_list_pairs(*, app_user_id: int) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    init_source_input()

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT source_id, source_name
            FROM source_input
            WHERE app_user_id = ?
            ORDER BY source_name ASC, source_id ASC
            """,
            (app_user_id,),
        ).fetchall()

    items: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            items.append({"source_id": r["source_id"], "source_name": r["source_name"]})
        else:
            items.append({"source_id": r[0], "source_name": r[1]})

    return {"items": items, "count": len(items)}


def mdm_source_systems_list(
    *,
    app_user_id: int,
    include_stats: bool = True,
    since: str = "",
    until: str = "",
) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    if not isinstance(include_stats, bool):
        raise ValidationError("include_stats must be a boolean")

    since_v = str(since or "").strip()
    until_v = str(until or "").strip()

    try:
        from db.sqlite_db import init_source_systems
    except Exception:
        from sqlite_db import init_source_systems

    init_source_input()
    init_source_systems()

    where = ["s.app_user_id = ?"]
    params: List[Any] = [app_user_id]

    if since_v:
        where.append("s.last_seen_at >= ?")
        params.append(since_v)
    if until_v:
        where.append("s.last_seen_at <= ?")
        params.append(until_v)

    where_sql = " AND ".join(where)

    base_sql = f"""
    SELECT
      s.id,
      s.system_name,
      s.system_type,
      s.description,
      s.active,
      s.last_seen_at,
      s.created_at
    FROM source_systems s
    WHERE {where_sql}
    ORDER BY s.system_name ASC, s.id ASC
    """

    stats_sql = f"""
    WITH counts AS (
      SELECT
        source_name AS system_name,
        COUNT(*) AS total_records,
        COUNT(DISTINCT source_id) AS total_unique_source_ids
      FROM source_input
      WHERE app_user_id = ?
      GROUP BY source_name
    )
    SELECT
      s.id,
      s.system_name,
      s.system_type,
      s.description,
      s.active,
      s.last_seen_at,
      s.created_at,
      COALESCE(c.total_records, 0) AS total_records,
      COALESCE(c.total_unique_source_ids, 0) AS total_unique_source_ids
    FROM source_systems s
    LEFT JOIN counts c
      ON c.system_name = s.system_name
    WHERE {where_sql}
    ORDER BY s.system_name ASC, s.id ASC
    """

    with get_conn() as conn:
        try:
            if include_stats:
                rows = conn.execute(stats_sql, tuple([app_user_id, *params])).fetchall()
            else:
                rows = conn.execute(base_sql, tuple(params)).fetchall()
        except Exception as ex:
            raise ServiceError(
                "source_systems query failed",
                payload={"detail": f"{ex.__class__.__name__}: {ex}"},
            )

    items: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            items.append(dict(r))
        else:
            if include_stats:
                items.append(
                    {
                        "id": r[0],
                        "system_name": r[1],
                        "system_type": r[2],
                        "description": r[3],
                        "active": r[4],
                        "last_seen_at": r[5],
                        "created_at": r[6],
                        "total_records": r[7],
                        "total_unique_source_ids": r[8],
                    }
                )
            else:
                items.append(
                    {
                        "id": r[0],
                        "system_name": r[1],
                        "system_type": r[2],
                        "description": r[3],
                        "active": r[4],
                        "last_seen_at": r[5],
                        "created_at": r[6],
                    }
                )

    return {"items": items, "count": len(items)}


# ---------------------------------------------------------------------------
# Recon cluster list service (moved from recon_cluster_records_api.py)
# ---------------------------------------------------------------------------

def recon_cluster_list_records(
    *,
    app_user_id: int,
    model_id: str = "",
    status: str = "",
    limit: int = 1000,
    offset: int = 0,
) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    status_v = str(status or "").strip().lower()

    if not isinstance(limit, int):
        raise ValidationError("limit must be an integer")
    if not isinstance(offset, int):
        raise ValidationError("offset must be an integer")

    if limit <= 0:
        raise ValidationError("limit must be >= 1")
    if limit > 5000:
        raise ValidationError("limit must be <= 5000")
    if offset < 0:
        raise ValidationError("offset must be >= 0")

    if status_v and status_v not in {"match", "exception", "exceptions"}:
        raise ValidationError("status must be one of: match | exceptions")

    try:
        _init_recon_cluster()
    except Exception as ex:
        raise ServiceError(f"init_recon_cluster failed: {ex.__class__.__name__}: {ex}")

    f_cols = [f"f{i:02d}" for i in range(1, 21)]
    cols = [
        "rowid AS id",
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

    sql = f"SELECT {', '.join(cols)} FROM recon_cluster WHERE app_user_id = ?"
    params: List[Any] = [app_user_id]

    if mid:
        sql += " AND model_id = ?"
        params.append(mid)

    if status_v == "match":
        sql += " AND (match_status IS NULL OR lower(match_status) = 'match')"
    elif status_v in {"exception", "exceptions"}:
        sql += " AND (lower(trim(coalesce(match_status, ''))) = 'exception')"


    sql += (
        " ORDER BY cluster_id ASC, (match_score IS NULL) ASC, match_score DESC"
        " LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])

    try:
        with get_conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
    except Exception as ex:
        raise ServiceError(
            "recon_cluster query failed",
            payload={"detail": f"{ex.__class__.__name__}: {ex}"},
        )

    items: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            items.append(dict(r))
        else:
            items.append(
                {
                    "id": r[0],
                    "job_id": r[1],
                    "cluster_id": r[2],
                    "record_id": r[3],
                    "model_id": r[4],
                    "model_name": r[5],
                    "source_name": r[6],
                    "source_id": r[7],
                    "match_status": r[8],
                    "match_score": r[9],
                    "created_at": r[10],
                    "created_by": r[11],
                    "updated_at": r[12],
                    "updated_by": r[13],
                    "f01": r[14],
                    "f02": r[15],
                    "f03": r[16],
                    "f04": r[17],
                    "f05": r[18],
                    "f06": r[19],
                    "f07": r[20],
                    "f08": r[21],
                    "f09": r[22],
                    "f10": r[23],
                    "f11": r[24],
                    "f12": r[25],
                    "f13": r[26],
                    "f14": r[27],
                    "f15": r[28],
                    "f16": r[29],
                    "f17": r[30],
                    "f18": r[31],
                    "f19": r[32],
                    "f20": r[33],
                }
            )

    if items:
        all_keys = list(items[0].keys())
        f_keys = sorted(
            [k for k in all_keys if k.startswith("f") and k[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )

        keep = []
        for k in f_keys:
            has_value = False
            for it in items:
                v = it.get(k)
                if v is None:
                    continue
                if str(v).strip():
                    has_value = True
                    break
            if has_value:
                keep.append(k)

        drop = [k for k in f_keys if k not in keep]
        if drop:
            for it in items:
                for k in drop:
                    it.pop(k, None)

    return {"items": items, "count": len(items), "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Golden record list service (moved from golden_record_records_api.py)
# ---------------------------------------------------------------------------

def golden_record_list_records(
    *,
    app_user_id: int,
    model_id: str = "",
    limit: int = 1000,
    offset: int = 0,
) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()

    if not isinstance(limit, int):
        raise ValidationError("limit must be an integer")
    if not isinstance(offset, int):
        raise ValidationError("offset must be an integer")

    if limit <= 0:
        raise ValidationError("limit must be >= 1")
    if limit > 5000:
        raise ValidationError("limit must be <= 5000")
    if offset < 0:
        raise ValidationError("offset must be >= 0")

    try:
        from db.sqlite_db import init_golden_record
    except Exception:
        from sqlite_db import init_golden_record

    init_golden_record()

    cols = [
        "rowid AS id",
        "model_id",
        "model_name",
        "record_id",
        "source_name",
        "source_id",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "golden_json",
    ]

    sql = f"SELECT {', '.join(cols)} FROM golden_record WHERE app_user_id = ?"
    params: List[Any] = [app_user_id]

    if mid:
        sql += " AND model_id = ?"
        params.append(mid)

    sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        with get_conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
    except Exception as ex:
        raise ServiceError(
            "golden_record query failed",
            payload={"detail": f"{ex.__class__.__name__}: {ex}"},
        )

    items: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            items.append(dict(r))
        else:
            items.append(
                {
                    "id": r[0],
                    "model_id": r[1],
                    "model_name": r[2],
                    "record_id": r[3],
                    "source_name": r[4],
                    "source_id": r[5],
                    "created_at": r[6],
                    "created_by": r[7],
                    "updated_at": r[8],
                    "updated_by": r[9],
                    "golden_json": r[10],
                }
            )

    return {"items": items, "count": len(items), "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Batch ingestion service (moved from ingestion_api.py)
# ---------------------------------------------------------------------------

def source_input_ingest_batch_from_fields_array(
    *,
    app_user_id: int,
    source_name: Any,
    actor: Any,
    rows: Any,
    headers: Any = None,
    max_rows_per_call: int = 5000,
) -> Dict[str, Any]:
    import sqlite3

    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    if not source_name or not isinstance(source_name, str):
        raise ValidationError("source_name is required (string)")
    sname = str(source_name).strip()
    if not sname:
        raise ValidationError("source_name is required (string)")

    if not actor or not isinstance(actor, str):
        raise ValidationError("actor is required (string)")
    actor_v = str(actor).strip()
    if not actor_v:
        raise ValidationError("actor is required (string)")

    if not isinstance(rows, list) or len(rows) == 0:
        raise ValidationError("rows must be a non-empty array")

    if not isinstance(max_rows_per_call, int):
        raise ValidationError("max_rows_per_call must be an integer")
    if max_rows_per_call <= 0:
        raise ValidationError("max_rows_per_call must be >= 1")

    if len(rows) > max_rows_per_call:
        raise ValidationError(f"too many rows: {len(rows)} (max {max_rows_per_call})")

    if headers is not None and not isinstance(headers, list):
        raise ValidationError("headers must be an array (optional)")

    init_source_input()

    f_cols = [f"f{str(i).zfill(2)}" for i in range(1, 21)]
    cols = [
        "app_user_id",
        "source_name",
        "source_id",
        *f_cols,
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
    ]

    placeholders = ",".join(["?"] * (3 + 20 + 2))
    update_set = ", ".join([f"{c}=excluded.{c}" for c in f_cols])

    insert_sql = f"""
    INSERT INTO source_input (
      {",".join(cols)}
    ) VALUES (
      {placeholders},
      NULL, NULL
    )
    ON CONFLICT(app_user_id, source_name, source_id) DO UPDATE SET
      {update_set},
      updated_at=?,
      updated_by=?
    WHERE source_input.app_user_id IS NULL OR source_input.app_user_id = excluded.app_user_id;
    """

    now = _utc_now_iso()

    samples = [[] for _ in range(20)]
    max_field_len = 0
    params = []

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValidationError(f"row {i} must be an object")

        source_id = r.get("source_id")
        fields = r.get("fields")

        if not source_id or not isinstance(source_id, str):
            raise ValidationError(f"row {i}: source_id is required (string)")
        source_id_v = str(source_id).strip()
        if not source_id_v:
            raise ValidationError(f"row {i}: source_id is required (string)")

        if not isinstance(fields, list):
            raise ValidationError(f"row {i}: fields is required (array length <= 20)")
        if len(fields) > 20:
            raise ValidationError(f"row {i}: fields length is {len(fields)} (max 20)")

        if len(fields) > max_field_len:
            max_field_len = len(fields)

        padded = (fields + [None] * 20)[:20]
        cleaned = []
        for j, v in enumerate(padded):
            if v is None:
                cleaned.append(None)
                continue
            sv = str(v).strip()
            cleaned.append(sv if sv else None)
            if cleaned[-1] is not None and len(samples[j]) < 50:
                samples[j].append(cleaned[-1])

        params.append(
            (
                app_user_id,
                sname,
                source_id_v,
                *cleaned,
                now,
                actor_v,
                now,
                actor_v,
            )
        )

    try:
        with get_conn() as conn:
            conn.executemany(insert_sql, params)
    except sqlite3.OperationalError as e:
        raise ServiceError("sqlite error", payload={"detail": str(e)})
    except Exception as ex:
        raise ServiceError(
            "ingestion failed",
            payload={"detail": f"{ex.__class__.__name__}: {ex}"},
        )

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
                "type": _csv_best_type(samples[j]),
            }
        )

    return {
        "ok": True,
        "source_name": sname,
        "received_rows": len(rows),
        "upserted_rows": len(rows),
        "max_rows_per_call": max_rows_per_call,
        "schema": {"fields": schema_fields},
    }

def source_input_ingest_batch(
    *,
    app_user_id: int,
    source_name: str,
    rows: List[Dict[str, Any]],
    max_rows_per_call: int = 5000,
) -> Dict[str, Any]:
    import sqlite3

    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    sname = str(source_name or "").strip()
    if not sname:
        raise ValidationError("source_name is required")

    if not isinstance(rows, list):
        raise ValidationError("rows must be a list")

    if not isinstance(max_rows_per_call, int):
        raise ValidationError("max_rows_per_call must be an integer")
    if max_rows_per_call <= 0:
        raise ValidationError("max_rows_per_call must be >= 1")

    if len(rows) > max_rows_per_call:
        raise ValidationError(
            f"too many rows (max {max_rows_per_call})",
            payload={"max_rows_per_call": max_rows_per_call},
        )

    init_source_input()

    reserved = {
        "id",
        "source_id",
        "source_name",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "app_user_id",
    }

    header_set = set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValidationError(f"row {i + 1} must be an object")
        for k in r.keys():
            kk = str(k or "").strip()
            if not kk:
                continue
            if kk in reserved:
                continue
            header_set.add(kk)

    header_list = sorted(header_set)

    if len(header_list) > 20:
        raise ValidationError(
            f"too many fields (max 20). got {len(header_list)}",
            payload={"fields": header_list},
        )

    f_cols = [f"f{i:02d}" for i in range(1, 21)]
    cols = [
        "source_id",
        "source_name",
        *f_cols,
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
    ON CONFLICT(app_user_id, source_name, source_id) DO UPDATE SET
      {update_set}
    WHERE source_input.app_user_id IS NULL OR source_input.app_user_id = excluded.app_user_id;
    """

    now = _utc_now_iso()

    def _clean_id(v: Any) -> str:
        s = "" if v is None else str(v)
        return s.strip()

    def _clean_field(v: Any) -> Optional[str]:
        s = "" if v is None else str(v)
        s = s.strip()
        return s if s else None

    samples: Dict[str, List[str]] = {h: [] for h in header_list}
    to_insert = []
    for i, r in enumerate(rows):
        record_id = _clean_id(r.get("id") or r.get("source_id"))
        if not record_id:
            raise ValidationError(f"row {i + 1}: missing id/source_id")

        created_at = _clean_id(r.get("created_at") or now)
        created_by = _clean_id(r.get("created_by") or "ingestion")
        updated_at = now
        updated_by = "ingestion"

        values = []
        for h in header_list:
            v = _clean_field(r.get(h))
            values.append(v)
            if v is not None and len(samples[h]) < 50:
                samples[h].append(v)

        padded = (values + [None] * 20)[:20]

        to_insert.append(
            (
                record_id,
                sname,
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
            conn.executemany(insert_sql, to_insert)
            conn.commit()
    except sqlite3.OperationalError as e:
        raise ServiceError("sqlite error", payload={"detail": str(e)})
    except Exception as ex:
        raise ServiceError(
            "ingestion failed",
            payload={"detail": f"{ex.__class__.__name__}: {ex}"},
        )

    schema_fields = []
    for j, h in enumerate(header_list):
        code = f"f{j + 1:02d}"
        schema_fields.append(
            {"code": code, "header": h, "type": _csv_best_type(samples[h])}
        )

    return {
        "ok": True,
        "source_name": sname,
        "received_rows": len(rows),
        "upserted_rows": len(rows),
        "max_rows_per_call": max_rows_per_call,
        "schema": {"fields": schema_fields},
    }


def source_input_ingest_batch_from_fields_array(
    *,
    app_user_id: int,
    source_name: str,
    actor: str,
    rows: List[Dict[str, Any]],
    headers: Optional[List[Any]] = None,
    max_rows_per_call: int = 5000,
) -> Dict[str, Any]:
    import sqlite3

    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    sname = str(source_name or "").strip()
    if not sname:
        raise ValidationError("source_name is required")

    actor_v = str(actor or "").strip()
    if not actor_v:
        raise ValidationError("actor is required")

    if not isinstance(rows, list):
        raise ValidationError("rows must be a list")

    if not isinstance(max_rows_per_call, int):
        raise ValidationError("max_rows_per_call must be an integer")
    if max_rows_per_call <= 0:
        raise ValidationError("max_rows_per_call must be >= 1")

    if len(rows) > max_rows_per_call:
        raise ValidationError(
            f"too many rows (max {max_rows_per_call})",
            payload={"max_rows_per_call": max_rows_per_call},
        )

    if headers is not None and not isinstance(headers, list):
        raise ValidationError("headers must be an array (optional)")

    init_source_input()

    now = _utc_now_iso()

    samples: List[List[str]] = [[] for _ in range(20)]
    max_field_len = 0

    to_insert = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValidationError(f"row {i + 1} must be an object")

        source_id = str(r.get("source_id") or "").strip()
        if not source_id:
            raise ValidationError(f"row {i + 1}: source_id is required")

        fields = r.get("fields")
        if not isinstance(fields, list):
            raise ValidationError(f"row {i + 1}: fields is required (array length <= 20)")
        if len(fields) > 20:
            raise ValidationError(f"row {i + 1}: fields length is {len(fields)} (max 20)")

        if len(fields) > max_field_len:
            max_field_len = len(fields)

        padded = (fields + [None] * 20)[:20]

        cleaned = []
        for j, v in enumerate(padded):
            if v is None:
                cleaned.append(None)
                continue

            s = str(v).strip()
            if not s:
                cleaned.append(None)
                continue

            cleaned.append(s)
            if len(samples[j]) < 50:
                samples[j].append(s)

        to_insert.append(
            (
                app_user_id,
                sname,
                source_id,
                *cleaned,
                now,
                actor_v,
                now,
                actor_v,
            )
        )

    f_cols = [f"f{i:02d}" for i in range(1, 21)]
    f_cols_sql = ", ".join(f_cols)
    f_placeholders = ",".join(["?"] * 20)

    update_set = ", ".join([f"{c}=excluded.{c}" for c in f_cols])

    insert_sql = f"""
    INSERT INTO source_input (
      app_user_id,
      source_name,
      source_id,
      {f_cols_sql},
      created_at,
      created_by,
      updated_at,
      updated_by
    ) VALUES (
      ?,
      ?,
      ?,
      {f_placeholders},
      ?,
      ?,
      NULL,
      NULL
    )
    ON CONFLICT(app_user_id, source_name, source_id) DO UPDATE SET
      {update_set},
      updated_at=?,
      updated_by=?
    """

    try:
        with get_conn() as conn:
            conn.executemany(insert_sql, to_insert)
            conn.commit()
    except sqlite3.OperationalError as e:
        raise ServiceError("sqlite error", payload={"detail": str(e)})
    except Exception as ex:
        raise ServiceError(
            "ingestion failed",
            payload={"detail": f"{ex.__class__.__name__}: {ex}"},
        )

    field_count = min(20, max_field_len)

    header_list: List[str] = []
    if isinstance(headers, list) and headers and field_count > 0:
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
                "type": _csv_best_type(samples[j]),
            }
        )

    return {
        "ok": True,
        "source_name": sname,
        "received_rows": len(rows),
        "upserted_rows": len(rows),
        "max_rows_per_call": max_rows_per_call,
        "schema": {"fields": schema_fields},
    }


# ---------------------------------------------------------------------------
# MDM model services (moved from mdm_model_api.py)
# ---------------------------------------------------------------------------


def _mdm_model_imports():
    try:
        from db.sqlite_db import (
            init_mdm_models,
            create_mdm_model,
            list_mdm_models,
            get_mdm_model_by_id,
            update_mdm_model,
            soft_delete_mdm_model,
        )
    except Exception:
        from sqlite_db import (
            init_mdm_models,
            create_mdm_model,
            list_mdm_models,
            get_mdm_model_by_id,
            update_mdm_model,
            soft_delete_mdm_model,
        )

    return (
        init_mdm_models,
        create_mdm_model,
        list_mdm_models,
        get_mdm_model_by_id,
        update_mdm_model,
        soft_delete_mdm_model,
    )


def _mdm_assert_model_owner(model_row: Dict[str, Any], app_user_id: int) -> None:
    owner_id = model_row.get("owner_user_id")
    model_app_user_id = model_row.get("app_user_id")

    if owner_id is not None and int(owner_id) != int(app_user_id):
        raise ForbiddenError("model does not belong to user")
    if model_app_user_id is not None and int(model_app_user_id) != int(app_user_id):
        raise ForbiddenError("model does not belong to user")


def _slugify(s: str) -> str:
    import re

    x = str(s or "").strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = x.strip("_")
    return x


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _is_field_obj(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    code = str(x.get("code") or "").strip()
    return bool(code)


def _clean_code(s: Any) -> str:
    x = str(s or "").strip()
    return x


def _clean_label(s: Any) -> str:
    x = str(s or "").strip()
    return x


def _normalize_config(cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Canonicalize model config to the app's expected structure.
    Raises ValueError on invalid input.
    """
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")

    out: Dict[str, Any] = {}

    out["domainModelName"] = str(
        cfg.get("domainModelName") or cfg.get("model_name") or model_name or ""
    ).strip()
    if not out["domainModelName"]:
        raise ValueError("config.domainModelName is required")

    config_obj = cfg.get("config")
    if config_obj is None:
        config_obj = {}
    if not isinstance(config_obj, dict):
        raise ValueError("config.config must be an object")

    fields = config_obj.get("fields")
    if fields is None:
        fields = cfg.get("fields")
    if fields is None:
        fields = []
    if not isinstance(fields, list):
        raise ValueError("config.fields must be an array")

    clean_fields: List[Dict[str, Any]] = []
    seen_codes = set()
    for f in fields:
        if not _is_field_obj(f):
            continue
        code = _clean_code(f.get("code"))
        if not code:
            continue
        if code in seen_codes:
            continue
        seen_codes.add(code)

        label = _clean_label(f.get("label") or code)
        clean_fields.append(
            {
                "code": code,
                "label": label,
                "type": str(f.get("type") or "text"),
            }
        )

    match_rules = config_obj.get("matchRules")
    if match_rules is None:
        match_rules = cfg.get("matchRules")
    if match_rules is None:
        match_rules = []
    if not isinstance(match_rules, list):
        raise ValueError("config.matchRules must be an array")

    clean_rules: List[Dict[str, Any]] = []
    for r in match_rules:
        if not isinstance(r, dict):
            continue
        field = _clean_code(r.get("field"))
        if not field:
            continue
        clean_rules.append(
            {
                "field": field,
                "weight": _to_float(r.get("weight"), 0.0),
                "on": bool(r.get("on", True)),
                "method": str(r.get("method") or ""),
                "threshold": _to_float(r.get("threshold"), 0.0),
            }
        )

    survivorship = config_obj.get("survivorship")
    if survivorship is None:
        survivorship = cfg.get("survivorship")
    if survivorship is None:
        survivorship = {}
    if not isinstance(survivorship, dict):
        raise ValueError("config.survivorship must be an object")

    surv_mode = survivorship.get("mode")
    if surv_mode is None:
        surv_mode = "best"
    surv_mode = str(surv_mode or "").strip() or "best"

    global_rule = survivorship.get("globalRule")
    if global_rule is None:
        global_rule = "best"
    global_rule = str(global_rule or "").strip() or "best"

    surv_rules = survivorship.get("rules")
    if surv_rules is None:
        surv_rules = []
    if not isinstance(surv_rules, list):
        raise ValueError("survivorship.rules must be an array")

    clean_surv_rules: List[Dict[str, Any]] = []
    for r in surv_rules:
        if not isinstance(r, dict):
            continue
        field = str(r.get("field") or "").strip()
        if not field:
            continue
        rule = str(r.get("rule") or "").strip() or global_rule
        sources = r.get("sources")
        clean_surv_rules.append(
            {
                "field": field,
                "rule": rule,
                "sources": sources if isinstance(sources, list) else [],
            }
        )

    out["config"] = {
        "fields": clean_fields,
        "matchRules": clean_rules,
        "survivorship": {
            "mode": surv_mode,
            "globalRule": global_rule,
            "rules": clean_surv_rules,
        },
    }

    return out


def mdm_model_list(*, app_user_id: int, include_deleted: bool = False) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")
    if not isinstance(include_deleted, bool):
        raise ValidationError("include_deleted must be a boolean")

    init_mdm_models, _, list_mdm_models, _, _, _ = _mdm_model_imports()
    init_mdm_models()

    rows = list_mdm_models(include_deleted=include_deleted) or []
    models: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            _mdm_assert_model_owner(d, app_user_id)
        except ForbiddenError:
            continue
        models.append(d)

    return {"ok": True, "models": models}


def mdm_model_get(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    init_mdm_models, _, _, get_mdm_model_by_id, _, _ = _mdm_model_imports()
    init_mdm_models()

    m = get_mdm_model_by_id(mid, include_deleted=False)
    if not m:
        raise NotFoundError("model not found")

    d = dict(m)
    _mdm_assert_model_owner(d, app_user_id)

    try:
        d["config"] = json.loads(d.get("config_json") or "{}")
    except Exception:
        d["config"] = None

    return {"ok": True, "model": d}


def mdm_model_create(
    *,
    app_user_id: int,
    actor: str = "api",
    model_name: str = "",
    model_key: str = "",
    config: Any = None,
) -> Dict[str, Any]:
    import sqlite3

    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    actor_v = str(actor or "").strip() or "api"

    name_v = str(model_name or "").strip()
    key_v = str(model_key or "").strip()

    cfg = config
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            raise ValidationError("config must be valid JSON")

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValidationError("config must be an object")

    if not name_v:
        name_v = str(cfg.get("domainModelName") or "").strip()

    if not name_v:
        raise ValidationError("model_name is required (or provide config.domainModelName)")

    if not key_v:
        key_v = _slugify(name_v)

    if not key_v:
        raise ValidationError("model_key could not be derived")

    try:
        cfg_norm = _normalize_config(cfg, name_v)
    except ValueError as ex:
        raise ValidationError(str(ex))

    config_json = json.dumps(cfg_norm, ensure_ascii=False, separators=(",", ":"))
    now = _utc_now_iso()

    init_mdm_models, create_mdm_model, _, _, _, _ = _mdm_model_imports()
    init_mdm_models()

    try:
        new_id = create_mdm_model(
            model_key=key_v,
            model_name=name_v,
            config_json=config_json,
            owner_user_id=app_user_id,
            actor=actor_v,
            now=now,
        )
    except sqlite3.IntegrityError:
        raise ConflictError("active model_name or model_key already exists")
    except Exception as ex:
        raise ServiceError("create model failed", payload={"detail": f"{ex.__class__.__name__}: {ex}"})

    return {"ok": True, "id": str(new_id), "model_key": key_v, "model_name": name_v}


def mdm_model_update(
    *,
    app_user_id: int,
    actor: str = "api",
    model_id: str,
    model_name: str = "",
    config: Any = None,
) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    actor_v = str(actor or "").strip() or "api"

    name_v = str(model_name or "").strip()
    cfg = config

    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            raise ValidationError("config must be valid JSON")

    if cfg is not None and not isinstance(cfg, dict):
        raise ValidationError("config must be an object")

    if not name_v and cfg is None:
        raise ValidationError("nothing to update")

    init_mdm_models, _, _, get_mdm_model_by_id, update_mdm_model, _ = _mdm_model_imports()
    init_mdm_models()

    existing = get_mdm_model_by_id(mid, include_deleted=True)
    if not existing:
        raise NotFoundError("model not found")

    exd = dict(existing)
    _mdm_assert_model_owner(exd, app_user_id)

    config_json = None
    if cfg is not None:
        use_name = name_v or str(cfg.get("domainModelName") or "").strip()
        if not use_name:
            raise ValidationError("model_name is required (or provide config.domainModelName)")
        try:
            cfg_norm = _normalize_config(cfg, use_name)
        except ValueError as ex:
            raise ValidationError(str(ex))
        config_json = json.dumps(cfg_norm, ensure_ascii=False, separators=(",", ":"))

    now = _utc_now_iso()

    ok = update_mdm_model(
        model_id=mid,
        model_name=name_v if name_v else None,
        config_json=config_json,
        actor=actor_v,
        now=now,
    )
    if not ok:
        raise NotFoundError("model not found")

    return {"ok": True, "message": "model updated"}


def mdm_model_delete(*, app_user_id: int, actor: str = "api", model_id: str = "") -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    actor_v = str(actor or "").strip() or "api"
    now = _utc_now_iso()

    init_mdm_models, _, _, get_mdm_model_by_id, _, soft_delete_mdm_model = _mdm_model_imports()
    init_mdm_models()

    existing = get_mdm_model_by_id(mid, include_deleted=True)
    if not existing:
        raise NotFoundError("model not found")

    exd = dict(existing)
    _mdm_assert_model_owner(exd, app_user_id)

    ok = soft_delete_mdm_model(model_id=mid, actor=actor_v, now=now)
    if not ok:
        raise NotFoundError("model not found")

    return {"ok": True}


# ---------------------------------------------------------------------------
# Matching summary service (moved from matching_summary_api.py)
# ---------------------------------------------------------------------------

SURVIVORSHIP_LABELS = {
    "best": "Best Available",
    "recency": "Most Recent",
    "source_priority": "Source Priority",
    "most_common": "Most Common",
}


def _survivorship_rule_label(rule: str) -> str:
    r = str(rule or "").strip().lower()
    return SURVIVORSHIP_LABELS.get(r, r or "best")


def _init_matching_summary_tables() -> None:
    try:
        from db.sqlite_db import (
            init_mdm_models,
            init_match_job,
            init_cluster_map,
            init_golden_record,
            init_match_exception,
        )
    except Exception:
        from sqlite_db import (
            init_mdm_models,
            init_match_job,
            init_cluster_map,
            init_golden_record,
            init_match_exception,
        )

    init_source_input()
    init_mdm_models()
    _init_recon_cluster()
    init_match_job()
    init_cluster_map()
    init_golden_record()
    init_match_exception()


def matching_summary_get(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    try:
        _init_matching_summary_tables()
    except Exception as ex:
        raise ServiceError(
            "init failed",
            payload={"detail": f"{ex.__class__.__name__}: {ex}"},
        )

    with get_conn() as conn:
        db_file = _get_db_file(conn)

        model_row = conn.execute(
            """
            SELECT id, model_name, config_json, owner_user_id, app_user_id
            FROM mdm_models
            WHERE id = ?
              AND deleted_at IS NULL
            LIMIT 1
            """,
            (mid,),
        ).fetchone()

        if not model_row:
            raise NotFoundError("model not found")

        owner_id = _row_get(model_row, "owner_user_id", 3)
        model_app_user_id = _row_get(model_row, "app_user_id", 4)

        if owner_id is not None and int(owner_id) != int(app_user_id):
            raise ForbiddenError("model does not belong to user")
        if model_app_user_id is not None and int(model_app_user_id) != int(app_user_id):
            raise ForbiddenError("model does not belong to user")

        model_name = str(_row_get(model_row, "model_name", 1) or "").strip()

        cfg_raw = _row_get(model_row, "config_json", 2)
        try:
            cfg = json.loads(cfg_raw or "{}") if cfg_raw else {}
        except Exception:
            cfg = {}

        code_to_label = _build_code_to_label_map(cfg)

        cfg_obj = cfg.get("config") if isinstance(cfg, dict) else None
        if not isinstance(cfg_obj, dict):
            cfg_obj = {}

        mr = cfg_obj.get("matchRules")
        if not isinstance(mr, list):
            mr = []

        matching_fields: List[Dict[str, Any]] = []
        for r in mr:
            if not isinstance(r, dict):
                continue
            code = str(r.get("field") or "").strip()
            if not code:
                continue
            label = code_to_label.get(code) or code
            matching_fields.append(
                {
                    "code": code,
                    "label": label,
                    "weight": float(_to_float(r.get("weight"), 0.0)),
                }
            )

        match_field_pills = [x["label"] for x in matching_fields if x.get("label")]

        survivorship = cfg_obj.get("survivorship")
        if not isinstance(survivorship, dict):
            survivorship = {}

        survivorship_mode = str(survivorship.get("mode") or "best").strip() or "best"
        global_rule = str(survivorship.get("globalRule") or "best").strip() or "best"
        survivorship_label = _survivorship_rule_label(global_rule)

        survivorship_rules: List[Dict[str, Any]] = []
        sv_rules = survivorship.get("rules")
        if isinstance(sv_rules, list):
            for r in sv_rules:
                if not isinstance(r, dict):
                    continue
                field = str(r.get("field") or "").strip()
                if not field:
                    continue
                rule = str(r.get("rule") or "").strip() or global_rule
                sources = r.get("sources")
                survivorship_rules.append(
                    {
                        "field": field,
                        "label": code_to_label.get(field) or field,
                        "rule": rule,
                        "rule_label": _survivorship_rule_label(rule),
                        "sources": sources if isinstance(sources, list) else [],
                    }
                )

        job_row = conn.execute(
            """
            SELECT *
            FROM match_job
            WHERE app_user_id = ?
              AND model_id = ?
            ORDER BY created_at DESC, updated_at DESC
            LIMIT 1
            """,
            (app_user_id, mid),
        ).fetchone()

        job = None
        job_id = ""

        if job_row:
            try:
                job_id = str(job_row["job_id"] or "").strip()
            except Exception:
                job_id = ""
            job = {
                "job_id": job_id,
                "status": str(job_row["status"] or ""),
                "created_at": str(job_row["created_at"] or ""),
                "updated_at": str(job_row["updated_at"] or ""),
                "started_at": str(job_row["started_at"] or ""),
                "finished_at": str(job_row["finished_at"] or ""),
            }

        golden_records = 0
        exceptions = 0

        if job_id:
            gr = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM golden_record
                WHERE app_user_id = ?
                  AND model_id = ?
                  AND job_id = ?
                """,
                (app_user_id, mid, job_id),
            ).fetchone()
            try:
                golden_records = int(gr["n"] or 0)
            except Exception:
                golden_records = 0

            exr = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM match_exception
                WHERE app_user_id = ?
                  AND model_id = ?
                  AND job_id = ?
                  AND resolved_at IS NULL
                """,
                (app_user_id, mid, job_id),
            ).fetchone()
            try:
                exceptions = int(exr["n"] or 0)
            except Exception:
                exceptions = 0

        cl = conn.execute(
            """
            SELECT COUNT(DISTINCT cluster_id) AS n
            FROM cluster_map
            WHERE app_user_id = ?
              AND model_id = ?
            """,
            (app_user_id, mid),
        ).fetchone()
        try:
            match_clusters = int(cl["n"] or 0)
        except Exception:
            match_clusters = 0

    return {
        "db_file": db_file,
        "model_id": mid,
        "model_name": model_name,
        "job": job,
        "golden_records": golden_records,
        "exceptions": exceptions,
        "match_clusters": match_clusters,
        "matching_fields": matching_fields,
        "match_field_pills": match_field_pills,
        "survivorship_mode": survivorship_mode,
        "survivorship_label": survivorship_label,
        "survivorship_global_rule": global_rule,
        "survivorship_rules": survivorship_rules,
    }


# ---------------------------------------------------------------------------
# API Key + API ingestion services
# ---------------------------------------------------------------------------

def _sha256_hex(s: str) -> str:
    import hashlib

    b = (s or "").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _new_api_token() -> str:
    import secrets

    return "mdm_" + secrets.token_urlsafe(32)


def api_key_rotate(*, app_user_id: int, model_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Rotate API token:
    - Enforces one active token per (user, model)
    - Manual rotate only (no expiry)
    - Stores only token_hash; returns raw token once
    """
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    name_v = None if name is None else str(name)

    import uuid

    _init_all_tables()

    api_key_id = str(uuid.uuid4())
    token = _new_api_token()
    token_hash = _sha256_hex(token)
    token_prefix = token[:10]
    now = _utc_now_iso()

    with get_conn() as conn:
        model_row = conn.execute(
            """
            SELECT owner_user_id, app_user_id, deleted_at
            FROM mdm_models
            WHERE id = ?
            LIMIT 1
            """,
            (mid,),
        ).fetchone()

        if not model_row:
            raise NotFoundError("model not found", payload={"model_id": mid})

        deleted_at = _row_get(model_row, "deleted_at", 2)
        if deleted_at:
            raise NotFoundError("model not found", payload={"model_id": mid})

        owner_id = _row_get(model_row, "owner_user_id", 0)
        model_app_user_id = _row_get(model_row, "app_user_id", 1)

        if owner_id is not None and int(owner_id) != int(app_user_id):
            raise ForbiddenError("model does not belong to user")
        if model_app_user_id is not None and int(model_app_user_id) != int(app_user_id):
            raise ForbiddenError("model does not belong to user")

        conn.execute(
            """
            UPDATE api_key
            SET status='revoked', revoked_at=?
            WHERE app_user_id=?
              AND model_id=?
              AND status='active'
            """,
            (now, app_user_id, mid),
        )

        conn.execute(
            """
            INSERT INTO api_key (
              api_key_id,
              app_user_id,
              model_id,
              name,
              token_prefix,
              token_hash,
              status,
              created_at,
              last_used_at,
              revoked_at
            ) VALUES (
              ?,
              ?,
              ?,
              ?,
              ?,
              ?,
              'active',
              ?,
              NULL,
              NULL
            )
            """,
            (api_key_id, app_user_id, mid, name_v, token_prefix, token_hash, now),
        )

    return {
        "ok": True,
        "api_key_id": api_key_id,
        "model_id": mid,
        "token_prefix": token_prefix,
        "token": token,
        "status": "active",
        "created_at": now,
    }


def api_key_revoke(*, app_user_id: int, model_id: str) -> Dict[str, Any]:
    if not isinstance(app_user_id, int):
        raise ValidationError("app_user_id must be an integer")
    if app_user_id <= 0:
        raise ValidationError("app_user_id must be a positive integer")

    mid = str(model_id or "").strip()
    if not mid:
        raise ValidationError("model_id is required")

    _init_all_tables()
    now = _utc_now_iso()

    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE api_key
            SET status='revoked', revoked_at=?
            WHERE app_user_id=?
              AND model_id=?
              AND status='active'
            """,
            (now, app_user_id, mid),
        )
        if cur.rowcount <= 0:
            raise NotFoundError(
                "no active api key found",
                payload={"app_user_id": app_user_id, "model_id": mid},
            )

    return {"ok": True, "model_id": mid, "revoked_at": now}


def _api_authenticate_bearer_token(authorization_header: str) -> Dict[str, Any]:
    """
    API auth for ingestion:
    - Requires Authorization: Bearer <token>
    - No app_user_id in payload (derived from token)
    """
    h = str(authorization_header or "").strip()
    if not h:
        raise UnauthorizedError("missing Authorization header")

    if not h.lower().startswith("bearer "):
        raise UnauthorizedError("invalid Authorization header (expected Bearer token)")

    token = h[7:].strip()
    if not token:
        raise UnauthorizedError("missing bearer token")

    token_hash = _sha256_hex(token)
    now = _utc_now_iso()

    _init_all_tables()

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT api_key_id, app_user_id, model_id, status
            FROM api_key
            WHERE token_hash = ?
              AND status = 'active'
            LIMIT 1
            """,
            (token_hash,),
        ).fetchone()

        if not row:
            raise UnauthorizedError("invalid or revoked API token")

        conn.execute(
            "UPDATE api_key SET last_used_at=? WHERE api_key_id=?",
            (now, row["api_key_id"]),
        )

        return {
            "api_key_id": row["api_key_id"],
            "app_user_id": int(row["app_user_id"]),
            "model_id": str(row["model_id"]),
            "status": str(row["status"]),
            "last_used_at": now,
        }


def api_batch_get(*, authorization: str, batch_id: str) -> Dict[str, Any]:
    key = _api_authenticate_bearer_token(authorization)

    bid = str(batch_id or "").strip()
    if not bid:
        raise ValidationError("batch_id is required")

    _init_all_tables()

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM api_batch
            WHERE batch_id = ?
              AND api_key_id = ?
            LIMIT 1
            """,
            (bid, key["api_key_id"]),
        ).fetchone()

        if not row:
            raise NotFoundError("batch not found", payload={"batch_id": bid})

        return dict(row)


def api_ingest_api(*, authorization: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/ingest/api

    Contract:
    - Authorization: Bearer <token>
    - JSON body: { model_id, batch_id?, finalize?, records: [...] }
    - records are STRICT source_input contract fields (NO app_user_id)
    - finalize=true:
        - blocks if any queued/running match_job exists for user
        - wipes source_input for user
        - promotes staging -> source_input
        - enqueues match job for model_id
        - wipes staging for that batch
    """
    import uuid
    import sqlite3

    key = _api_authenticate_bearer_token(authorization)

    if not isinstance(body, dict):
        raise ValidationError("request body must be a JSON object")

    allowed_body_keys = {"model_id", "batch_id", "finalize", "records"}
    extra_body_keys = set(body.keys()) - allowed_body_keys
    if extra_body_keys:
        raise ValidationError(
            "unexpected top-level fields",
            payload={"unexpected": sorted([str(x) for x in extra_body_keys])},
        )

    req_model_id = str(body.get("model_id") or "").strip()
    if not req_model_id:
        raise ValidationError("model_id is required")

    if req_model_id != str(key["model_id"]):
        raise UnauthorizedError("model_id does not match API token")

    finalize = bool(body.get("finalize", False))

    records = body.get("records")
    if not isinstance(records, list):
        raise ValidationError("records must be a list")

    bid_raw = body.get("batch_id")
    batch_id = None if bid_raw is None else str(bid_raw).strip()
    if not batch_id:
        batch_id = str(uuid.uuid4())

    now = _utc_now_iso()

    allowed_record_keys = {"source_id", "source_name", "created_at", "created_by", "updated_at", "updated_by"}
    for i in range(1, 21):
        allowed_record_keys.add(f"f{str(i).zfill(2)}")

    to_insert = []
    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValidationError(f"record {idx + 1}: must be an object")

        extra = set(r.keys()) - allowed_record_keys
        if extra:
            raise ValidationError(
                f"record {idx + 1}: unexpected fields",
                payload={"unexpected": sorted([str(x) for x in extra])},
            )

        source_id = str(r.get("source_id") or "").strip()
        source_name = str(r.get("source_name") or "").strip()
        created_at = str(r.get("created_at") or "").strip()
        created_by = str(r.get("created_by") or "").strip()

        if not source_id:
            raise ValidationError(f"record {idx + 1}: source_id is required")
        if not source_name:
            raise ValidationError(f"record {idx + 1}: source_name is required")
        if not created_at:
            raise ValidationError(f"record {idx + 1}: created_at is required")
        if not created_by:
            raise ValidationError(f"record {idx + 1}: created_by is required")

        fvals = []
        for i in range(1, 21):
            k = f"f{str(i).zfill(2)}"
            v = r.get(k)
            if v is None:
                fvals.append(None)
            else:
                s = str(v).strip()
                fvals.append(s if s else None)

        updated_at = r.get("updated_at")
        updated_by = r.get("updated_by")
        updated_at = None if updated_at is None else str(updated_at).strip() or None
        updated_by = None if updated_by is None else str(updated_by).strip() or None

        to_insert.append(
            (
                batch_id,
                key["api_key_id"],
                key["app_user_id"],
                req_model_id,
                source_name,
                source_id,
                *fvals,
                created_at,
                created_by,
                updated_at,
                updated_by,
            )
        )

    _init_all_tables()

    try:
        with get_conn() as conn:
            existing_batch = conn.execute(
                """
                SELECT batch_id, status, job_id
                FROM api_batch
                WHERE batch_id = ?
                  AND api_key_id = ?
                LIMIT 1
                """,
                (batch_id, key["api_key_id"]),
            ).fetchone()

            if not existing_batch:
                try:
                    conn.execute(
                        """
                        INSERT INTO api_batch (
                          batch_id,
                          api_key_id,
                          idempotency_key,
                          status,
                          job_id,
                          request_meta_json,
                          log_json,
                          error_message,
                          created_at,
                          updated_at,
                          started_at,
                          finished_at
                        ) VALUES (
                          ?,
                          ?,
                          ?,
                          'received',
                          NULL,
                          ?,
                          NULL,
                          NULL,
                          datetime('now'),
                          datetime('now'),
                          NULL,
                          NULL
                        )
                        """,
                        (
                            batch_id,
                            key["api_key_id"],
                            batch_id,
                            json.dumps({"model_id": req_model_id}, ensure_ascii=False),
                        ),
                    )
                except sqlite3.IntegrityError:
                    pass

                existing_batch = conn.execute(
                    """
                    SELECT batch_id, status, job_id
                    FROM api_batch
                    WHERE batch_id = ?
                      AND api_key_id = ?
                    LIMIT 1
                    """,
                    (batch_id, key["api_key_id"]),
                ).fetchone()

            if not existing_batch:
                raise ServiceError("failed to create or read api_batch", payload={"batch_id": batch_id})

            status = str(existing_batch["status"] or "")
            job_id_existing = existing_batch["job_id"]

            if status != "received":
                return {
                    "ok": True,
                    "batch_id": batch_id,
                    "status": status,
                    "job_id": job_id_existing,
                    "model_id": req_model_id,
                }

            if to_insert:
                f_cols = [f"f{str(i).zfill(2)}" for i in range(1, 21)]
                cols = [
                    "batch_id",
                    "api_key_id",
                    "app_user_id",
                    "model_id",
                    "source_name",
                    "source_id",
                    *f_cols,
                    "created_at",
                    "created_by",
                    "updated_at",
                    "updated_by",
                ]
                placeholders = ",".join(["?"] * len(cols))
                update_set = ", ".join(
                    [
                        *[f"{c}=excluded.{c}" for c in f_cols],
                        "created_at=excluded.created_at",
                        "created_by=excluded.created_by",
                        "updated_at=excluded.updated_at",
                        "updated_by=excluded.updated_by",
                    ]
                )

                insert_sql = f"""
                INSERT INTO api_source_input (
                  {",".join(cols)}
                ) VALUES (
                  {placeholders}
                )
                ON CONFLICT(app_user_id, model_id, batch_id, source_name, source_id) DO UPDATE SET
                  {update_set}
                """

                conn.executemany(insert_sql, to_insert)

            conn.execute(
                "UPDATE api_batch SET updated_at=datetime('now') WHERE batch_id=? AND api_key_id=?",
                (batch_id, key["api_key_id"]),
            )

            staged_row = conn.execute(
                """
                SELECT COUNT(1) AS n
                FROM api_source_input
                WHERE app_user_id=?
                  AND model_id=?
                  AND batch_id=?
                """,
                (key["app_user_id"], req_model_id, batch_id),
            ).fetchone()
            staged_n = int(_row_get(staged_row, "n", 0) or 0) if staged_row else 0

            if not finalize:
                return {
                    "ok": True,
                    "batch_id": batch_id,
                    "status": "received",
                    "model_id": req_model_id,
                    "received_records": len(records),
                    "staged_records": staged_n,
                    "updated_at": now,
                }

            active_job = conn.execute(
                """
                SELECT job_id, status
                FROM match_job
                WHERE app_user_id = ?
                  AND status IN ('queued', 'running')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (key["app_user_id"],),
            ).fetchone()
            if active_job:
                raise ConflictError(
                    "cannot finalize while a match job is queued or running for this user",
                    payload={
                        "job_id": _row_get(active_job, "job_id", 0),
                        "status": _row_get(active_job, "status", 1),
                    },
                )

            if staged_n <= 0:
                raise ValidationError("no staged records for batch", payload={"batch_id": batch_id})

            conn.execute("DELETE FROM source_input WHERE app_user_id=?", (key["app_user_id"],))

            f_cols_sql = ", ".join([f"f{str(i).zfill(2)}" for i in range(1, 21)])
            conn.execute(
                f"""
                INSERT INTO source_input (
                  app_user_id,
                  source_name,
                  source_id,
                  {f_cols_sql},
                  created_at,
                  created_by,
                  updated_at,
                  updated_by
                )
                SELECT
                  app_user_id,
                  source_name,
                  source_id,
                  {f_cols_sql},
                  created_at,
                  created_by,
                  updated_at,
                  updated_by
                FROM api_source_input
                WHERE app_user_id=?
                  AND model_id=?
                  AND batch_id=?
                """,
                (key["app_user_id"], req_model_id, batch_id),
            )
    except ServiceError:
        raise
    except sqlite3.OperationalError as e:
        raise ServiceError("sqlite error", payload={"detail": str(e)})
    except Exception as ex:
        raise ServiceError("api ingestion failed", payload={"detail": f"{ex.__class__.__name__}: {ex}"})

    try:
        job = match_enqueue_job(app_user_id=key["app_user_id"], model_id=req_model_id)
    except ServiceError as se:
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE api_batch
                SET status='failed', error_message=?, updated_at=datetime('now')
                WHERE batch_id=?
                  AND api_key_id=?
                """,
                (str(se), batch_id, key["api_key_id"]),
            )
        raise
    except Exception as ex:
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE api_batch
                SET status='failed', error_message=?, updated_at=datetime('now')
                WHERE batch_id=?
                  AND api_key_id=?
                """,
                (f"{ex.__class__.__name__}: {ex}", batch_id, key["api_key_id"]),
            )
        raise

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE api_batch
            SET status='queued', job_id=?, updated_at=datetime('now')
            WHERE batch_id=?
              AND api_key_id=?
            """,
            (job.get("job_id"), batch_id, key["api_key_id"]),
        )
        conn.execute(
            """
            DELETE FROM api_source_input
            WHERE app_user_id=?
              AND model_id=?
              AND batch_id=?
            """,
            (key["app_user_id"], req_model_id, batch_id),
        )

    return {
        "ok": True,
        "batch_id": batch_id,
        "status": "queued",
        "job_id": job.get("job_id"),
        "model_id": req_model_id,
        "finalized_at": now,
    }



