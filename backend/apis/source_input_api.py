import os
import json
from flask import Blueprint, jsonify, request

try:
    from db.sqlite_db import get_conn, init_source_input
except Exception:
    from sqlite_db import get_conn, init_source_input

bp = Blueprint("source_input_api", __name__, url_prefix="/api")

def _require_app_user_id():
    v = (
        request.headers.get("X-User-Id")
        or request.headers.get("X-App-User-Id")
        or request.args.get("app_user_id")
        or ""
    )
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



@bp.get("/source-input/summary")
def source_input_summary():
    init_source_input()

    app_user_id, err = _require_app_user_id()
    if err:
        return err

    model_id = str(request.args.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400


    model_id = str(request.args.get("model_id") or "").strip()

    with get_conn() as conn:
        code_to_label = {}

        if model_id:
            # Load model config to map f01..f20 => labels
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
                return jsonify({"error": "model not found"}), 404

            try:
                owner_id = row["owner_user_id"] if hasattr(row, "keys") else row[2]
            except Exception:
                owner_id = None
            try:
                model_app_user_id = row["app_user_id"] if hasattr(row, "keys") else row[3]
            except Exception:
                model_app_user_id = None

            # Multi-user safety: model must belong to the current user.
            if owner_id is not None and int(owner_id) != int(app_user_id):
                return jsonify({"error": "model does not belong to user"}), 403
            if model_app_user_id is not None and int(model_app_user_id) != int(app_user_id):
                return jsonify({"error": "model does not belong to user"}), 403

            try:
                cfg_raw = row["config_json"] if hasattr(row, "keys") else row[1]
                model_cfg = json.loads(cfg_raw or "{}") if cfg_raw else {}
            except Exception:
                model_cfg = {}

            # Support both shapes: {config:{fields:[...]}} or {fields:[...]}.
            cfg_obj = model_cfg.get("config") if isinstance(model_cfg, dict) else None
            if isinstance(cfg_obj, dict):
                fields_arr = cfg_obj.get("fields")
            else:
                fields_arr = model_cfg.get("fields") if isinstance(model_cfg, dict) else None

            if isinstance(fields_arr, list):
                for f in fields_arr:
                    if not isinstance(f, dict):
                        continue
                    code = str(f.get("code") or "").strip()
                    label = str(f.get("label") or "").strip()
                    if not code or not label:
                        continue
                    # Filter out generic labels like f01/f02 (or label==code)
                    c_lo = code.lower()
                    l_lo = label.lower()
                    if l_lo == c_lo:
                        continue
                    if l_lo.startswith("f") and l_lo[1:].isdigit():
                        continue
                    code_to_label[code] = label


        # Which sqlite file is this endpoint actually reading?
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


        # Detect actual f* columns in the table (handles f1 vs f01, etc.)
        table_cols = []
        try:
            info_rows = conn.execute("PRAGMA table_info(source_input)").fetchall()
            for r in info_rows:
                try:
                    col = r["name"]
                except Exception:
                    col = r[1]
                if col:
                    table_cols.append(str(col))
        except Exception:
            table_cols = []

        f_cols = []
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

        # Sort by numeric suffix
        f_cols.sort(key=lambda x: int(x[1:]))

        # Fallback only if none detected
        field_keys = f_cols if f_cols else [f"f{str(i).zfill(2)}" for i in range(1, 21)]

        total_records = 0
        sources = []
        field_stats = [{"key": k, "non_empty": 0} for k in field_keys]

        total_row = conn.execute("SELECT COUNT(*) AS n FROM source_input WHERE app_user_id=?", (app_user_id,)).fetchone()
        try:
            total_records = int(total_row["n"] or 0)
        except Exception:
            total_records = int(total_row[0] or 0)

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

        # Only compute field stats for columns that actually exist
        existing = set(table_cols)
        stat_keys = [k for k in field_keys if k in existing]

        if stat_keys:
            exprs = ", ".join(
                [
                    f"SUM(CASE WHEN TRIM(COALESCE({k}, '')) <> '' THEN 1 ELSE 0 END) AS {k}"
                    for k in stat_keys
                ]
            )
            stats_row = conn.execute(f"SELECT {exprs} FROM source_input WHERE app_user_id=?", (app_user_id,)).fetchone()

            if stats_row:
                out = []
                for k in field_keys:
                    if k not in existing:
                        out.append({"key": k, "non_empty": 0})
                        continue
                    try:
                        out.append({"key": k, "non_empty": int(stats_row[k] or 0)})
                    except Exception:
                        out.append({"key": k, "non_empty": 0})
                field_stats = out

    fields_with_data = sum(1 for x in field_stats if x["non_empty"] > 0)

    # Labeled field list for dashboard pills
    labeled_fields = []
    for fs in field_stats:
        k = str(fs.get("key") or "")
        if not k:
            continue
        lbl = code_to_label.get(k)
        if not lbl:
            continue
        labeled_fields.append({"code": k, "label": lbl, "non_empty": int(fs.get("non_empty") or 0)})

    field_pills = [x["label"] for x in labeled_fields]

    labeled_fields = []
    for fs in field_stats:
        k = str(fs.get("key") or "").strip()
        if not k:
            continue
        lbl = code_to_label.get(k)
        if not lbl:
            continue
        labeled_fields.append({"code": k, "label": lbl, "non_empty": int(fs.get("non_empty") or 0)})

    field_pills = [x["label"] for x in labeled_fields]

    return jsonify(
        {
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
    )


