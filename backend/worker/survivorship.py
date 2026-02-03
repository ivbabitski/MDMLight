from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from db.sqlite_db import get_conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_key(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _esc_pipe(s: Any) -> str:
    # keep record_id unambiguous even if source_name/source_id contain "|"
    return str(s or "").replace("|", "||")


def _make_record_id(source_name: Any, source_id: Any) -> str:
    return f"{_esc_pipe(source_name)}|{_esc_pipe(source_id)}"


def _parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _dt_key(dt: Optional[datetime], direction: str) -> float:
    # smaller == better, missing always last
    if dt is None:
        return math.inf
    ts = dt.timestamp()
    return -ts if direction == "desc" else ts


def _direction(x: Any, default: str = "desc") -> str:
    s = _norm_key(x)
    if s in ("asc", "ascending", "oldest", "oldest_wins"):
        return "asc"
    if s in ("desc", "descending", "newest", "newest_wins", "latest", "latest_wins"):
        return "desc"
    return default


def _strategy(x: Any, default: str = "updated_date") -> str:
    s = _norm_key(x)
    mapping = {
        "updated": "updated_date",
        "updated_at": "updated_date",
        "updated_date": "updated_date",
        "recency_updated_date": "updated_date",

        "created": "created_date",
        "created_at": "created_date",
        "created_date": "created_date",
        "recency_created_date": "created_date",

        "recency": "recency",
        "last_activity": "recency",

        "system": "system_priority",
        "system_priority": "system_priority",

        "created_user": "created_user_priority",
        "created_by": "created_user_priority",
        "created_user_priority": "created_user_priority",

        "updated_user": "updated_user_priority",
        "updated_by": "updated_user_priority",
        "updated_user_priority": "updated_user_priority",

        # ambiguous, default to updated-user priority
        "user_priority": "updated_user_priority",
    }
    return mapping.get(s, default)


def _priority_rank(value: Any, priority_list: Any) -> int:
    # lower is better, missing/unknown => huge (last)
    if not isinstance(priority_list, list) or len(priority_list) == 0:
        return 10**9
    pr = [_norm_key(v) for v in priority_list[:50]]
    m = {v: i for i, v in enumerate(pr) if v}
    v = _norm_key(value)
    if not v:
        return 10**9
    return m.get(v, 10**9)


@dataclass
class SourceRow:
    record_id: str
    source_name: str
    source_id: str
    fields20: List[Any]
    created_at: str
    created_by: str
    updated_at: Optional[str]
    updated_by: Optional[str]
    created_dt: Optional[datetime]
    updated_dt: Optional[datetime]
    completeness: int


def _load_source_rows(app_user_id: Optional[int] = None) -> Dict[str, SourceRow]:
    """
    source_input does NOT have an `id` column. We use _make_record_id(source_name, source_id) using "|" (escaped).
    """
    cols = ["source_id", "source_name"] + [f"f{str(i).zfill(2)}" for i in range(1, 21)] + [
        "created_at", "created_by", "updated_at", "updated_by"
    ]

    where = ""
    params: Tuple[Any, ...] = ()
    if app_user_id is not None:
        where = " WHERE app_user_id=?"
        params = (app_user_id,)

    sql = f"SELECT {', '.join(cols)} FROM source_input{where}"

    out: Dict[str, SourceRow] = {}
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    for r in rows:
        record_id = _make_record_id(r["source_name"], r["source_id"])
        f20 = [r[f"f{str(i).zfill(2)}"] for i in range(1, 21)]
        c_at = r["created_at"]
        u_at = r["updated_at"]

        cdt = _parse_dt(c_at)
        udt = _parse_dt(u_at)

        completeness = sum(1 for v in f20 if v is not None and str(v).strip() != "")

        out[record_id] = SourceRow(
            record_id=record_id,
            source_name=r["source_name"],
            source_id=r["source_id"],
            fields20=f20,
            created_at=c_at,
            created_by=r["created_by"],
            updated_at=u_at,
            updated_by=r["updated_by"],
            created_dt=cdt,
            updated_dt=udt,
            completeness=completeness,
        )

    return out



def _load_job_model(job_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute("SELECT model_json FROM match_job WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError(f"match_job not found for job_id={job_id}")
        return json.loads(row["model_json"]) if row["model_json"] else {}


def _load_clusters(job_id: str) -> Dict[str, List[str]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT cluster_id, record_id FROM recon_cluster WHERE job_id=?",
            (job_id,),
        ).fetchall()

    clusters: Dict[str, List[str]] = {}
    for r in rows:
        clusters.setdefault(r["cluster_id"], []).append(r["record_id"])
    return clusters


def _canonical_survivorship(model: Dict[str, Any]) -> Dict[str, Any]:
    s = model.get("survivorship") or {}
    if not isinstance(s, dict):
        s = {}

    mode = _norm_key(s.get("mode") or "simple")
    if mode not in ("simple", "advanced"):
        mode = "simple"

    base = {
        "mode": mode,
        "strategy": _strategy(s.get("strategy"), default="updated_date"),
        "direction": _direction(s.get("direction"), default="desc"),
        "system_priority": s.get("system_priority", []),
        "created_user_priority": s.get("created_user_priority", []),
        "updated_user_priority": s.get("updated_user_priority", []),
    }

    per_field = s.get("per_field") or s.get("field_strategies") or {}
    if not isinstance(per_field, dict):
        per_field = {}

    pf_norm: Dict[str, Any] = {}
    for k, v in per_field.items():
        fk = _norm_key(k)
        if not (len(fk) == 3 and fk.startswith("f") and fk[1:].isdigit()):
            continue
        if not isinstance(v, dict):
            continue
        pf_norm[fk] = {
            "strategy": _strategy(v.get("strategy"), default=base["strategy"]),
            "direction": _direction(v.get("direction"), default=base["direction"]),
            "system_priority": v.get("system_priority", base["system_priority"]),
            "created_user_priority": v.get("created_user_priority", base["created_user_priority"]),
            "updated_user_priority": v.get("updated_user_priority", base["updated_user_priority"]),
        }

    base["per_field"] = pf_norm
    return base


def _record_sort_key(
    r: SourceRow,
    strategy: str,
    direction: str,
    system_priority: Any,
    created_user_priority: Any,
    updated_user_priority: Any,
    field_value: Optional[Any] = "__ignore_missing__",
) -> Tuple:
    # Missing value should go last ONLY when selecting per-field winners
    if field_value == "__ignore_missing__":
        missing_value_rank = 0
    else:
        missing_value_rank = 1 if (field_value is None or str(field_value).strip() == "") else 0

    if strategy == "updated_date":
        crit = _dt_key(r.updated_dt, direction)
    elif strategy == "created_date":
        crit = _dt_key(r.created_dt, direction)
    elif strategy == "recency":
        # last_activity = max(updated, created)
        if r.updated_dt and r.created_dt:
            dt = max(r.updated_dt, r.created_dt)
        else:
            dt = r.updated_dt or r.created_dt
        crit = _dt_key(dt, direction)
    elif strategy == "system_priority":
        crit = _priority_rank(r.source_name, system_priority)
    elif strategy == "created_user_priority":
        crit = _priority_rank(r.created_by, created_user_priority)
    elif strategy == "updated_user_priority":
        crit = _priority_rank(r.updated_by, updated_user_priority)
    else:
        crit = 10**9

    return (
        missing_value_rank,
        crit,
        -r.completeness,
        _norm_key(r.source_name),
        _norm_key(r.source_id),
        _norm_key(r.record_id),
    )


def _pick_representative(members: List[SourceRow], cfg: Dict[str, Any]) -> SourceRow:
    return sorted(
        members,
        key=lambda r: _record_sort_key(
            r,
            strategy=cfg["strategy"],
            direction=cfg["direction"],
            system_priority=cfg.get("system_priority"),
            created_user_priority=cfg.get("created_user_priority"),
            updated_user_priority=cfg.get("updated_user_priority"),
            field_value="__ignore_missing__",
        ),
    )[0]


def _pick_field_value(members: List[SourceRow], field_idx: int, field_cfg: Dict[str, Any]) -> Tuple[Any, str]:
    best = sorted(
        members,
        key=lambda r: _record_sort_key(
            r,
            strategy=field_cfg["strategy"],
            direction=field_cfg["direction"],
            system_priority=field_cfg.get("system_priority"),
            created_user_priority=field_cfg.get("created_user_priority"),
            updated_user_priority=field_cfg.get("updated_user_priority"),
            field_value=r.fields20[field_idx],
        ),
    )[0]
    v = best.fields20[field_idx]
    if v is None or str(v).strip() == "":
        return None, best.record_id
    return v, best.record_id


def select_golden_records_for_job(
    job_id: str,
    model: Optional[Dict[str, Any]] = None,
    app_user_id: Optional[int] = None,
    model_id: Optional[str] = None,
    actor: str = "worker",
    mdm_source_name: str = "MDM",
) -> Tuple[Dict[str, str], List[Tuple]]:
    """
    Selection only (NO DB writes).
    Returns:
      - rep_by_cluster: {cluster_id: record_id}
      - golden_upserts: tuples for worker to INSERT/UPSERT into golden_record
    """
    if model is None:
        model = _load_job_model(job_id)

    match_threshold = float(model.get("model_threshold") or 0.85)
    surv = _canonical_survivorship(model)

    source_rows = _load_source_rows(app_user_id=app_user_id)  # record_id -> SourceRow
    clusters = _load_clusters(job_id)                         # cluster_id -> [record_id...]

    if not clusters:
        return {}, []

    now = _utc_now_iso()

    rep_by_cluster: Dict[str, str] = {}
    golden_upserts: List[Tuple] = []

    for cluster_id, record_ids in clusters.items():
        members = [source_rows[rid] for rid in record_ids if rid in source_rows]
        if not members:
            continue

        rep = _pick_representative(members, surv)
        rep_by_cluster[cluster_id] = rep.record_id

        lineage_json = None

        if surv["mode"] == "simple":
            golden_fields = rep.fields20
        else:
            golden_fields = [None] * 20
            lineage: Dict[str, Any] = {}

            for i in range(20):
                fkey = f"f{str(i+1).zfill(2)}"
                fcfg = surv["per_field"].get(fkey) or {
                    "strategy": surv["strategy"],
                    "direction": surv["direction"],
                    "system_priority": surv.get("system_priority"),
                    "created_user_priority": surv.get("created_user_priority"),
                    "updated_user_priority": surv.get("updated_user_priority"),
                }

                v, contrib_id = _pick_field_value(members, i, fcfg)
                golden_fields[i] = v
                lineage[fkey] = {"record_id": contrib_id, "strategy": fcfg["strategy"], "direction": fcfg["direction"]}

            lineage_json = json.dumps(lineage)

        survivorship_json = json.dumps(surv)

        golden_upserts.append(
            (
                cluster_id,  # master_id
                job_id,
                mdm_source_name,
                match_threshold,
                survivorship_json,
                rep.record_id,
                lineage_json,
                *golden_fields,
                now,
                actor,
                now,
                actor,
                app_user_id,
                model_id,
            )
        )

    return rep_by_cluster, golden_upserts
