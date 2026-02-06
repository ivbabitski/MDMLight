from __future__ import annotations

import json
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


def _make_record_id(app_user_id: int, model_id: str, source_name: Any, source_id: Any) -> str:
    # required stable composite identifier
    return f"{int(app_user_id)}|{_esc_pipe(model_id)}|{_esc_pipe(source_name)}|{_esc_pipe(source_id)}"


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


def _val_norm(v: Any) -> str:
    # "Exact" matching, but stable vs leading/trailing whitespace.
    if v is None:
        return ""
    return str(v).strip()


@dataclass
class ReconRow:
    cluster_id: str
    model_id: str
    source_name: str
    source_id: str
    fields20: List[Any]
    created_at: str
    created_by: str
    updated_at: Optional[str]
    updated_by: Optional[str]
    created_dt: Optional[datetime]
    updated_dt: Optional[datetime]


def _load_match_job(job_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM match_job WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError(f"match_job not found for job_id={job_id}")
        return dict(row)


def _resolve_model_id(job_id: str, model: Dict[str, Any]) -> str:
    # Preferred: explicit model_id in payload/model snapshot. Fallback: match_job.model_id.
    candidates = [
        model.get("model_id"),
        model.get("modelId"),
        model.get("id"),
        model.get("mdm_model_id"),
    ]
    for c in candidates:
        v = str(c or "").strip()
        if v:
            return v

    job = _load_match_job(job_id)
    v = str(job.get("model_id") or "").strip()
    if not v:
        raise ValueError("model_id is required (missing in model payload and match_job)")
    return v


def _resolve_match_threshold(model: Dict[str, Any]) -> float:
    # MDM model config uses matchThreshold; internal worker config sometimes uses model_threshold.
    v = model.get("matchThreshold")
    if v is None:
        v = model.get("model_threshold")
    try:
        return float(v or 0.85)
    except Exception:
        return 0.85


def _normalize_global_rule(x: Any) -> str:
    s = _norm_key(x)
    if s in ("recency_updated_date", "updated_date", "updated", "updated_at"):
        return "recency_updated_date"
    if s in ("recency_created_date", "created_date", "created", "created_at"):
        return "recency_created_date"
    if s in ("first_updated_date", "first_updated", "oldest_updated", "asc_updated_date"):
        return "first_updated_date"
    if s in ("first_created_date", "first_created", "oldest_created", "asc_created_date"):
        return "first_created_date"
    if s in ("system", "system_priority"):
        return "system"
    if s in ("specific_value_priority", "specificvaluepriority", "specific_value"):
        return "specific_value_priority"
    # default
    return "recency_updated_date"


def _canonical_survivorship_cfg(model: Dict[str, Any]) -> Dict[str, Any]:
    # Survivorship config is stored directly in the MDM model JSON.
    global_rule = _normalize_global_rule(model.get("globalRule"))

    system_priority = model.get("systemPriority")
    if not isinstance(system_priority, list):
        system_priority = []

    svp = model.get("specificValuePriority")
    if not isinstance(svp, list):
        svp = []

    return {
        "globalRule": global_rule,
        "systemPriority": system_priority,
        "specificValuePriority": svp,
    }


def _discover_cluster_ids(conn, app_user_id: int, model_id: str) -> List[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT cluster_id
        FROM recon_cluster
        WHERE app_user_id=?
          AND model_id=?
          AND match_status != 'exception'
        ORDER BY cluster_id
        """,
        (int(app_user_id), str(model_id)),
    ).fetchall()
    return [str(r["cluster_id"]) for r in rows]


def _load_cluster_members(conn, app_user_id: int, model_id: str, cluster_id: str) -> List[ReconRow]:
    cols = [
        "cluster_id", "model_id", "source_name", "source_id",
        *[f"f{str(i).zfill(2)}" for i in range(1, 21)],
        "created_at", "created_by", "updated_at", "updated_by"
    ]

    sql = f"SELECT {', '.join(cols)} FROM recon_cluster WHERE app_user_id=? AND model_id=? AND cluster_id=? AND match_status != 'exception'"

    rows = conn.execute(sql, (int(app_user_id), str(model_id), str(cluster_id))).fetchall()

    out: List[ReconRow] = []
    for r in rows:
        f20 = [r[f"f{str(i).zfill(2)}"] for i in range(1, 21)]
        c_at = r["created_at"]
        u_at = r["updated_at"]
        out.append(
            ReconRow(
                cluster_id=str(r["cluster_id"]),
                model_id=str(r["model_id"]),
                source_name=str(r["source_name"]),
                source_id=str(r["source_id"]),
                fields20=f20,
                created_at=str(c_at),
                created_by=str(r["created_by"]),
                updated_at=str(u_at) if u_at is not None else None,
                updated_by=str(r["updated_by"]) if r["updated_by"] is not None else None,
                created_dt=_parse_dt(c_at),
                updated_dt=_parse_dt(u_at),
            )
        )

    return out


def _final_tiebreaker(candidates: List[ReconRow]) -> ReconRow:
    # Deterministic final tie-breaker: (source_name, source_id)
    return sorted(candidates, key=lambda r: (_norm_key(r.source_name), _norm_key(r.source_id)))[0]


def _apply_fallback_chain(candidates: List[ReconRow], *, allow_updated: bool = True) -> Tuple[ReconRow, List[str]]:
    steps: List[str] = []
    remaining = list(candidates)

    # 1) Most recently updated (skip NULL updated_at)
    if allow_updated:
        with_updated = [r for r in remaining if r.updated_dt is not None]
        if with_updated:
            best_dt = max(r.updated_dt for r in with_updated if r.updated_dt is not None)
            remaining = [r for r in with_updated if r.updated_dt == best_dt]
            steps.append("most_recently_updated")
            if len(remaining) == 1:
                return remaining[0], steps

    # 2) Most recently created
    with_created = [r for r in remaining if r.created_dt is not None]
    if with_created:
        best_cdt = max(r.created_dt for r in with_created if r.created_dt is not None)
        remaining = [r for r in with_created if r.created_dt == best_cdt]
    steps.append("most_recently_created")
    if len(remaining) == 1:
        return remaining[0], steps

    # 3) Final deterministic source key tie-break
    steps.append("source_name+source_id")
    return _final_tiebreaker(remaining), steps


def _pick_recency_updated(candidates: List[ReconRow]) -> Tuple[ReconRow, List[str]]:
    # Primary: most recent updated_at (skip NULL updated_at)
    with_updated = [r for r in candidates if r.updated_dt is not None]
    if with_updated:
        best_dt = max(r.updated_dt for r in with_updated if r.updated_dt is not None)
        top = [r for r in with_updated if r.updated_dt == best_dt]
        if len(top) == 1:
            return top[0], []
        # updated already used; go straight to created + id tie-break
        return _apply_fallback_chain(top, allow_updated=False)

    # If all updated_at are NULL: fallback chain (updated step skipped automatically)
    return _apply_fallback_chain(candidates, allow_updated=True)


def _pick_recency_created(candidates: List[ReconRow]) -> Tuple[ReconRow, List[str]]:
    # Primary: most recent created_at
    with_created = [r for r in candidates if r.created_dt is not None]
    if with_created:
        best_cdt = max(r.created_dt for r in with_created if r.created_dt is not None)
        top = [r for r in with_created if r.created_dt == best_cdt]
        if len(top) == 1:
            return top[0], []
        return _apply_fallback_chain(top, allow_updated=True)
    return _apply_fallback_chain(candidates, allow_updated=True)


def _pick_first_updated(candidates: List[ReconRow]) -> Tuple[ReconRow, List[str]]:
    # Primary: first updated_at (oldest). Skip NULL updated_at.
    with_updated = [r for r in candidates if r.updated_dt is not None]
    if with_updated:
        best_dt = min(r.updated_dt for r in with_updated if r.updated_dt is not None)
        top = [r for r in with_updated if r.updated_dt == best_dt]
        if len(top) == 1:
            return top[0], []
        # updated already used; go straight to created + id tie-break
        return _apply_fallback_chain(top, allow_updated=False)
    return _apply_fallback_chain(candidates, allow_updated=True)


def _pick_first_created(candidates: List[ReconRow]) -> Tuple[ReconRow, List[str]]:
    # Primary: first created_at (oldest). created_at is NOT NULL in recon_cluster, but parse can fail.
    with_created = [r for r in candidates if r.created_dt is not None]
    if with_created:
        best_cdt = min(r.created_dt for r in with_created if r.created_dt is not None)
        top = [r for r in with_created if r.created_dt == best_cdt]
        if len(top) == 1:
            return top[0], []
        return _apply_fallback_chain(top, allow_updated=True)
    return _apply_fallback_chain(candidates, allow_updated=True)


def _pick_system_priority(candidates: List[ReconRow], system_priority: List[Any]) -> Tuple[ReconRow, List[str]]:
    # First system in the list is highest priority.
    pr = [_norm_key(x) for x in (system_priority or []) if str(x or "").strip()]
    if not pr:
        return _apply_fallback_chain(candidates, allow_updated=True)

    rank = {name: i for i, name in enumerate(pr)}

    listed = [r for r in candidates if _norm_key(r.source_name) in rank]
    if listed:
        best_rank = min(rank[_norm_key(r.source_name)] for r in listed)
        top = [r for r in listed if rank[_norm_key(r.source_name)] == best_rank]
        if len(top) == 1:
            return top[0], []
        return _apply_fallback_chain(top, allow_updated=True)

    # No listed systems present -> pure fallback chain across all candidates
    return _apply_fallback_chain(candidates, allow_updated=True)


def _get_field_value(r: ReconRow, field_code: str) -> Any:
    fc = str(field_code or "").strip()
    if not fc:
        return None

    nfc = _norm_key(fc)
    if nfc in ("source_name", "source"):
        return r.source_name
    if nfc in ("source_id", "id"):
        return r.source_id
    if nfc == "created_by":
        return r.created_by
    if nfc == "updated_by":
        return r.updated_by

    # f01..f20
    if len(nfc) == 3 and nfc.startswith("f") and nfc[1:].isdigit():
        idx = int(nfc[1:]) - 1
        if 0 <= idx < 20:
            return r.fields20[idx]

    return None


def _canonical_specific_value_priority(raw: Any) -> Tuple[List[str], Dict[str, List[str]]]:
    # raw: ordered list of {fieldCode, value} rows
    field_order: List[str] = []
    value_map: Dict[str, List[str]] = {}

    if not isinstance(raw, list):
        return field_order, value_map

    for row in raw:
        if not isinstance(row, dict):
            continue
        fc = row.get("fieldCode")
        if fc is None:
            fc = row.get("field")
        if fc is None:
            fc = row.get("code")
        val = row.get("value")

        fc_s = str(fc or "").strip()
        val_s = _val_norm(val)
        if not fc_s or not val_s:
            # Ignore rows with missing fieldCode or missing value
            continue

        if fc_s not in value_map:
            value_map[fc_s] = []
            field_order.append(fc_s)

        # Keep list order; duplicates don't add meaning
        if val_s not in value_map[fc_s]:
            value_map[fc_s].append(val_s)

    return field_order, value_map


def _pick_specific_value_priority(candidates: List[ReconRow], specific_value_priority: Any) -> Tuple[ReconRow, List[str]]:
    field_order, value_map = _canonical_specific_value_priority(specific_value_priority)
    if not field_order:
        return _apply_fallback_chain(candidates, allow_updated=True)

    remaining = list(candidates)

    for fc in field_order:
        prefs = value_map.get(fc) or []
        if not prefs:
            continue

        # Find the highest priority value that matches at least one remaining record
        best_match: Optional[str] = None
        best_rows: List[ReconRow] = []

        for pv in prefs:
            pv_s = _val_norm(pv)
            if not pv_s:
                continue
            hits = [r for r in remaining if _val_norm(_get_field_value(r, fc)) == pv_s]
            if hits:
                best_match = pv_s
                best_rows = hits
                break

        if best_match is None:
            # No preferred value matches for this field -> move to next field
            continue

        remaining = best_rows
        if len(remaining) == 1:
            return remaining[0], []

    if len(remaining) == 1:
        return remaining[0], []

    return _apply_fallback_chain(remaining, allow_updated=True)


def _pick_winner(candidates: List[ReconRow], surv_cfg: Dict[str, Any]) -> Tuple[ReconRow, List[str]]:
    gr = surv_cfg.get("globalRule")
    if gr == "recency_updated_date":
        return _pick_recency_updated(candidates)
    if gr == "recency_created_date":
        return _pick_recency_created(candidates)
    if gr == "first_updated_date":
        return _pick_first_updated(candidates)
    if gr == "first_created_date":
        return _pick_first_created(candidates)
    if gr == "system":
        return _pick_system_priority(candidates, surv_cfg.get("systemPriority") or [])
    if gr == "specific_value_priority":
        return _pick_specific_value_priority(candidates, surv_cfg.get("specificValuePriority") or [])

    # default
    return _pick_recency_updated(candidates)


def run_survivorship(job_id: str, app_user_id: int, model: Dict[str, Any], actor: str = "survivorship") -> Dict[str, Any]:
    """
    Independent survivorship job. Reads recon_cluster and writes golden_record.

    Inputs (as required): job_id, app_user_id, model config JSON.
    """
    if not str(job_id or "").strip():
        raise ValueError("job_id is required")

    if app_user_id is None:
        raise ValueError("app_user_id is required")

    if not isinstance(model, dict):
        raise ValueError("model config must be a dict")

    model_id = _resolve_model_id(job_id, model)
    match_threshold = _resolve_match_threshold(model)
    surv_cfg = _canonical_survivorship_cfg(model)

    now = _utc_now_iso()

    upserted = 0
    processed = 0

    cols = [
        "master_id", "job_id", "model_id", "source_name", "match_threshold", "survivorship_json",
        "representative_record_id", "representative_source_name", "representative_source_id", "lineage_json",
        *[f"f{str(i).zfill(2)}" for i in range(1, 21)],
        "created_at", "created_by", "updated_at", "updated_by", "app_user_id"
    ]

    placeholders = ", ".join(["?"] * len(cols))

    update_set = [
        "job_id=excluded.job_id",
        "model_id=excluded.model_id",
        "source_name=excluded.source_name",
        "match_threshold=excluded.match_threshold",
        "survivorship_json=excluded.survivorship_json",
        "representative_record_id=excluded.representative_record_id",
        "representative_source_name=excluded.representative_source_name",
        "representative_source_id=excluded.representative_source_id",
        "lineage_json=excluded.lineage_json",
    ]
    for i in range(1, 21):
        update_set.append(f"f{str(i).zfill(2)}=excluded.f{str(i).zfill(2)}")
    update_set.extend([
        "updated_at=excluded.updated_at",
        "updated_by=excluded.updated_by",
        "app_user_id=excluded.app_user_id",
    ])

    sql = f"""
    INSERT INTO golden_record ({', '.join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(master_id) DO UPDATE SET
      {', '.join(update_set)}
    """

    with get_conn() as conn:
        cluster_ids = _discover_cluster_ids(conn, app_user_id=app_user_id, model_id=model_id)

        for cluster_id in cluster_ids:
            members = _load_cluster_members(conn, app_user_id=app_user_id, model_id=model_id, cluster_id=cluster_id)
            if not members:
                continue

            processed += 1

            winner, fallbacks = _pick_winner(members, surv_cfg)
            rep_record_id = _make_record_id(app_user_id, model_id, winner.source_name, winner.source_id)

            survivorship_json = json.dumps({
                "globalRule": surv_cfg.get("globalRule"),
                "systemPriority": surv_cfg.get("systemPriority") if surv_cfg.get("globalRule") == "system" else None,
                "specificValuePriority": surv_cfg.get("specificValuePriority") if surv_cfg.get("globalRule") == "specific_value_priority" else None,
                "exception_records_excluded": True,
                "fallbacks_applied": fallbacks,
                "winner": {"source_name": winner.source_name, "source_id": winner.source_id},
            })

            lineage_json = json.dumps({
                "eligible_member_count": len(members),
                "eligible_members": [{"source_name": m.source_name, "source_id": m.source_id} for m in members],
                "winner": {"source_name": winner.source_name, "source_id": winner.source_id},
                "exception_records_excluded": True,
            })

            values: List[Any] = [
                str(cluster_id),               # master_id
                str(job_id),                   # job_id
                str(model_id),                 # model_id
                "MDM",                        # source_name
                float(match_threshold),        # match_threshold
                survivorship_json,
                rep_record_id,
                winner.source_name,
                winner.source_id,
                lineage_json,
                *winner.fields20,
                now,
                str(actor),
                now,
                str(actor),
                int(app_user_id),
            ]

            conn.execute(sql, tuple(values))
            upserted += 1

    return {
        "ok": True,
        "job_id": str(job_id),
        "app_user_id": int(app_user_id),
        "model_id": str(model_id),
        "clusters_discovered": len(cluster_ids),
        "clusters_processed": processed,
        "golden_upserted": upserted,
    }
