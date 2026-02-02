import os
import json
import time
import uuid
import sqlite3
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

from rapidfuzz.distance import Levenshtein

from db.sqlite_db import get_conn, init_all_tables
from worker.survivorship import select_golden_records_for_job

POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "1.0"))

# Job-level concurrency (processes). Keep low unless you know what you're doing.
JOB_WORKERS = int(os.environ.get("WORKER_JOB_WORKERS", "1"))

# Matching tuning
MAX_BUCKET_SIZE = int(os.environ.get("MATCH_MAX_BUCKET_SIZE", "500"))
BUCKETS_PER_TASK = int(os.environ.get("MATCH_BUCKETS_PER_TASK", "50"))

# Total CPU workers budget (split across concurrent jobs)
MATCH_WORKERS_TOTAL = int(os.environ.get("MATCH_WORKERS", str(os.cpu_count() or 2)))

# Globals shared via fork inside ONE job process
_G_RECORDS = None
_G_CFG = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s


def _to_0_1(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
    except Exception:
        return default
    if v > 1.0:
        v = v / 100.0
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def _is_f_code(code: Any) -> bool:
    if not isinstance(code, str):
        return False
    c = code.strip()
    if len(c) != 3 or c[0].lower() != "f" or not c[1:].isdigit():
        return False
    n = int(c[1:])
    return 1 <= n <= 20


def _field_index(fname: str) -> int:
    # "f01" -> 0
    if not _is_f_code(fname):
        raise ValueError(f"invalid field name: {fname}")
    return int(fname[1:]) - 1


def _prefix_len_from_threshold(t: float) -> int:
    # conservative, simple
    if t >= 0.95:
        return 5
    if t >= 0.90:
        return 4
    if t >= 0.80:
        return 3
    if t >= 0.60:
        return 2
    return 1


@dataclass
class Record:
    record_id: str            # unique per-job id: f"{source_name}::{source_id}"
    source_name: str
    source_id: str
    raw20: List[Any]          # len 20: f01..f20
    norm_selected: List[str]  # len = selected_fields


def _init_globals(records: List[Record], cfg: Dict[str, Any]) -> None:
    global _G_RECORDS, _G_CFG
    _G_RECORDS = records
    _G_CFG = cfg


def _pair_score(i: int, j: int) -> Tuple[bool, float]:
    """
    Returns (is_match, score) using:
      - per-field thresholds (strict reject)
      - weighted average over present fields
      - conservative similarity: 1 - d/minLen
    """
    recs: List[Record] = _G_RECORDS
    cfg: Dict[str, Any] = _G_CFG

    a = recs[i].norm_selected
    b = recs[j].norm_selected

    weights = cfg["weights"]          # list[float] aligned with selected_fields
    thresholds = cfg["thresholds"]    # list[float]
    model_T = cfg["model_T"]

    sims = []
    ws = []

    for idx in range(len(a)):
        va = a[idx]
        vb = b[idx]
        if not va or not vb:
            continue

        min_len = min(len(va), len(vb))
        if min_len == 0:
            continue

        t = thresholds[idx]
        allowed = int((1.0 - t) * min_len)

        # Early cutoff: if distance > allowed, rapidfuzz returns allowed+1
        d = Levenshtein.distance(va, vb, score_cutoff=allowed)
        if d > allowed:
            return False, 0.0  # strict per-field threshold reject

        sim = 1.0 - (d / float(min_len))
        sims.append(sim)
        ws.append(weights[idx])

    if not sims:
        return False, 0.0

    wsum = sum(ws)
    if wsum <= 0:
        return False, 0.0

    score = 0.0
    for sim, w in zip(sims, ws):
        score += (w / wsum) * sim

    return (score >= model_T), score


def _process_bucket_chunk(bucket_lists: List[List[int]]) -> Tuple[List[Tuple[int, int]], int, int, List[Tuple[int, int, float]]]:
    """
    Returns (match_edges, pairs_scored, matches_found, possible_pairs)
    possible_pairs are (a, b, score) where score >= possible_T but < model_T
    """
    edges: List[Tuple[int, int]] = []
    possibles: List[Tuple[int, int, float]] = []
    pairs_scored = 0
    matches_found = 0

    cfg: Dict[str, Any] = _G_CFG or {}
    possible_T = float(cfg.get("possible_T", 0.0) or 0.0)

    seen = set()  # dedupe inside this task

    for ids in bucket_lists:
        m = len(ids)
        if m < 2:
            continue
        # O(m^2) inside bucket (bounded by MAX_BUCKET_SIZE)
        for x in range(m):
            i = ids[x]
            for y in range(x + 1, m):
                j = ids[y]
                a, b = (i, j) if i < j else (j, i)
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)

                pairs_scored += 1
                ok, score = _pair_score(a, b)
                if ok:
                    edges.append(key)
                    matches_found += 1
                elif possible_T > 0.0 and score >= possible_T:
                    possibles.append((a, b, float(score)))

    return edges, pairs_scored, matches_found, possibles



class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            p = self.parent[p]
        # path compress
        while x != p:
            nx = self.parent[x]
            self.parent[x] = p
            x = nx
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ----------------------------
# DB helpers (job + model)
# ----------------------------

def _set_job_status(job_id: str, status: str, message: Optional[str] = None, **metrics) -> None:
    now = _utc_now_iso()

    # Use COALESCE so we don't overwrite started_at on metric updates
    cols = ["status=?", "updated_at=?"]
    vals: List[Any] = [status, now]

    if message is not None:
        cols.append("message=?")
        vals.append(message)

    for k, v in metrics.items():
        cols.append(f"{k}=?")
        vals.append(v)

    if status == "running":
        cols.append("started_at=COALESCE(started_at, ?)")
        vals.append(now)
    if status in ("completed", "failed"):
        cols.append("finished_at=COALESCE(finished_at, ?)")
        vals.append(now)

    vals.append(job_id)

    with get_conn() as conn:
        conn.execute(f"UPDATE match_job SET {', '.join(cols)} WHERE job_id=?", vals)


def _claim_queued_jobs(limit: int) -> List[Dict[str, Any]]:
    """
    Atomically claim up to `limit` queued jobs.
    Enforces: no parallel runs for the same (app_user_id, model_id).
    """
    now = _utc_now_iso()
    picked: List[Dict[str, Any]] = []

    with get_conn() as conn:
        # serialize claims across worker instances
        conn.execute("BEGIN IMMEDIATE")

        # running pairs (including other workers)
        running = conn.execute(
            """
            SELECT DISTINCT app_user_id, model_id
            FROM match_job
            WHERE status='running'
              AND app_user_id IS NOT NULL
              AND model_id IS NOT NULL
            """
        ).fetchall()
        running_pairs = {(r["app_user_id"], r["model_id"]) for r in running}

        # queued jobs in order
        rows = conn.execute(
            """
            SELECT job_id, app_user_id, model_id, model_json
            FROM match_job
            WHERE status='queued'
            ORDER BY created_at ASC
            """
        ).fetchall()

        for r in rows:
            if len(picked) >= limit:
                break
            app_user_id = r["app_user_id"]
            model_id = r["model_id"]
            if app_user_id is None or model_id is None:
                # can't safely run without scoping
                continue
            pair = (app_user_id, model_id)
            if pair in running_pairs:
                continue

            picked.append(
                {
                    "job_id": r["job_id"],
                    "app_user_id": app_user_id,
                    "model_id": model_id,
                    "model_json": r["model_json"],
                }
            )
            running_pairs.add(pair)

        # mark claimed as running (reserve)
        for j in picked:
            conn.execute(
                """
                UPDATE match_job
                SET status='running', updated_at=?, started_at=COALESCE(started_at, ?)
                WHERE job_id=? AND status='queued'
                """,
                (now, now, j["job_id"]),
            )

    return picked


def _load_model_config_from_mdm_models(model_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT config_json FROM mdm_models WHERE id=? AND deleted_at IS NULL LIMIT 1",
            (model_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"mdm_models not found (or deleted): id={model_id}")
        return json.loads(row["config_json"]) if row["config_json"] else {}


def _normalize_model_json(raw: Any) -> Dict[str, Any]:
    """
    Accepts either:
      - your wizard payload: {app_user_id, config:{fields, matchFieldCodes, matchThreshold,...}}
      - just the config object
      - internal worker format: {selected_fields, weights, field_thresholds, model_threshold, survivorship, blocking}
    Returns internal worker format.
    """
    if raw is None:
        return {}

    if isinstance(raw, str):
        raw = json.loads(raw)

    if not isinstance(raw, dict):
        return {}

    # Internal worker schema passthrough
    if "selected_fields" in raw and ("weights" in raw or "field_thresholds" in raw):
        return raw

    cfg = raw.get("config") if isinstance(raw.get("config"), dict) else raw

    # If cfg still looks like internal, pass
    if "selected_fields" in cfg and ("weights" in cfg or "field_thresholds" in cfg):
        return cfg

    fields = cfg.get("fields") or []
    if not isinstance(fields, list):
        fields = []

    match_fields = cfg.get("matchFieldCodes") or []
    if not isinstance(match_fields, list):
        match_fields = []

    # build field defs by code
    fdefs: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        if not isinstance(f, dict):
            continue
        code = f.get("code")
        if not _is_f_code(code):
            continue
        if f.get("include") is False:
            continue
        fdefs[str(code)] = f

    selected_fields = [c for c in match_fields if _is_f_code(c) and c in fdefs]

    possible_T = _to_0_1(cfg.get("possibleThreshold"), 0.0)
    weights: Dict[str, float] = {}
    thresholds: Dict[str, float] = {}

    for c in selected_fields:
        fd = fdefs[c]
        w = float(fd.get("weight", 1.0) or 0.0)
        t = _to_0_1(fd.get("matchThreshold"), possible_T)
        weights[c] = w
        thresholds[c] = t

    model_T = _to_0_1(cfg.get("matchThreshold"), 0.85)

    # Survivorship mapping from your config (basic)
    adv = bool(cfg.get("advanced", False))
    surv = {
        "mode": "advanced" if adv else "simple",
        "strategy": cfg.get("globalRule") or "recency_updated_date",
        "direction": "desc",
        "system_priority": cfg.get("systemPriority") or [],
        "updated_user_priority": cfg.get("userPriority") or [],
        "created_user_priority": cfg.get("userPriority") or [],
    }

    # Per-field survivorship rules (if present)
    per_field: Dict[str, Any] = {}
    for f in fields:
        if not isinstance(f, dict):
            continue
        code = f.get("code")
        if not _is_f_code(code):
            continue
        rule = f.get("rule")
        if rule:
            per_field[str(code)] = {"strategy": rule}
    if per_field:
        surv["per_field"] = per_field

    # Blocking defaults: use the top-2 weighted selected fields
    pairs = [(c, weights.get(c, 0.0)) for c in selected_fields]
    pairs.sort(key=lambda x: x[1], reverse=True)
    default_block = [p[0] for p in pairs[:2] if p[0]]

    blocking = cfg.get("blocking") if isinstance(cfg.get("blocking"), dict) else {}
    if not blocking:
        blocking = {
            "blocking_fields": default_block,
            "blocking_passes": "composite+single",
            "max_bucket_size": MAX_BUCKET_SIZE,
        }

    return {
        "selected_fields": selected_fields,
        "weights": weights,
        "field_thresholds": thresholds,
        "model_threshold": model_T,
        "possible_threshold": possible_T,
        "blocking": blocking,
        "survivorship": surv,
    }


def _build_cfg(model: Dict[str, Any]) -> Dict[str, Any]:
    selected_fields = model.get("selected_fields", [])
    weights_map = model.get("weights", {}) or {}
    thresh_map = model.get("field_thresholds", {}) or {}
    model_T = _to_0_1(model.get("model_threshold"), 0.85)
    possible_T = _to_0_1(model.get("possible_threshold"), 0.0)

    sel: List[str] = []
    weights: List[float] = []
    thresholds: List[float] = []

    for f in selected_fields:
        if not _is_f_code(f):
            continue
        w = float(weights_map.get(f, 1.0) or 0.0)
        if w <= 0:
            continue
        sel.append(f)
        weights.append(w)
        thresholds.append(_to_0_1(thresh_map.get(f, 0.0), 0.0))

    # Blocking
    blocking = model.get("blocking", {}) or {}
    max_bucket = int(blocking.get("max_bucket_size", MAX_BUCKET_SIZE))
    passes = blocking.get("blocking_passes", "composite+single")  # composite_only | composite+single
    block_fields = blocking.get("blocking_fields")

    # pick default blocking fields: top-2 by weight
    if not isinstance(block_fields, list) or len(block_fields) == 0:
        pairs = list(zip(sel, weights))
        pairs.sort(key=lambda x: x[1], reverse=True)
        block_fields = [p[0] for p in pairs[:2]] if pairs else []

    return {
        "selected_fields": sel,
        "selected_indices": [_field_index(f) for f in sel],
        "weights": weights,
        "thresholds": thresholds,
        "model_T": model_T,
        "possible_T": possible_T,
        "max_bucket_size": max_bucket,
        "blocking_fields": block_fields[:2],
        "blocking_passes": passes,
    }


def _load_source_input(app_user_id: int) -> List[Tuple]:
    cols = ["source_id", "source_name"] + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
    sql = f"SELECT {', '.join(cols)} FROM source_input WHERE app_user_id=?"
    with get_conn() as conn:
        rows = conn.execute(sql, (app_user_id,)).fetchall()
        return [tuple(r) for r in rows]


def _build_records(raw_rows: List[Tuple], cfg: Dict[str, Any]) -> List[Record]:
    sel_idx = cfg["selected_indices"]
    records: List[Record] = []
    for row in raw_rows:
        source_id = str(row[0])
        source_name = str(row[1])
        record_id = f"{source_name}::{source_id}"
        raw20 = list(row[2:22])  # f01..f20
        norm_selected = [_norm(raw20[i]) for i in sel_idx]
        records.append(Record(record_id, source_name, source_id, raw20, norm_selected))
    return records



def _make_buckets(records: List[Record], cfg: Dict[str, Any]) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """
    Candidate buckets built from blocking_fields using prefix keys.
    Returns (bucket_lists, exceptions)
    """
    block_fields = cfg["blocking_fields"]
    passes = cfg["blocking_passes"]
    max_bucket = cfg["max_bucket_size"]

    if not block_fields:
        return [], [{"reason": "no_block_fields"}]

    block_idx = [_field_index(f) for f in block_fields]

    thresh_map = {f: t for f, t in zip(cfg["selected_fields"], cfg["thresholds"])}
    prefix_lens = []
    for f in block_fields:
        t = thresh_map.get(f, 0.8)
        prefix_lens.append(_prefix_len_from_threshold(t))

    buckets: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        pfx = []
        for bi, plen in zip(block_idx, prefix_lens):
            v = _norm(r.raw20[bi])
            p = v[:plen] if v else ""
            pfx.append(p)

        # Composite key
        if len(pfx) >= 2 and pfx[0] and pfx[1]:
            k = f"{block_fields[0]}:{pfx[0]}|{block_fields[1]}:{pfx[1]}"
            buckets.setdefault(k, []).append(i)

        if passes == "composite+single":
            for f, p in zip(block_fields, pfx):
                if p:
                    k = f"{f}:{p}"
                    buckets.setdefault(k, []).append(i)

    bucket_lists: List[List[int]] = []
    exceptions: List[Dict[str, Any]] = []

    for k, ids in buckets.items():
        if len(ids) < 2:
            continue
        if len(ids) > max_bucket:
            exceptions.append({"reason": "hot_bucket_skipped", "key": k, "size": len(ids)})
            continue
        bucket_lists.append(ids)

    return bucket_lists, exceptions


def _load_cluster_map(app_user_id: int, model_id: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Returns:
      {(source_name, source_id): (cluster_id, first_seen_at)}
    """
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    with get_conn() as conn:
        try:
            rows = conn.execute(
                """
                SELECT source_name, source_id, cluster_id, first_seen_at
                FROM cluster_map
                WHERE app_user_id=? AND model_id=?
                """,
                (app_user_id, model_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return out

        for r in rows:
            out[(r["source_name"], r["source_id"])] = (r["cluster_id"], r["first_seen_at"])

    return out


def _assign_stable_cluster_ids(
    records: List[Record],
    clusters: Dict[int, List[int]],
    existing_map: Dict[Tuple[str, str], Tuple[str, str]],
) -> Dict[int, str]:
    """
    Stable ID assignment with split/merge handling:

    - Build overlaps between NEW clusters and OLD cluster_ids (from cluster_map)
    - Greedy assign old cluster_ids to new clusters by highest overlap, so each old id is used at most once
    - Remaining new clusters get new UUIDs
    """
    # root -> {old_cid: (count, earliest_first_seen)}
    overlaps: Dict[int, Dict[str, Tuple[int, str]]] = {}

    for root, members in clusters.items():
        m: Dict[str, Tuple[int, str]] = {}
        for idx in members:
            r = records[idx]
            key = (r.source_name, r.source_id)
            if key not in existing_map:
                continue
            old_cid, first_seen = existing_map[key]
            if not old_cid:
                continue
            if old_cid not in m:
                m[old_cid] = (1, first_seen or "")
            else:
                cnt, fs = m[old_cid]
                # keep earliest first_seen across members
                best_fs = fs
                if first_seen and (not fs or first_seen < fs):
                    best_fs = first_seen
                m[old_cid] = (cnt + 1, best_fs)
        overlaps[root] = m

    candidates: List[Tuple[int, str, str, int]] = []
    # sort key: (-count, earliest_first_seen, old_cid) and carry root
    for root, m in overlaps.items():
        for old_cid, (cnt, first_seen) in m.items():
            candidates.append((cnt, first_seen or "", old_cid, root))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    assigned_roots = set()
    assigned_old = set()
    cluster_ids: Dict[int, str] = {}

    for cnt, first_seen, old_cid, root in candidates:
        if root in assigned_roots:
            continue
        if old_cid in assigned_old:
            continue
        cluster_ids[root] = old_cid
        assigned_roots.add(root)
        assigned_old.add(old_cid)

    for root in clusters.keys():
        if root not in cluster_ids:
            cluster_ids[root] = str(uuid.uuid4())

    return cluster_ids


def _write_outputs(
    job_id: str,
    app_user_id: int,
    model_id: str,
    records: List[Record],
    clusters: Dict[int, List[int]],
    cluster_ids: Dict[int, str],
) -> None:
    now = _utc_now_iso()

    recon_rows = []
    map_rows = []

    for root, members in clusters.items():
        cid = cluster_ids[root]
        cluster_size = len(members)

        for idx in members:
            r = records[idx]
            recon_rows.append(
                (
                    job_id, cid,
                    r.record_id, r.source_name, r.source_id,
                    cluster_size,
                    0,   # representative filled after survivorship
                    now,
                    app_user_id,
                    model_id,
                )
            )

            map_rows.append(
                (
                    app_user_id, model_id,
                    r.source_name, r.source_id,
                    cid,
                    now, now,
                )
            )

    with get_conn() as conn:
        # Clear old outputs for this job_id (if rerun)
        conn.execute("DELETE FROM recon_cluster WHERE job_id=?", (job_id,))

        # Write recon_cluster
        conn.executemany(
            """
            INSERT INTO recon_cluster (
              job_id, cluster_id,
              record_id, source_name, source_id,
              cluster_size, is_representative, created_at,
              app_user_id, model_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            recon_rows
        )

        # Upsert cluster_map (persistent, scoped by user+model)
        conn.executemany(
            """
            INSERT INTO cluster_map (
              app_user_id, model_id,
              source_name, source_id,
              cluster_id,
              first_seen_at, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(app_user_id, model_id, source_name, source_id) DO UPDATE SET
              cluster_id = excluded.cluster_id,
              last_seen_at = excluded.last_seen_at
            """,
            map_rows
        )


def _write_survivorship(
    job_id: str,
    app_user_id: int,
    model_id: str,
    model: Dict[str, Any],
) -> None:
    """
    - Picks cluster representatives
    - UPSERTs golden_record
    - Updates recon_cluster.is_representative
    """
    rep_by_cluster, golden_upserts = select_golden_records_for_job(
        job_id=job_id,
        model=model,
        app_user_id=app_user_id,
        model_id=model_id,
        actor="worker",
        mdm_source_name="MDM",
    )

    if not golden_upserts:
        return

    with get_conn() as conn:
        # mark all non-representative first
        conn.execute("UPDATE recon_cluster SET is_representative=0 WHERE job_id=?", (job_id,))

        # set reps
        rep_updates = [(job_id, rid) for rid in rep_by_cluster.values()]
        conn.executemany(
            "UPDATE recon_cluster SET is_representative=1 WHERE job_id=? AND record_id=?",
            rep_updates,
        )

        # upsert golden_record
        conn.executemany(
            """
            INSERT INTO golden_record (
              master_id, job_id,
              source_name,
              match_threshold, survivorship_json,
              representative_record_id, lineage_json,
              f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,
              f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,
              created_at, created_by,
              updated_at, updated_by,
              app_user_id, model_id
            ) VALUES (
              ?, ?, ?, ?, ?, ?, ?,
              ?,?,?,?,?,?,?,?,?,?,
              ?,?,?,?,?,?,?,?,?,?,
              ?, ?, ?, ?,
              ?, ?
            )
            ON CONFLICT(master_id) DO UPDATE SET
              job_id = excluded.job_id,
              source_name = excluded.source_name,
              match_threshold = excluded.match_threshold,
              survivorship_json = excluded.survivorship_json,
              representative_record_id = excluded.representative_record_id,
              lineage_json = excluded.lineage_json,
              f01=excluded.f01,f02=excluded.f02,f03=excluded.f03,f04=excluded.f04,f05=excluded.f05,
              f06=excluded.f06,f07=excluded.f07,f08=excluded.f08,f09=excluded.f09,f10=excluded.f10,
              f11=excluded.f11,f12=excluded.f12,f13=excluded.f13,f14=excluded.f14,f15=excluded.f15,
              f16=excluded.f16,f17=excluded.f17,f18=excluded.f18,f19=excluded.f19,f20=excluded.f20,
              updated_at = excluded.updated_at,
              updated_by = excluded.updated_by,
              app_user_id = excluded.app_user_id,
              model_id = excluded.model_id
            """,
            golden_upserts,
        )


def run_one_job(job_id: str, match_workers: int) -> None:
    # Load job row
    with get_conn() as conn:
        row = conn.execute(
            "SELECT job_id, app_user_id, model_id, model_json FROM match_job WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"match_job not found: job_id={job_id}")

    app_user_id = row["app_user_id"]
    model_id = row["model_id"]
    model_json_raw = row["model_json"]

    if app_user_id is None or model_id is None:
        _set_job_status(job_id, "failed", "job missing app_user_id or model_id")
        return

    # Model config: prefer job snapshot; else load from mdm_models
    try:
        raw_model = json.loads(model_json_raw) if model_json_raw else None
    except Exception:
        raw_model = None

    if not raw_model:
        raw_model = _load_model_config_from_mdm_models(model_id)

    model = _normalize_model_json(raw_model)
    if not model:
        _set_job_status(job_id, "failed", "model_json/config is empty or invalid")
        return

    cfg = _build_cfg(model)
    if not cfg["selected_fields"]:
        _set_job_status(job_id, "failed", "no match fields with positive weight", total_records=0)
        return

    raw_rows = _load_source_input(app_user_id)
    if not raw_rows:
        _set_job_status(job_id, "failed", f"source_input empty for app_user_id={app_user_id}", total_records=0)
        return

    records = _build_records(raw_rows, cfg)
    total_records = len(records)

    _set_job_status(job_id, "running", None, total_records=total_records)

    bucket_lists, exceptions = _make_buckets(records, cfg)
    total_buckets = len(bucket_lists)

    tasks = [bucket_lists[i:i + BUCKETS_PER_TASK] for i in range(0, len(bucket_lists), BUCKETS_PER_TASK)]

    ctx = mp.get_context("fork")  # linux container -> fork is fastest (shared memory)
    dsu = DSU(total_records)

    pairs_scored = 0
    matches_found = 0

    exc_topk = int(os.environ.get("MATCH_EXCEPTION_TOPK", "5"))
    possible_cands: Dict[int, Dict[int, float]] = {}


    _init_globals(records, cfg)

with ProcessPoolExecutor(max_workers=match_workers, mp_context=ctx) as ex:
        futures = [ex.submit(_process_bucket_chunk, chunk) for chunk in tasks]
        for fut in as_completed(futures):
            edges, ps, mf, possibles = fut.result()
            pairs_scored += ps
            matches_found += mf
            for a, b in edges:
                dsu.union(a, b)

            if exc_topk > 0 and possibles:
                for a, b, score in possibles:
                    da = possible_cands.get(a)
                    if da is None:
                        da = {}
                        possible_cands[a] = da
                    prev = da.get(b)
                    if prev is None or score > prev:
                        da[b] = score
                        if len(da) > exc_topk:
                            worst_b, _ = min(da.items(), key=lambda kv: kv[1])
                            da.pop(worst_b, None)

                    db = possible_cands.get(b)
                    if db is None:
                        db = {}
                        possible_cands[b] = db
                    prev = db.get(a)
                    if prev is None or score > prev:
                        db[a] = score
                        if len(db) > exc_topk:
                            worst_a, _ = min(db.items(), key=lambda kv: kv[1])
                            db.pop(worst_a, None)


    # build clusters (root index -> member indices)
    clusters: Dict[int, List[int]] = {}
    for i in range(total_records):
        root = dsu.find(i)
        clusters.setdefault(root, []).append(i)

    total_clusters = len(clusters)

    # stable cluster ids scoped by (app_user_id, model_id)
    existing_map = _load_cluster_map(app_user_id, model_id)
cluster_ids = _assign_stable_cluster_ids(records, clusters, existing_map)

    # write match_exception (possible matches across clusters)
    with get_conn() as conn:
        conn.execute("DELETE FROM match_exception WHERE job_id=?", (job_id,))

    if exc_topk > 0 and possible_cands:
        idx_to_root: List[int] = [0] * total_records
        for root, members in clusters.items():
            for idx in members:
                idx_to_root[idx] = root

        # dedupe: best score per (record_id, candidate_cluster_id)
        best: Dict[Tuple[str, str], Tuple[float, str]] = {}
        for i, cmap in possible_cands.items():
            root_i = idx_to_root[i]
            cluster_i = cluster_ids[root_i]
            rec_i = records[i]
            for j, score in cmap.items():
                root_j = idx_to_root[j]
                cluster_j = cluster_ids[root_j]
                if cluster_j == cluster_i:
                    continue
                key = (rec_i.record_id, cluster_j)
                prev = best.get(key)
                cand_rec_id = records[j].record_id
                if prev is None or score > prev[0]:
                    best[key] = (score, cand_rec_id)

        if best:
            now = _utc_now_iso()
            rows = []
            for (record_id, cand_cluster_id), (score, cand_rec_id) in best.items():
                # record_id is unique per job: source_name::source_id
                if "::" in record_id:
                    source_name, source_id = record_id.split("::", 1)
                else:
                    source_name, source_id = "", record_id

                rows.append(
                    (
                        job_id,
                        model_id,
                        record_id,
                        source_name,
                        source_id,
                        cand_cluster_id,
                        cand_rec_id,
                        float(score),
                        "possible_match",
                        None,
                        now,
                        None,
                        None,
                        app_user_id,
                    )
                )

            with get_conn() as conn:
                conn.executemany(
                    """
                    INSERT INTO match_exception (
                      job_id, model_id,
                      record_id, source_name, source_id,
                      candidate_cluster_id, candidate_record_id, score,
                      reason, details_json,
                      created_at, resolved_at, resolved_by,
                      app_user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )


    # write recon_cluster + cluster_map
    _write_outputs(job_id, app_user_id, model_id, records, clusters, cluster_ids)

    # survivorship + golden + representative flags
    _write_survivorship(job_id, app_user_id, model_id, model)

    _set_job_status(
        job_id,
        "completed",
        None,
        total_buckets=total_buckets,
        total_pairs_scored=pairs_scored,
        total_matches=matches_found,
        total_clusters=total_clusters,
        exceptions_json=json.dumps(exceptions[:50]),
    )


def _job_process_entry(job_id: str, match_workers: int) -> None:
    try:
        run_one_job(job_id, match_workers=match_workers)
    except Exception as e:
        try:
            _set_job_status(job_id, "failed", str(e))
        except Exception:
            pass


def main() -> None:
    print("Worker started")
    init_all_tables()

    # split CPU budget across concurrent jobs
    job_slots = max(1, JOB_WORKERS)
    per_job_workers = max(1, MATCH_WORKERS_TOTAL // job_slots)

    ctx = mp.get_context("fork")
    active: Dict[str, mp.Process] = {}

    while True:
        # reap finished jobs
        for jid, p in list(active.items()):
            if not p.is_alive():
                p.join(timeout=0.1)
                active.pop(jid, None)

        # fill slots
        slots = job_slots - len(active)
        if slots > 0:
            claimed = _claim_queued_jobs(slots)
            for j in claimed:
                jid = j["job_id"]
                p = ctx.Process(
                    target=_job_process_entry,
                    args=(jid, per_job_workers),
                    daemon=False,   # must be False, job spawns child processes
                )
                p.start()
                active[jid] = p

        if not active:
            time.sleep(POLL_SECONDS)
        else:
            time.sleep(0.2)


if __name__ == "__main__":
    main()
