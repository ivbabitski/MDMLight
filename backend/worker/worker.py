import os
import json
import time
import uuid
import sqlite3
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from rapidfuzz.distance import Levenshtein
from db.sqlite_db import get_conn
from worker.survivorship import select_golden_records_for_job

POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "1.0"))
MAX_BUCKET_SIZE = int(os.environ.get("MATCH_MAX_BUCKET_SIZE", "500"))
BUCKETS_PER_TASK = int(os.environ.get("MATCH_BUCKETS_PER_TASK", "50"))
MAX_WORKERS = int(os.environ.get("MATCH_WORKERS", str(os.cpu_count() or 2)))

# Globals shared via fork
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

def _field_index(fname: str) -> int:
    # "f01" -> 0
    if not isinstance(fname, str) or len(fname) != 3 or fname[0].lower() != "f":
        raise ValueError(f"invalid field name: {fname}")
    n = int(fname[1:])
    if not (1 <= n <= 20):
        raise ValueError(f"field out of range: {fname}")
    return n - 1

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
    record_id: str
    source_name: str
    source_id: str
    raw20: List[Any]          # length 20
    norm_selected: List[str]  # length = selected_fields

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
            # missing => ignore this field (MVP behavior)
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

def _process_bucket_chunk(bucket_lists: List[List[int]]) -> Tuple[List[Tuple[int, int]], int, int]:
    """
    Returns (match_edges, pairs_scored, matches_found)
    """
    edges: List[Tuple[int, int]] = []
    pairs_scored = 0
    matches_found = 0

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
                ok, _score = _pair_score(a, b)
                if ok:
                    edges.append(key)
                    matches_found += 1

    return edges, pairs_scored, matches_found

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

def _pick_job() -> Tuple[str, Dict[str, Any]] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT job_id, model_json FROM match_job WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return row["job_id"], json.loads(row["model_json"])

def _set_job_status(job_id: str, status: str, message: str | None = None, **metrics) -> None:
    now = _utc_now_iso()
    cols = ["status=?", "updated_at=?"]
    vals = [status, now]
    if message is not None:
        cols.append("message=?")
        vals.append(message)

    for k, v in metrics.items():
        cols.append(f"{k}=?")
        vals.append(v)

    if status == "running":
        cols.append("started_at=?")
        vals.append(now)
    if status in ("completed", "failed"):
        cols.append("finished_at=?")
        vals.append(now)

    vals.append(job_id)

    with get_conn() as conn:
        conn.execute(f"UPDATE match_job SET {', '.join(cols)} WHERE job_id=?", vals)

def _load_source_input() -> List[Tuple]:
    cols = ["id", "source_name", "source_id"] + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
    sql = f"SELECT {', '.join(cols)} FROM source_input"
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
        return [tuple(r) for r in rows]

def _load_cluster_map() -> Dict[Tuple[str, str], str]:
    m = {}
    with get_conn() as conn:
        try:
            rows = conn.execute("SELECT source_name, source_id, cluster_id FROM cluster_map").fetchall()
        except sqlite3.OperationalError:
            return m
        for r in rows:
            m[(r["source_name"], r["source_id"])] = r["cluster_id"]
    return m

def _write_outputs(
    job_id: str,
    records: List[Record],
    clusters: Dict[int, List[int]],
    cluster_ids: Dict[int, str],
    representatives: Dict[int, int],
    survivorship_method: str = "placeholder_most_complete",
) -> None:
    now = _utc_now_iso()

    # Prepare inserts
    recon_rows = []
    golden_rows = []
    map_rows = []

    for root, members in clusters.items():
        cid = cluster_ids[root]
        rep_idx = representatives[root]
        cluster_size = len(members)

        # recon rows
        for idx in members:
            r = records[idx]
            recon_rows.append((
                job_id, cid,
                r.record_id, r.source_name, r.source_id,
                cluster_size,
                1 if idx == rep_idx else 0,
                now
            ))

            map_rows.append((r.source_name, r.source_id, cid, now, now))

        # golden record row (copy representative)
        rep = records[rep_idx]
        golden_rows.append((
            job_id, cid,
            rep.record_id,
            survivorship_method,
            now,
            *rep.raw20
        ))

    with get_conn() as conn:
        # Clear old outputs for this job_id (if rerun)
        conn.execute("DELETE FROM recon_cluster WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM golden_record WHERE job_id=?", (job_id,))

        # Write recon_cluster
        conn.executemany(
            """
            INSERT INTO recon_cluster (
              job_id, cluster_id,
              record_id, source_name, source_id,
              cluster_size, is_representative, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            recon_rows
        )

        # Write golden_record
        conn.executemany(
            """
            INSERT INTO golden_record (
              job_id, cluster_id,
              representative_record_id,
              survivorship_method, created_at,
              f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,
              f11,f12,f13,f14,f15,f16,f17,f18,f19,f20
            ) VALUES (?, ?, ?, ?, ?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            golden_rows
        )

        # Upsert cluster_map
        conn.executemany(
            """
            INSERT INTO cluster_map (source_name, source_id, cluster_id, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_name, source_id) DO UPDATE SET
              cluster_id = excluded.cluster_id,
              last_seen_at = excluded.last_seen_at
            """,
            map_rows
        )

def _choose_representatives(records: List[Record], clusters: Dict[int, List[int]]) -> Dict[int, int]:
    reps = {}
    for root, members in clusters.items():
        best = None
        best_key = None
        for idx in members:
            r = records[idx]
            completeness = sum(1 for v in r.raw20 if v is not None and str(v).strip() != "")
            # stable tie-breakers
            key = (-completeness, r.source_name, r.source_id, r.record_id)
            if best_key is None or key < best_key:
                best_key = key
                best = idx
        reps[root] = best
    return reps

def _build_cfg(model: Dict[str, Any]) -> Dict[str, Any]:
    selected_fields = model.get("selected_fields", [])
    weights_map = model.get("weights", {}) or {}
    thresh_map = model.get("field_thresholds", {}) or {}
    model_T = _to_0_1(model.get("model_threshold"), 0.85)

    sel = []
    weights = []
    thresholds = []

    for f in selected_fields:
        sel.append(f)
        weights.append(float(weights_map.get(f, 1.0)))
        thresholds.append(_to_0_1(thresh_map.get(f, 0.0), 0.0))

    # Blocking
    blocking = model.get("blocking", {}) or {}
    max_bucket = int(blocking.get("max_bucket_size", MAX_BUCKET_SIZE))
    passes = blocking.get("blocking_passes", "composite+single")  # composite_only | composite+single
    block_fields = blocking.get("blocking_fields")

    # pick default blocking fields: top-2 by weight
    if not isinstance(block_fields, list) or len(block_fields) == 0:
        pairs = list(zip(selected_fields, weights))
        pairs.sort(key=lambda x: x[1], reverse=True)
        block_fields = [p[0] for p in pairs[:2]] if pairs else []

    return {
        "selected_fields": sel,
        "selected_indices": [_field_index(f) for f in sel],
        "weights": weights,
        "thresholds": thresholds,
        "model_T": model_T,
        "max_bucket_size": max_bucket,
        "blocking_fields": block_fields[:2],
        "blocking_passes": passes,
    }

def _build_records(raw_rows: List[Tuple], cfg: Dict[str, Any]) -> List[Record]:
    sel_idx = cfg["selected_indices"]
    records: List[Record] = []
    for row in raw_rows:
        record_id = row[0]
        source_name = row[1]
        source_id = row[2]
        raw20 = list(row[3:23])  # 20 fields
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
        # degenerate: no fields selected
        return [], [{"reason": "no_block_fields"}]

    # Map field -> selected_fields position (if present), else use raw20 index
    # We'll block on raw20 directly (more robust; selected_fields might be subset)
    block_idx = [_field_index(f) for f in block_fields]

    # Prefix lengths derived from per-field thresholds if the field is selected; else default 3
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

        # Composite key (most selective)
        if len(pfx) >= 2 and pfx[0] and pfx[1]:
            k = f"{block_fields[0]}:{pfx[0]}|{block_fields[1]}:{pfx[1]}"
            buckets.setdefault(k, []).append(i)

        if passes == "composite+single":
            # Single keys (fallback)
            for f, p in zip(block_fields, pfx):
                if p:
                    k = f"{f}:{p}"
                    buckets.setdefault(k, []).append(i)

    bucket_lists = []
    exceptions = []

    for k, ids in buckets.items():
        if len(ids) < 2:
            continue
        if len(ids) > max_bucket:
            exceptions.append({"reason": "hot_bucket_skipped", "key": k, "size": len(ids)})
            continue
        bucket_lists.append(ids)

    return bucket_lists, exceptions

def _cluster_with_stable_ids(
    records: List[Record],
    clusters: Dict[int, List[int]],
    existing_map: Dict[Tuple[str, str], str],
) -> Dict[int, str]:
    cluster_ids: Dict[int, str] = {}
    for root, members in clusters.items():
        seen = set()
        for idx in members:
            r = records[idx]
            cid = existing_map.get((r.source_name, r.source_id))
            if cid:
                seen.add(cid)

        if seen:
            cluster_ids[root] = sorted(seen)[0]  # canonical
        else:
            cluster_ids[root] = str(uuid.uuid4())

    return cluster_ids

def run_one_job(job_id: str, model: Dict[str, Any]) -> None:
    cfg = _build_cfg(model)

    raw_rows = _load_source_input()
    if not raw_rows:
        _set_job_status(job_id, "failed", "source_input is empty", total_records=0)
        return

    records = _build_records(raw_rows, cfg)
    total_records = len(records)
    _set_job_status(job_id, "running", None, total_records=total_records)

    bucket_lists, exceptions = _make_buckets(records, cfg)
    total_buckets = len(bucket_lists)

    # parallel scoring by bucket chunks
    tasks = [bucket_lists[i:i + BUCKETS_PER_TASK] for i in range(0, len(bucket_lists), BUCKETS_PER_TASK)]

    ctx = mp.get_context("fork")  # linux container => fork, shares memory
    dsu = DSU(total_records)

    pairs_scored = 0
    matches_found = 0

    _init_globals(records, cfg)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        futures = [ex.submit(_process_bucket_chunk, chunk) for chunk in tasks]
        for fut in as_completed(futures):
            edges, ps, mf = fut.result()
            pairs_scored += ps
            matches_found += mf
            for a, b in edges:
                dsu.union(a, b)

    # build clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(total_records):
        root = dsu.find(i)
        clusters.setdefault(root, []).append(i)

    total_clusters = len(clusters)

    # stable cluster ids
    existing_map = _load_cluster_map()
    cluster_ids = _cluster_with_stable_ids(records, clusters, existing_map)

    # survivorship placeholder
    representatives = _choose_representatives(records, clusters)

    # write outputs
    _write_outputs(job_id, records, clusters, cluster_ids, representatives)

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

def main() -> None:
    print("Worker started")
    while True:
        picked = _pick_job()
        if not picked:
            time.sleep(POLL_SECONDS)
            continue

        job_id, model = picked
        try:
            _set_job_status(job_id, "running", None)
            run_one_job(job_id, model)
        except Exception as e:
            _set_job_status(job_id, "failed", str(e))

if __name__ == "__main__":
    main()
