import os
import json
import time
import traceback
import uuid
import sqlite3
import multiprocessing as mp
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

from worker.levenshtein import weighted_gated_score
from worker.survivorship import run_survivorship

from db.sqlite_db import get_conn, init_all_tables


POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "1.0"))

# Job-level concurrency (processes). Keep low unless you know what you're doing.
JOB_WORKERS = int(os.environ.get("WORKER_JOB_WORKERS", "1"))

# Matching tuning
MAX_BUCKET_SIZE = int(os.environ.get("MATCH_MAX_BUCKET_SIZE", "500"))

# Total CPU workers budget (split across concurrent jobs)
MATCH_WORKERS_TOTAL = int(os.environ.get("MATCH_WORKERS", str(os.cpu_count() or 2)))

# Candidate generation (blocking)
CANDIDATE_MIN = 250
CANDIDATE_MAX = 500

# Backward-compatible env var (no longer used for candidate selection range).
MATCH_CANDIDATE_TARGET = int(os.environ.get("MATCH_CANDIDATE_TARGET", "500"))
MATCH_BLOCK_MAX_PREFIX_LEN = int(os.environ.get("MATCH_BLOCK_MAX_PREFIX_LEN", "10"))



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: Any) -> str:
    if s is None:
        return ""
    v = str(s).strip().lower()

    # Strip punctuation by converting non-alnum chars to spaces,
    # then collapse whitespace.
    cleaned = []
    for ch in v:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    v = "".join(cleaned)
    v = " ".join(v.split())
    return v


def _esc_pipe(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("|", "||")

def _make_record_id(app_user_id: int, model_id: str, source_name: Any, source_id: Any) -> str:
    return f"{app_user_id}|{model_id}|{_esc_pipe(source_name)}|{_esc_pipe(source_id)}"



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


def _load_model_name_from_mdm_models(model_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT model_name FROM mdm_models WHERE id=? AND deleted_at IS NULL LIMIT 1",
            (model_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"mdm_models not found (or deleted): id={model_id}")
        return str(row["model_name"])


def _load_model_config_from_mdm_models(model_id: str) -> Dict[str, Any]:
    """
    Loads the model configuration JSON from mdm_models.

    The worker expects to receive a model config in match_job.model_json.
    If that snapshot is empty, we load the current model config from mdm_models.

    NOTE: This function is intentionally strict: it only reads from known
    JSON-bearing columns and raises a clear error if none are present.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM mdm_models WHERE id=? AND deleted_at IS NULL LIMIT 1",
            (model_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"mdm_models not found (or deleted): id={model_id}")

        candidate_cols = (
            "model_json",
            "model_config_json",
            "config_json",
            "model_config",
            "config",
        )

        for col in candidate_cols:
            if col not in row.keys():
                continue

            raw = row[col]
            if raw is None:
                raise ValueError(f"mdm_models.{col} is NULL for id={model_id}")

            if isinstance(raw, dict):
                return raw

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")

            if isinstance(raw, str):
                s = raw.strip()
                if not s:
                    raise ValueError(f"mdm_models.{col} is empty for id={model_id}")
                try:
                    parsed = json.loads(s)
                except Exception as e:
                    raise ValueError(f"mdm_models.{col} is not valid JSON for id={model_id}: {e}")
                if not isinstance(parsed, dict):
                    raise ValueError(f"mdm_models.{col} must be a JSON object for id={model_id}")
                return parsed

            raise ValueError(f"mdm_models.{col} has unsupported type {type(raw)} for id={model_id}")

        raise ValueError(
            "no model config column found in mdm_models for this worker; "
            f"looked for {list(candidate_cols)}; available columns={list(row.keys())}"
        )


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

    total_w = float(sum(weights))
    if total_w > 0.0:
        weights = [float(w) / total_w for w in weights]

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


def _load_recon_cluster_rows(app_user_id: int, model_id: str) -> List[Tuple]:
    cols = ["source_id", "source_name"] + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
    sql = f"SELECT {', '.join(cols)} FROM recon_cluster WHERE app_user_id=? AND model_id=?"
    with get_conn() as conn:
        rows = conn.execute(sql, (app_user_id, model_id)).fetchall()
        return [tuple(r) for r in rows]


def _load_unprocessed_source_input_rows(app_user_id: int, model_id: str) -> List[sqlite3.Row]:
    cols = (
        ["source_name", "source_id"]
        + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
        + ["created_at", "created_by", "updated_at", "updated_by"]
    )

    sql = f"""
      SELECT {", ".join([f"s.{c}" for c in cols])}
      FROM source_input s
      WHERE s.app_user_id=?
        AND NOT EXISTS (
          SELECT 1
          FROM recon_cluster r
          WHERE r.app_user_id = s.app_user_id
            AND r.model_id = ?
            AND r.source_name = s.source_name
            AND r.source_id = s.source_id
        )
      ORDER BY s.source_name, s.source_id
    """

    with get_conn() as conn:
        return conn.execute(sql, (app_user_id, model_id)).fetchall()


def _candidate_query(
    conn: sqlite3.Connection,
    app_user_id: int,
    model_id: str,
    cfg: Dict[str, Any],
    f1: str,
    p1: str,
    l1: int,
    f2: Optional[str] = None,
    p2: str = "",
    l2: int = 0,
) -> List[sqlite3.Row]:
    if not _is_f_code(f1):
        return []
    if f2 is not None and not _is_f_code(f2):
        return []

    cols = ["cluster_id", "source_name", "source_id"] + list(cfg.get("selected_fields") or [])
    where = [
        "app_user_id=?",
        "model_id=?",
        f"substr(norm(coalesce({f1}, '')), 1, ?) = ?",
    ]
    params: List[Any] = [app_user_id, model_id, int(l1), str(p1)]

    if f2 is not None:
        where.append(f"substr(norm(coalesce({f2}, '')), 1, ?) = ?")
        params.extend([int(l2), str(p2)])

    params.append(int(CANDIDATE_MAX) + 1)

    sql = (
        f"SELECT {', '.join(cols)} "
        f"FROM recon_cluster "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY cluster_id ASC, source_name ASC, source_id ASC "
        f"LIMIT ?"
    )
    return conn.execute(sql, tuple(params)).fetchall()



def _candidate_scan(
    conn: sqlite3.Connection,
    app_user_id: int,
    model_id: str,
    cfg: Dict[str, Any],
    limit: int,
) -> List[sqlite3.Row]:
    cols = ["cluster_id", "source_name", "source_id"] + list(cfg.get("selected_fields") or [])
    sql = f"""
      SELECT {", ".join(cols)}
      FROM recon_cluster
      WHERE app_user_id=?
        AND model_id=?
      ORDER BY cluster_id ASC, source_name ASC, source_id ASC
      LIMIT ?
    """
    return conn.execute(sql, (app_user_id, model_id, int(limit))).fetchall()



def _adaptive_candidates(
    conn: sqlite3.Connection,
    app_user_id: int,
    model_id: str,
    cfg: Dict[str, Any],
    norm_by_field: Dict[str, str],
    scope_total: Optional[int] = None,
) -> List[sqlite3.Row]:
    # Candidate selection requirements:
    # - Start with highest-weight field.
    # - Increment prefixes proportionally to weights via priority = weight / (prefix_len + 1).
    # - Stop once candidate count is in [CANDIDATE_MIN, CANDIDATE_MAX] when possible.
    # - Never return empty due to over-tightening; back off to the last non-empty set.
    selected_fields: List[str] = list(cfg.get("selected_fields") or [])
    weights: List[float] = list(cfg.get("weights") or [])

    if scope_total is not None and int(scope_total) <= int(CANDIDATE_MAX):
        return _candidate_scan(conn, app_user_id, model_id, cfg, limit=CANDIDATE_MAX)

    pairs = list(zip(selected_fields, weights))
    pairs.sort(key=lambda x: float(x[1] or 0.0), reverse=True)

    usable: List[Tuple[str, float, str]] = []
    for f, w in pairs:
        v = norm_by_field.get(f, "")
        if not v:
            continue
        usable.append((f, float(w or 0.0), v))
        if len(usable) >= 2:
            break

    if not usable:
        return _candidate_scan(conn, app_user_id, model_id, cfg, limit=CANDIDATE_MAX)

    f1, w1, v1 = usable[0]
    max1 = min(len(v1), int(MATCH_BLOCK_MAX_PREFIX_LEN))

    f2: Optional[str] = None
    w2 = 0.0
    v2 = ""
    max2 = 0

    if len(usable) >= 2:
        f2, w2, v2 = usable[1]
        max2 = min(len(v2), int(MATCH_BLOCK_MAX_PREFIX_LEN))

    l1 = 0
    l2 = 0

    last_non_empty: Optional[List[sqlite3.Row]] = None
    last_over_max: Optional[List[sqlite3.Row]] = None

    while True:
        best_priority = -1.0
        best_weight = -1.0
        best_order = 10**9

        if l1 < max1:
            pri1 = float(w1) / float(l1 + 1)
            best_priority = pri1
            best_weight = float(w1)
            best_order = 0

        if f2 is not None and l2 < max2:
            pri2 = float(w2) / float(l2 + 1)
            if pri2 > best_priority or (pri2 == best_priority and (float(w2) > best_weight or (float(w2) == best_weight and 1 < best_order))):
                best_priority = pri2
                best_weight = float(w2)
                best_order = 1

        if best_order == 10**9:
            break

        if best_order == 0:
            l1 += 1
        else:
            l2 += 1

        if f2 is not None and l2 > 0:
            rows = _candidate_query(
                conn,
                app_user_id,
                model_id,
                cfg,
                f1=f1,
                p1=v1[:l1],
                l1=l1,
                f2=f2,
                p2=v2[:l2],
                l2=l2,
            )
        else:
            rows = _candidate_query(conn, app_user_id, model_id, cfg, f1=f1, p1=v1[:l1], l1=l1)

        if len(rows) == 0:
            if last_non_empty is not None:
                if len(last_non_empty) > int(CANDIDATE_MAX):
                    return last_non_empty[: int(CANDIDATE_MAX)]
                return last_non_empty
            return []

        last_non_empty = rows

        if len(rows) > int(CANDIDATE_MAX):
            last_over_max = rows
            continue

        if len(rows) >= int(CANDIDATE_MIN):
            return rows

        if last_over_max is not None:
            return last_over_max[: int(CANDIDATE_MAX)]

        return rows

    if last_over_max is not None:
        return last_over_max[: int(CANDIDATE_MAX)]
    if last_non_empty is not None:
        if len(last_non_empty) > int(CANDIDATE_MAX):
            return last_non_empty[: int(CANDIDATE_MAX)]
        return last_non_empty
    return []



class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

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


def _build_prefix_index_for_batch(
    cfg: Dict[str, Any],
    norm_rows: List[Dict[str, str]],
) -> Dict[str, Dict[int, Dict[str, List[int]]]]:
    selected_fields: List[str] = list(cfg.get("selected_fields") or [])
    index: Dict[str, Dict[int, Dict[str, List[int]]]] = {f: {} for f in selected_fields}
    max_pref = int(MATCH_BLOCK_MAX_PREFIX_LEN)

    for i, row in enumerate(norm_rows):
        for f in selected_fields:
            v = row.get(f, "")
            if not v:
                continue
            max_l = min(len(v), max_pref)
            by_len = index.get(f)
            if by_len is None:
                by_len = {}
                index[f] = by_len
            for l in range(1, max_l + 1):
                p = v[:l]
                by_prefix = by_len.get(l)
                if by_prefix is None:
                    by_prefix = {}
                    by_len[l] = by_prefix
                lst = by_prefix.get(p)
                if lst is None:
                    lst = []
                    by_prefix[p] = lst
                lst.append(i)

    return index


def _adaptive_candidates_batch(
    record_idx: int,
    cfg: Dict[str, Any],
    norm_rows: List[Dict[str, str]],
    prefix_index: Dict[str, Dict[int, Dict[str, List[int]]]],
) -> List[int]:
    selected_fields: List[str] = list(cfg.get("selected_fields") or [])
    weights: List[float] = list(cfg.get("weights") or [])

    total_other = max(0, len(norm_rows) - 1)
    if total_other <= int(CANDIDATE_MAX):
        return [i for i in range(len(norm_rows)) if i != record_idx]

    pairs = list(zip(selected_fields, weights))
    pairs.sort(key=lambda x: float(x[1] or 0.0), reverse=True)

    norm_by_field = norm_rows[record_idx]

    usable: List[Tuple[str, float, str]] = []
    for f, w in pairs:
        v = norm_by_field.get(f, "")
        if not v:
            continue
        usable.append((f, float(w or 0.0), v))
        if len(usable) >= 2:
            break

    if not usable:
        out: List[int] = [i for i in range(len(norm_rows)) if i != record_idx]
        return out[: int(CANDIDATE_MAX)]

    def _list_for(f: str, l: int, p: str) -> List[int]:
        by_len = prefix_index.get(f) or {}
        by_prefix = by_len.get(int(l)) or {}
        return list(by_prefix.get(str(p)) or [])

    f1, w1, v1 = usable[0]
    max1 = min(len(v1), int(MATCH_BLOCK_MAX_PREFIX_LEN))

    f2: Optional[str] = None
    w2 = 0.0
    v2 = ""
    max2 = 0

    if len(usable) >= 2:
        f2, w2, v2 = usable[1]
        max2 = min(len(v2), int(MATCH_BLOCK_MAX_PREFIX_LEN))

    l1 = 0
    l2 = 0

    last_non_empty: Optional[List[int]] = None
    last_over_max: Optional[List[int]] = None

    while True:
        best_priority = -1.0
        best_weight = -1.0
        best_order = 10**9

        if l1 < max1:
            pri1 = float(w1) / float(l1 + 1)
            best_priority = pri1
            best_weight = float(w1)
            best_order = 0

        if f2 is not None and l2 < max2:
            pri2 = float(w2) / float(l2 + 1)
            if pri2 > best_priority or (pri2 == best_priority and (float(w2) > best_weight or (float(w2) == best_weight and 1 < best_order))):
                best_priority = pri2
                best_weight = float(w2)
                best_order = 1

        if best_order == 10**9:
            break

        if best_order == 0:
            l1 += 1
        else:
            l2 += 1

        if l1 <= 0:
            cand: List[int] = []
        elif f2 is not None and l2 > 0:
            l1_list = _list_for(f1, l1, v1[:l1])
            l2_list = _list_for(f2, l2, v2[:l2])

            if not l1_list or not l2_list:
                cand = []
            elif len(l1_list) <= len(l2_list):
                small = set(l1_list)
                cand = [x for x in l2_list if x in small]
            else:
                small = set(l2_list)
                cand = [x for x in l1_list if x in small]
        else:
            cand = _list_for(f1, l1, v1[:l1])

        cand = [x for x in cand if x != record_idx]
        cand.sort()

        if len(cand) > int(CANDIDATE_MAX) + 1:
            cand = cand[: int(CANDIDATE_MAX) + 1]

        if len(cand) == 0:
            if last_non_empty is not None:
                if len(last_non_empty) > int(CANDIDATE_MAX):
                    return last_non_empty[: int(CANDIDATE_MAX)]
                return last_non_empty
            return []

        last_non_empty = cand

        if len(cand) > int(CANDIDATE_MAX):
            last_over_max = cand
            continue

        if len(cand) >= int(CANDIDATE_MIN):
            return cand

        if last_over_max is not None:
            return last_over_max[: int(CANDIDATE_MAX)]

        return cand

    if last_over_max is not None:
        return last_over_max[: int(CANDIDATE_MAX)]
    if last_non_empty is not None:
        if len(last_non_empty) > int(CANDIDATE_MAX):
            return last_non_empty[: int(CANDIDATE_MAX)]
        return last_non_empty
    return []


def _seed_recon_cluster_from_source_input(app_user_id: int, model_id: str, model_name: str) -> int:
    src_cols = (
        ["source_name", "source_id"]
        + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
        + ["created_at", "created_by", "updated_at", "updated_by"]
    )

    now = _utc_now_iso()

    with get_conn() as conn:
        # Compatibility: some DBs have recon_cluster.cluster_size defined as NOT NULL.
        # If present, we must supply it on INSERT.
        recon_cols = conn.execute("PRAGMA table_info(recon_cluster)").fetchall()
        recon_has_cluster_size = any(str(r["name"]) == "cluster_size" for r in recon_cols)

        insert_cols = (
            ["cluster_id", "model_id", "model_name", "source_name", "source_id", "app_user_id"]
            + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
            + ["created_at", "created_by", "updated_at", "updated_by", "match_status"]
        )

        if recon_has_cluster_size:
            insert_cols = list(insert_cols)
            insert_cols.insert(1, "cluster_size")

        placeholders = ", ".join(["?"] * len(insert_cols))

        rows = conn.execute(
            f"SELECT {', '.join(src_cols)} FROM source_input WHERE app_user_id=?",
            (app_user_id,),
        ).fetchall()

        if not rows:
            return 0

        to_insert: List[Tuple[Any, ...]] = []
        map_rows: List[Tuple[Any, ...]] = []

        for r in rows:
            cid = str(uuid.uuid4())

            vals: List[Any] = [
                cid,
                model_id,
                model_name,
                r["source_name"],
                r["source_id"],
                app_user_id,
            ]

            for i in range(1, 21):
                vals.append(r[f"f{str(i).zfill(2)}"])

            vals.append(r["created_at"])
            vals.append(r["created_by"])
            vals.append(r["updated_at"])
            vals.append(r["updated_by"])
            vals.append("no_match")

            if recon_has_cluster_size:
                vals.insert(1, 1)

            to_insert.append(tuple(vals))

            map_rows.append(
                (
                    app_user_id,
                    model_id,
                    r["source_name"],
                    r["source_id"],
                    cid,
                    now,
                    now,
                )
            )

        conn.executemany(
            f"INSERT INTO recon_cluster ({', '.join(insert_cols)}) VALUES ({placeholders})",
            to_insert,
        )

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
            map_rows,
        )

        return len(to_insert)



def _step1_recon_cluster_sync(app_user_id: int, model_id: str) -> Tuple[int, int]:
    """
    Ensures recon_cluster is not empty for this (app_user_id, model_id).

    - If recon_cluster already has rows: do nothing.
    - If empty: bootstrap recon_cluster from source_input (each record becomes its own cluster).
    """
    with get_conn() as conn:
        has_any = (
            conn.execute(
                "SELECT 1 FROM recon_cluster WHERE app_user_id=? AND model_id=? LIMIT 1",
                (app_user_id, model_id),
            ).fetchone()
            is not None
        )

    inserted = 0
    if not has_any:
        model_name = _load_model_name_from_mdm_models(model_id)
        inserted = _seed_recon_cluster_from_source_input(app_user_id, model_id, model_name)

    rows = _load_recon_cluster_rows(app_user_id, model_id)
    return len(rows), inserted

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

    _set_job_status(job_id, "running", None)

    # Model config: prefer job snapshot; else load from mdm_models
    try:
        raw_model = json.loads(model_json_raw) if model_json_raw else None
    except Exception:
        raw_model = None

    if not raw_model:
        try:
            raw_model = _load_model_config_from_mdm_models(model_id)
        except Exception as e:
            _set_job_status(job_id, "failed", f"failed_to_load_model_config: {e}")
            return

    surv_model = raw_model.get("config") if isinstance(raw_model.get("config"), dict) else raw_model

    model = _normalize_model_json(surv_model)
    if not model:
        _set_job_status(job_id, "failed", "model_json/config is empty or invalid")
        return

    cfg = _build_cfg(model)
    if not cfg.get("selected_fields"):
        _set_job_status(job_id, "failed", "no match fields with positive weight", total_records=0)
        return

    # STEP 1: determine operating mode for this (app_user_id, model_id)
    with get_conn() as conn:
        has_recon = (
            conn.execute(
                "SELECT 1 FROM recon_cluster WHERE app_user_id=? AND model_id=? LIMIT 1",
                (app_user_id, model_id),
            ).fetchone()
            is not None
        )

    # STEP 2: load NEW source_input rows (incremental) OR full batch (bootstrap when recon is empty)
    src_rows = _load_unprocessed_source_input_rows(app_user_id, model_id)
    if not src_rows:
        try:
            run_survivorship(job_id=job_id, app_user_id=int(app_user_id), model=surv_model, actor="match_worker")
        except Exception as e:
            _set_job_status(
                job_id,
                "failed",
                f"survivorship_failed: {e}",
                total_records=0,
                total_matches=0,
                total_clusters=0,
                total_pairs_scored=0,
            )
            return

        _set_job_status(
            job_id,
            "completed",
            "no_new_source_input_records",
            total_records=0,
            total_matches=0,
            total_clusters=0,
            total_pairs_scored=0,
        )
        return


    model_name = _load_model_name_from_mdm_models(model_id)
    now = _utc_now_iso()

    # Compatibility: some DBs have recon_cluster.cluster_size defined as NOT NULL.
    # If present, we must supply it on INSERT.
    with get_conn() as conn:
        recon_cols_info = conn.execute("PRAGMA table_info(recon_cluster)").fetchall()
        recon_has_cluster_size = any(str(r["name"]) == "cluster_size" for r in recon_cols_info)

    recon_cols = (
        ["cluster_id", "job_id", "record_id", "model_id", "model_name", "source_name", "source_id", "app_user_id"]
        + [f"f{str(i).zfill(2)}" for i in range(1, 21)]
        + ["created_at", "created_by", "updated_at", "updated_by", "match_status", "match_score"]
    )

    if recon_has_cluster_size:
        recon_cols.insert(1, "cluster_size")
    recon_placeholders = ", ".join(["?"] * len(recon_cols))

    recon_inserts: List[Tuple[Any, ...]] = []
    map_rows: List[Tuple[Any, ...]] = []
    exc_rows: List[Tuple[Any, ...]] = []


    match_count = 0
    exception_count = 0
    new_cluster_count = 0
    total_pairs_scored = 0

    with get_conn() as conn:
        # rerun safety
        conn.execute("DELETE FROM match_exception WHERE job_id=?", (job_id,))

        if not has_recon:
            # Mode A: Initial run (bootstrap) — cluster the batch against itself
            norm_rows: List[Dict[str, str]] = []
            selected_values: List[List[str]] = []
            for r in src_rows:
                nb: Dict[str, str] = {}
                for f in cfg["selected_fields"]:
                    nb[f] = _norm(r[f])
                norm_rows.append(nb)
                selected_values.append([nb[f] for f in cfg["selected_fields"]])

            prefix_index = _build_prefix_index_for_batch(cfg, norm_rows)
            uf = _UnionFind(len(src_rows))

            best_strong_score: List[float] = [0.0] * len(src_rows)

            # A1) Build strong-match links (score >= matchThreshold)
            for i in range(len(src_rows)):
                cand_idxs = _adaptive_candidates_batch(i, cfg, norm_rows, prefix_index)
                for j in cand_idxs:
                    if j <= i:
                        continue
                    s = float(
                        weighted_gated_score(
                            selected_values[i],
                            selected_values[j],
                            cfg["weights"],
                            cfg["thresholds"],
                        )
                    )
                    total_pairs_scored += 1

                    if s >= float(cfg["model_T"]):
                        uf.union(i, j)
                        if s > best_strong_score[i]:
                            best_strong_score[i] = s
                        if s > best_strong_score[j]:
                            best_strong_score[j] = s

            # Build components
            comps: Dict[int, List[int]] = {}
            for i in range(len(src_rows)):
                root = uf.find(i)
                comps.setdefault(root, []).append(i)

            assigned_cluster: List[Optional[str]] = [None] * len(src_rows)
            match_statuses: List[str] = [""] * len(src_rows)
            match_scores: List[float] = [0.0] * len(src_rows)

            # A2) Assign cluster_id to strong clusters (connected components of strong links)
            for _, members in comps.items():
                if len(members) < 2:
                    continue

                cid = str(uuid.uuid4())
                for i in members:
                    assigned_cluster[i] = cid
                    match_statuses[i] = "match"
                    match_scores[i] = float(best_strong_score[i])
                    match_count += 1

            # A3) “Find home” for remaining records (possibleThreshold)
            for i in range(len(src_rows)):
                if assigned_cluster[i] is not None:
                    continue

                cand_idxs = _adaptive_candidates_batch(i, cfg, norm_rows, prefix_index)

                best_score = 0.0
                best_cluster_id: Optional[str] = None
                best_source_name: Optional[str] = None
                best_source_id: Optional[str] = None

                for j in cand_idxs:
                    cid = assigned_cluster[j]
                    if cid is None:
                        continue

                    s = float(
                        weighted_gated_score(
                            selected_values[i],
                            selected_values[j],
                            cfg["weights"],
                            cfg["thresholds"],
                        )
                    )
                    total_pairs_scored += 1

                    if s > best_score or (s == best_score and (best_cluster_id is None or cid < best_cluster_id)):
                        best_score = s
                        best_cluster_id = cid
                        best_source_name = str(src_rows[j]["source_name"])
                        best_source_id = str(src_rows[j]["source_id"])

                if best_cluster_id and best_score >= float(cfg.get("possible_T", 0.0) or 0.0):
                    assigned_cluster[i] = best_cluster_id
                    match_statuses[i] = "exception"
                    match_scores[i] = float(best_score)
                    exception_count += 1

                    record_id = _make_record_id(app_user_id, model_id, src_rows[i]["source_name"], src_rows[i]["source_id"])
                    cand_record_id = _make_record_id(app_user_id, model_id, best_source_name, best_source_id)

                    exc_rows.append(
                        (
                            job_id,
                            model_id,
                            record_id,
                            src_rows[i]["source_name"],
                            src_rows[i]["source_id"],
                            best_cluster_id,
                            cand_record_id,
                            best_source_name,
                            best_source_id,
                            float(best_score),
                            "exception",
                            None,
                            now,
                            None,
                            None,
                            app_user_id,
                        )
                    )
                else:
                    cid = str(uuid.uuid4())
                    assigned_cluster[i] = cid
                    match_statuses[i] = "no_match"
                    match_scores[i] = 0.0
                    new_cluster_count += 1


            # A4) Persist all records
            for i, r in enumerate(src_rows):
                cid = assigned_cluster[i]
                if cid is None:
                    # Should not be possible; safety fallback
                    cid = str(uuid.uuid4())
                    assigned_cluster[i] = cid
                    match_statuses[i] = "no_match"
                    match_scores[i] = 0.0
                    new_cluster_count += 1

                record_id = _make_record_id(app_user_id, model_id, r["source_name"], r["source_id"])

                vals: List[Any] = [
                    cid,
                    job_id,
                    record_id,
                    model_id,
                    model_name,
                    r["source_name"],
                    r["source_id"],
                    app_user_id,
                ]

                if recon_has_cluster_size:
                    vals.insert(1, 1)

                for k in range(1, 21):
                    vals.append(r[f"f{str(k).zfill(2)}"])

                vals.append(r["created_at"])
                vals.append(r["created_by"])
                vals.append(r["updated_at"])
                vals.append(r["updated_by"])
                vals.append(match_statuses[i])
                vals.append(float(match_scores[i]))

                recon_inserts.append(tuple(vals))

                map_rows.append(
                    (
                        app_user_id,
                        model_id,
                        r["source_name"],
                        r["source_id"],
                        cid,
                        now,
                        now,
                    )
                )


            total_clusters = len(set([c for c in assigned_cluster if c is not None]))

        else:
            # Mode B: Incremental run — match only NEW source_input rows into existing clusters
            total_clusters = 0

            recon_scope_total = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM recon_cluster WHERE app_user_id=? AND model_id=?",
                    (app_user_id, model_id),
                ).fetchone()["n"]
            )

            # Incremental mode must still cluster the incoming batch against itself.
            # Otherwise: identical new records processed in the same run will fragment into
            # separate new clusters because they are not yet present in recon_cluster.
            norm_rows: List[Dict[str, str]] = []
            selected_values: List[List[str]] = []
            for r in src_rows:
                nb: Dict[str, str] = {}
                for f in cfg["selected_fields"]:
                    nb[f] = _norm(r[f])
                norm_rows.append(nb)
                selected_values.append([nb[f] for f in cfg["selected_fields"]])

            prefix_index = _build_prefix_index_for_batch(cfg, norm_rows)
            assigned_cluster_batch: List[Optional[str]] = [None] * len(src_rows)

            for i, r in enumerate(src_rows):
                # normalized inputs for match fields
                norm_by_field = norm_rows[i]

                incoming_selected = selected_values[i]

                cand_rows = _adaptive_candidates(
                    conn,
                    app_user_id,
                    model_id,
                    cfg,
                    norm_by_field,
                    scope_total=recon_scope_total,
                )


                best_score = 0.0
                best_cluster_id: Optional[str] = None
                best_source_name: Optional[str] = None
                best_source_id: Optional[str] = None

                if cand_rows:
                    for c in cand_rows:
                        cand_selected = [_norm(c[f]) for f in cfg["selected_fields"]]
                        s = float(
                            weighted_gated_score(
                                incoming_selected,
                                cand_selected,
                                cfg["weights"],
                                cfg["thresholds"],
                            )
                        )
                        total_pairs_scored += 1

                        cid = str(c["cluster_id"])
                        if s > best_score or (s == best_score and (best_cluster_id is None or cid < best_cluster_id)):
                            best_score = s
                            best_cluster_id = cid
                            best_source_name = str(c["source_name"])
                            best_source_id = str(c["source_id"])

                # Also consider already-processed records in the same incoming batch.
                # This prevents within-run fragmentation in incremental mode.
                cand_idxs_batch = _adaptive_candidates_batch(i, cfg, norm_rows, prefix_index)
                if cand_idxs_batch:
                    for j in cand_idxs_batch:
                        cid = assigned_cluster_batch[j]
                        if cid is None:
                            continue

                        s = float(
                            weighted_gated_score(
                                incoming_selected,
                                selected_values[j],
                                cfg["weights"],
                                cfg["thresholds"],
                            )
                        )
                        total_pairs_scored += 1

                        if s > best_score or (s == best_score and (best_cluster_id is None or cid < best_cluster_id)):
                            best_score = s
                            best_cluster_id = cid
                            best_source_name = str(src_rows[j]["source_name"])
                            best_source_id = str(src_rows[j]["source_id"])

                # classification + cluster_id assignment
                if best_cluster_id and best_score >= float(cfg["model_T"]):
                    match_status = "match"
                    cluster_id = best_cluster_id
                    match_score = float(best_score)
                    match_count += 1
                elif best_cluster_id and best_score >= float(cfg.get("possible_T", 0.0) or 0.0):
                    match_status = "exception"
                    cluster_id = best_cluster_id
                    match_score = float(best_score)
                    exception_count += 1
                else:
                    match_status = "no_match"
                    cluster_id = str(uuid.uuid4())
                    match_score = 0.0
                    new_cluster_count += 1

                assigned_cluster_batch[i] = cluster_id

                record_id = _make_record_id(app_user_id, model_id, r["source_name"], r["source_id"])

                vals: List[Any] = [
                    cluster_id,
                    job_id,
                    record_id,
                    model_id,
                    model_name,
                    r["source_name"],
                    r["source_id"],
                    app_user_id,
                ]

                if recon_has_cluster_size:
                    vals.insert(1, 1)

                for k in range(1, 21):
                    vals.append(r[f"f{str(k).zfill(2)}"])

                vals.append(r["created_at"])
                vals.append(r["created_by"])
                vals.append(r["updated_at"])
                vals.append(r["updated_by"])
                vals.append(match_status)
                vals.append(float(match_score))

                recon_inserts.append(tuple(vals))



                map_rows.append(
                    (
                        app_user_id,
                        model_id,
                        r["source_name"],
                        r["source_id"],
                        cluster_id,
                        now,
                        now,
                    )
                )

                if match_status == "exception" and best_source_name is not None and best_source_id is not None:
                    record_id = _make_record_id(app_user_id, model_id, r["source_name"], r["source_id"])
                    cand_record_id = _make_record_id(app_user_id, model_id, best_source_name, best_source_id)

                    exc_rows.append(
                        (
                            job_id,
                            model_id,
                            record_id,
                            r["source_name"],
                            r["source_id"],
                            cluster_id,
                            cand_record_id,
                            best_source_name,
                            best_source_id,
                            float(best_score),
                            "exception",
                            None,
                            now,
                            None,
                            None,
                            app_user_id,
                        )
                    )

            total_clusters = new_cluster_count


        if recon_inserts:
            conn.executemany(
                f"INSERT INTO recon_cluster ({', '.join(recon_cols)}) VALUES ({recon_placeholders})",
                recon_inserts,
            )

            if recon_has_cluster_size:
                affected_cluster_ids = sorted({str(t[0]) for t in recon_inserts if t and t[0]})
                if affected_cluster_ids:
                    chunk_size = 900
                    for start in range(0, len(affected_cluster_ids), chunk_size):
                        chunk = affected_cluster_ids[start : start + chunk_size]
                        cluster_placeholders = ", ".join(["?"] * len(chunk))
                        params: List[Any] = [app_user_id, model_id] + chunk
                        conn.execute(
                            f"""
                            UPDATE recon_cluster
                            SET cluster_size = (
                                SELECT COUNT(*)
                                FROM recon_cluster rc2
                                WHERE rc2.app_user_id = recon_cluster.app_user_id
                                  AND rc2.model_id = recon_cluster.model_id
                                  AND rc2.cluster_id = recon_cluster.cluster_id
                            )
                            WHERE app_user_id = ?
                              AND model_id = ?
                              AND cluster_id IN ({cluster_placeholders})
                            """,
                            tuple(params),
                        )

        if map_rows:
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
                map_rows,
            )

        if exc_rows:
            conn.executemany(
                """
                INSERT INTO match_exception (
                  job_id, model_id,
                  record_id, source_name, source_id,
                  candidate_cluster_id, candidate_record_id, candidate_source_name, candidate_source_id, score,
                  reason, details_json,
                  created_at, resolved_at, resolved_by,
                  app_user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                exc_rows,
            )

    try:
        run_survivorship(job_id=job_id, app_user_id=int(app_user_id), model=surv_model, actor="match_worker")
    except Exception as e:
        _set_job_status(job_id, "failed", f"survivorship_failed: {e}")
        return

    _set_job_status(
        job_id,
        "completed",
        f"processed={len(src_rows)} match={match_count} exception={exception_count} new_clusters={new_cluster_count}",
        total_records=len(src_rows),
        total_matches=match_count,
        total_clusters=total_clusters,
        total_pairs_scored=total_pairs_scored,
        exceptions_json=json.dumps(
            {
                "exception_count": exception_count,
                "candidate_min": CANDIDATE_MIN,
                "candidate_max": CANDIDATE_MAX,
                "blocking_max_prefix_len": MATCH_BLOCK_MAX_PREFIX_LEN,
                "blocking_fields_by_weight": cfg.get("blocking_fields") or [],
            }
        ),
    )


def _job_process_entry(job_id: str, match_workers: int) -> None:
    print(f"[worker] job_start job_id={job_id} match_workers={match_workers}", flush=True)

    try:
        run_one_job(job_id, match_workers=match_workers)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[worker] job_failed job_id={job_id} err={type(e).__name__}: {e}\n{tb}", flush=True)
        try:
            _set_job_status(job_id, "failed", f"{type(e).__name__}: {e}")
        except Exception:
            pass
        return

    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT status, message FROM match_job WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row:
            print(f"[worker] job_done job_id={job_id} status={row['status']} message={row['message']}", flush=True)
        else:
            print(f"[worker] job_done job_id={job_id} status=unknown message=job_row_missing", flush=True)
    except Exception as e:
        print(f"[worker] job_done_log_error job_id={job_id} err={type(e).__name__}: {e}", flush=True)



def main() -> None:
    print(
        f"Worker started DB_PATH={os.environ.get('DB_PATH', '/data/app.db')} "
        f"poll={POLL_SECONDS} job_workers={JOB_WORKERS} match_workers_total={MATCH_WORKERS_TOTAL}",
        flush=True,
    )
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
                print(
                    f"[worker] claimed job_id={jid} app_user_id={j['app_user_id']} model_id={j['model_id']}",
                    flush=True,
                )
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
