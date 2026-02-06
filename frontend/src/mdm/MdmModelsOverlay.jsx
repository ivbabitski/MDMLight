"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import "./mdm.css";

import { loadAuth } from "../authStorage";

// MdmModelsOverlay.jsx
const API_BASE = (() => {
  const win =
    (typeof window !== "undefined" && window.__MDM_API_BASE__) ? String(window.__MDM_API_BASE__) : "";
  const env = (import.meta.env.VITE_API_URL || "").trim();
  const raw = String(win || env || "").trim();

  if (raw) return raw.endsWith("/") ? raw.slice(0, -1) : raw;

  if (typeof window === "undefined") return "";
  const proto = window.location.protocol || "http:";
  const host = window.location.hostname || "localhost";
  return `${proto}//${host}:5000`;
})();

const PAGE_SIZE_OPTIONS = [10, 25, 50];

const LS_SELECTED_MODEL_ID = "mdm_selected_model_id";


function apiUrl(path) {
  return `${API_BASE}${path}`;
}


function fmtDate(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
  } catch {
    return String(iso);
  }
}

function safeJson(v) {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v ?? "");
  }
}

function toTime(iso) {
  if (!iso) return 0;
  const d = new Date(iso);
  const t = d.getTime();
  return Number.isNaN(t) ? 0 : t;
}

function getAuthUserId() {
  const auth = loadAuth();
  const raw = auth?.user_id;
  if (raw === null || raw === undefined) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return n;
}

function getAuthUsername() {
  const auth = loadAuth();
  const u = String(auth?.username || "").trim();
  return u || "";
}

export default function MdmModelsOverlay({ open, onClose, currentUser, onRequireLogin, onOpenModel }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [models, setModels] = useState([]);
  const [deletingId, setDeletingId] = useState("");
  const [runningId, setRunningId] = useState("");

  const [runInfoByModelId, setRunInfoByModelId] = useState({});
  const [lastRunModelId, setLastRunModelId] = useState("");
  const pollTimersRef = useRef({});
  const openRef = useRef(Boolean(open));

  const [sortKey, setSortKey] = useState("id"); // id | name | created
  const [sortDir, setSortDir] = useState("asc"); // asc | desc


  const [searchText, setSearchText] = useState("");
  const [pageSize, setPageSize] = useState(10);
  const [page, setPage] = useState(0);


  const [jsonOpenId, setJsonOpenId] = useState("");

  const userLabel = String(currentUser || getAuthUsername() || "").trim();
  const COLS = "minmax(160px, 280px) minmax(260px, 1fr) 140px 180px 180px 260px";
  const HEAD_BG = "#f6f7f9";


  function applySort(nextKey) {



    if (sortKey === nextKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(nextKey);
    setSortDir("asc");
  }

  function sortMark(key) {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " ▲" : " ▼";
  }

  const sortedModels = useMemo(() => {
    const dir = sortDir === "asc" ? 1 : -1;

    function getVal(m) {
      if (sortKey === "name") return String(m?.model_name || "").toLowerCase();
      if (sortKey === "created") return toTime(m?.created_at);
      return String(m?.id || "").toLowerCase();
    }

    return [...(Array.isArray(models) ? models : [])].sort((a, b) => {
      const av = getVal(a);
      const bv = getVal(b);

      if (typeof av === "number" && typeof bv === "number") {
        if (av !== bv) return (av - bv) * dir;
      } else {
        const cmp = String(av).localeCompare(String(bv));
        if (cmp !== 0) return cmp * dir;
      }

      return String(a?.id || "").localeCompare(String(b?.id || ""));
    });
  }, [models, sortDir, sortKey]);

  const filteredModels = useMemo(() => {
    const q = String(searchText || "").trim().toLowerCase();
    if (!q) return sortedModels;

    return sortedModels.filter((m) => {
      const name = String(m?.model_name || "").toLowerCase();
      const id = String(m?.id || "").toLowerCase();
      return name.includes(q) || id.includes(q);
    });
  }, [searchText, sortedModels]);

  const totalFiltered = filteredModels.length;
  const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize));
  const safePage = Math.min(Math.max(0, page), totalPages - 1);
  const pageStart = safePage * pageSize;
  const pageEnd = Math.min(pageStart + pageSize, totalFiltered);

  const pagedModels = useMemo(() => {
    return filteredModels.slice(pageStart, pageEnd);
  }, [filteredModels, pageEnd, pageStart]);

  const lastRunInfo = lastRunModelId ? (runInfoByModelId[String(lastRunModelId)] || {}) : {};
  const runHeaderSuffix =
    lastRunModelId && (lastRunInfo.job_id || lastRunInfo.status)
      ? "Run: " +
        String(lastRunInfo.status || "queued") +
        " (job_id: " +
        String(lastRunInfo.job_id || "") +
        ")" +
        (lastRunInfo.message ? " — " + String(lastRunInfo.message) : "")
      : "";


  useEffect(() => {
    setPage(0);
  }, [pageSize, searchText]);


  useEffect(() => {
    if (page !== safePage) setPage(safePage);
  }, [page, safePage]);

  async function loadModels() {
    setError("");
    setLoading(true);


    try {
      const userId = getAuthUserId();
      const actor = String(currentUser || getAuthUsername() || "").trim();

      if (!userId && !actor) {
        setModels([]);
        setError("Login required to view models.");
        return;
      }

      const headers = { Accept: "application/json" };
      if (userId) headers["X-User-Id"] = String(userId);
      else headers["X-Actor"] = actor;

      const res = await fetch(apiUrl("/mdm/models"), { method: "GET", headers });
      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.ok) {
        throw new Error(data.error || `Failed to load models (${res.status})`);
      }

      setModels(Array.isArray(data.models) ? data.models : []);
    } catch (e) {
      setModels([]);
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function deleteModel(model) {
    if (!model || !model.id) return;

    const actor = String(currentUser || getAuthUsername() || "").trim();
    if (!actor) {
      setError("Login required to delete a model.");
      return;
    }

    const name = String(model.model_name || model.model_key || model.id);
    if (!window.confirm(`Delete model "${name}"?`)) return;

    setDeletingId(String(model.id));
    setError("");

    try {
      const res = await fetch(apiUrl(`/mdm/models/${model.id}`), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.ok) {
        throw new Error(data.error || `Failed to delete model (${res.status})`);
      }

      await loadModels();
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setDeletingId("");
    }
  }


  function clearJobPoll(jobId) {
    const timers = pollTimersRef.current || {};
    const t = timers[String(jobId)];
    if (!t) return;

    try {
      clearTimeout(t);
    } catch {
      // ignore
    }

    delete timers[String(jobId)];
    pollTimersRef.current = timers;
  }

  async function fetchMatchJob(jobId, userId) {
    const res = await fetch(apiUrl(`/match/status/${jobId}`), {
      method: "GET",
      headers: { Accept: "application/json", "X-User-Id": String(userId) },
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || `Failed to fetch job status (${res.status})`);
    }

    return data;
  }

  function scheduleJobPoll(modelId, jobId, userId) {
    clearJobPoll(jobId);

    pollTimersRef.current = pollTimersRef.current || {};
    pollTimersRef.current[String(jobId)] = window.setTimeout(async () => {
      if (!openRef.current) return;

      try {
        const data = await fetchMatchJob(jobId, userId);
        const status = String(data?.status || "");
        const message = String(data?.message || "");

        setRunInfoByModelId((prev) => ({
          ...prev,
          [String(modelId)]: {
            job_id: String(jobId),
            status,
            message,
            updated_at: data?.updated_at || "",
          },
        }));

        if (status && status !== "completed" && status !== "failed") {
          scheduleJobPoll(modelId, jobId, userId);
        } else {
          clearJobPoll(jobId);
        }
      } catch (e) {
        setRunInfoByModelId((prev) => ({
          ...prev,
          [String(modelId)]: {
            job_id: String(jobId),
            status: "status_error",
            message: String(e?.message || e),
            updated_at: "",
          },
        }));
        clearJobPoll(jobId);
      }
    }, 750);
  }

  async function runModel(model) {
    if (!model || !model.id) return;

    const userId = getAuthUserId();
    if (!userId) {
      setError("X-User-Id (app_user_id) is required to run a model.");
      return;
    }

    setRunningId(String(model.id));
    setError("");

    try {
      const res = await fetch(apiUrl("/match/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-User-Id": String(userId) },
        body: JSON.stringify({ model_id: String(model.id) }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.ok) {
        throw new Error(data.error || `Failed to run model (${res.status})`);
      }

      const jobId = String(data.job_id || "");
      const status = String(data.status || "queued");

      setRunInfoByModelId((prev) => ({
        ...prev,
        [String(model.id)]: { job_id: jobId, status, message: "", updated_at: "" },
      }));

      if (jobId) {
        scheduleJobPoll(String(model.id), jobId, userId);
      }

      setLastRunModelId(String(model.id));
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setRunningId("");
    }
  }

  function selectModel(model) {
    if (!model || !model.id) return;

    const id = String(model.id || "").trim();

    try {
      if (id) localStorage.setItem(LS_SELECTED_MODEL_ID, id);
      else localStorage.removeItem(LS_SELECTED_MODEL_ID);
    } catch {}

    try {
      window.dispatchEvent(new CustomEvent("mdm:selected_model_changed", { detail: { model_id: id } }));
    } catch {}

    if (onClose) onClose();
  }


  useEffect(() => {
    openRef.current = Boolean(open);

    if (!open) {
      const timers = pollTimersRef.current || {};
      Object.values(timers).forEach((t) => {
        try {
          clearTimeout(t);
        } catch {
          // ignore
        }
      });
      pollTimersRef.current = {};
      return;
    }

    loadModels();

    return () => {
      const timers = pollTimersRef.current || {};
      Object.values(timers).forEach((t) => {
        try {
          clearTimeout(t);
        } catch {
          // ignore
        }
      });
      pollTimersRef.current = {};
    };
  }, [open]);


  if (!open) return null;

  return (
    <div className="mdmOverlay" role="dialog" aria-modal="true" aria-label="MDM models">
      <div className="mdmDialog" style={{ width: "min(1200px, calc(100vw - 48px))" }}>
        <div className="mdmDialog__head">
          <div>
            <div className="mdmDialog__title">MDM models</div>
            <div className="mdmDialog__sub">Simple list (open, view JSON, delete)</div>
          </div>
          <button className="mdmX" type="button" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="mdmDialog__body" style={{ alignContent: "start", gridAutoRows: "max-content" }}>
          <div className="mdmRecordsHead" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            <div style={{ minWidth: 240, flex: "1 1 240px" }}>
              <div style={{ color: "var(--muted)", fontSize: 13 }}>
                {loading ? "Loading..." : error ? error : runHeaderSuffix}
              </div>
            </div>


            <div className="mdmInputWithIcon" style={{ flex: "0 0 340px", maxWidth: "44vw", minWidth: 220 }}>
              <input
                className="mdmInput mdmInput--withIcon"
                type="search"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Search"
                aria-label="Search models"
                disabled={loading}
              />
              {searchText ? (
                <button
                  type="button"
                  className="mdmInputIconBtn"
                  onClick={() => setSearchText("")}
                  title="Clear search"
                  aria-label="Clear search"
                >
                  ✕
                </button>
              ) : null}
            </div>

            <div className="mdmBtnGroup" style={{ marginLeft: "auto" }}>
              <button className="mdmBtn mdmBtn--soft" type="button" onClick={loadModels} disabled={loading}>
                Refresh
              </button>
            </div>
          </div>




          <div className="mdmTable mdmModelsTable" style={{ overflowX: "auto", overflowY: "auto", maxHeight: "min(56vh, 560px)", position: "relative" }}>


            <div
              className="mdmTHead"
              style={{
                gridTemplateColumns: COLS,
                background: HEAD_BG,
                borderBottom: "1px solid var(--border, rgba(0,0,0,0.10))",
                alignItems: "stretch",
                justifyItems: "stretch",

                position: "sticky",
                top: 0,
                zIndex: 4,
              }}
            >


              <div
                style={{ fontWeight: 900, cursor: "pointer", userSelect: "none", textAlign: "center" }}
                onClick={() => applySort("name")}
                title="Sort by name"
              >
                Model name{sortMark("name")}
              </div>
              <div
                style={{ fontWeight: 900, cursor: "pointer", userSelect: "none", textAlign: "center" }}
                onClick={() => applySort("id")}
                title="Sort by model id"
              >
                Model id{sortMark("id")}
              </div>
              <div style={{ fontWeight: 900, textAlign: "center" }}>User</div>
              <div
                style={{ fontWeight: 900, cursor: "pointer", userSelect: "none", textAlign: "center" }}
                onClick={() => applySort("created")}
                title="Sort by created date"
              >
                Created{sortMark("created")}
              </div>
              <div style={{ fontWeight: 900, textAlign: "center" }}>Updated</div>
              <div
                style={{
                  fontWeight: 900,
                  textAlign: "center",
                  position: "sticky",
                  right: 0,
                  zIndex: 3,
                  background: "inherit",
                  borderLeft: "1px solid var(--border, rgba(0,0,0,0.12))",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                Actions
              </div>

            </div>

            

            <div className="mdmTableBody" style={{ maxHeight: "none", overflow: "visible" }}>
              {!loading && filteredModels.length === 0 ? (
                <div style={{ padding: 12, color: "var(--muted)", fontSize: 13 }}>
                  {searchText ? "No matches." : "No models found."}
                </div>
              ) : (
                pagedModels.map((m) => {
                  const isJsonOpen = jsonOpenId === String(m.id || "");
                  const createdLabel = fmtDate(m.created_at);
                  const updatedLabel = fmtDate(m.updated_at || m.created_at);
                  const runInfo = runInfoByModelId[String(m.id || "")] || {};
                  const runStatus = String(runInfo.status || "");
                  const disableRun =
                    runningId === String(m.id || "") || runStatus === "queued" || runStatus === "running";
                  const runBtnLabel =
                    runningId === String(m.id || "")
                      ? "Running..."
                      : runStatus === "queued"
                        ? "Queued..."
                        : runStatus === "running"
                          ? "Running..."
                          : "Run";


                  return (
                    <React.Fragment key={m.id}>
                      <div
                        className="mdmTRow"
                        onClick={() => selectModel(m)}
                        style={{
                          gridTemplateColumns: COLS,
                          overflow: "visible",
                          cursor: "pointer",
                        }}
                      >

                        <div title={m.model_name || ""} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {m.model_name || ""}
                        </div>

                        <div title={m.id || ""} className="mdmMono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {m.id || ""}
                        </div>

                        <div title={userLabel} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {userLabel}
                        </div>

                        <div title={createdLabel} style={{ color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {createdLabel || "—"}
                        </div>

                        <div title={updatedLabel} style={{ color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {updatedLabel || "—"}
                        </div>

                        <div
                          onClick={(e) => {
                            e.stopPropagation();
                          }}
                          style={{
                            display: "flex",
                            gap: 4,
                            justifyContent: "center",
                            alignItems: "center",
                            flexWrap: "nowrap",
                            position: "sticky",
                            right: 0,
                            zIndex: 1,
                            background: "var(--panel, #fff)",
                            borderLeft: "1px solid var(--border, rgba(0,0,0,0.12))",
                          }}
                        >
                          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                            <button
                              className="mdmBtn mdmBtn--run mdmBtn--xs"
                              type="button"
                              onClick={() => runModel(m)}
                              disabled={disableRun}
                            >
                              {runBtnLabel}
                            </button>
                          </div>

                          <a
                            className="mdmBtn mdmBtn--ghost mdmBtn--xs"
                            href="#"
                            onClick={(e) => {
                              e.preventDefault();
                              if (onOpenModel) onOpenModel(m);
                            }}
                          >
                            Open
                          </a>

                          <button
                            className="mdmBtn mdmBtn--soft mdmBtn--xs"
                            type="button"
                            onClick={() => setJsonOpenId(isJsonOpen ? "" : String(m.id || ""))}
                          >
                            {isJsonOpen ? "Hide JSON" : "View JSON"}
                          </button>

                          <button
                            className="mdmBtn mdmBtn--danger mdmBtn--xs"
                            type="button"
                            onClick={() => deleteModel(m)}
                            disabled={deletingId === m.id}
                          >
                            {deletingId === m.id ? "Deleting..." : "Delete"}
                          </button>
                        </div>
                      </div>

                      {isJsonOpen ? (
                        <div
                          className="mdmTRow"
                          style={{
                            gridTemplateColumns: COLS,
                            overflow: "visible",
                          }}
                        >
                          <div style={{ gridColumn: "1 / -1" }}>
                            <div className="mdmLabel" style={{ marginTop: 4 }}>raw model json</div>
                            <pre
                              className="mdmPre"
                              style={{
                                margin: 0,
                                width: "100%",
                                maxHeight: 260,
                                overflowY: "auto",
                                overflowX: "hidden",
                                whiteSpace: "pre-wrap",
                                overflowWrap: "anywhere",
                                maxWidth: "100%",
                                boxSizing: "border-box",
                              }}
                            >
                              {safeJson(m)}
                            </pre>


                          </div>
                        </div>
                      ) : null}
                    </React.Fragment>
                  );
                })
              )}
            </div>
          </div>

          <div className="mdmRecordsFooter" style={{ marginTop: 8 }} role="navigation" aria-label="Models pagination">
            <div className="mdmRecordsFooterLeft">
              <span className="mdmTiny">Rows per page</span>
              <select
                className="mdmSelect mdmSelect--xs"
                value={pageSize}
                onChange={(e) => setPageSize(Number(e.target.value) || 10)}
                aria-label="Rows per page"
                disabled={loading}
              >
                {PAGE_SIZE_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>

            <div className="mdmRecordsFooterMid">
              <span className="mdmTiny">
                Showing {totalFiltered ? pageStart + 1 : 0}–{totalFiltered ? pageEnd : 0} of {totalFiltered}
              </span>
            </div>

            <div className="mdmRecordsFooterRight" style={{ alignItems: "center" }}>
              <button
                className="mdmBtn mdmBtn--xs mdmBtn--soft"
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={loading || safePage <= 0}
                aria-label="Previous page"
                title="Previous page"
                style={{ width: 34, height: 34, padding: 0, display: "grid", placeItems: "center", lineHeight: 1 }}
              >
                ‹
              </button>

              <span className="mdmTag mdmTag--soft" style={{ height: 34, display: "inline-flex", alignItems: "center" }}>
                {safePage + 1} / {totalPages}
              </span>

              <button
                className="mdmBtn mdmBtn--xs mdmBtn--soft"
                type="button"
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={loading || safePage >= totalPages - 1 || totalFiltered === 0}
                aria-label="Next page"
                title="Next page"
                style={{ width: 34, height: 34, padding: 0, display: "grid", placeItems: "center", lineHeight: 1 }}
              >
                ›
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
