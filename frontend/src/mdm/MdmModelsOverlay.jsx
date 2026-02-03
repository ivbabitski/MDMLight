"use client";

import React, { useEffect, useMemo, useState } from "react";
import "./mdm.css";

import { loadAuth } from "../authStorage";

// MdmModelsOverlay.jsx
const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");


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

  const [sortKey, setSortKey] = useState("id"); // id | name | created
  const [sortDir, setSortDir] = useState("asc"); // asc | desc


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

      setError(`Run queued (job_id: ${data.job_id})`);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setRunningId("");
    }
  }

  useEffect(() => {
    if (!open) return;
    loadModels();
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

        <div className="mdmDialog__body">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            <div style={{ color: "var(--muted)", fontSize: 13 }}>
              {loading ? "Loading..." : error ? error : `${models.length} model(s)`}
            </div>
            <div className="mdmBtnGroup">
              <button className="mdmBtn mdmBtn--soft mdmBtn--xs" type="button" onClick={loadModels} disabled={loading}>
                Refresh
              </button>
            </div>
          </div>

          <div className="mdmTable mdmModelsTable" style={{ overflowX: "auto", position: "relative" }}>

            <div
              className="mdmTHead"
              style={{
                gridTemplateColumns: COLS,
                background: HEAD_BG,
                borderBottom: "1px solid var(--border, rgba(0,0,0,0.10))",
                alignItems: "stretch",
                justifyItems: "stretch",
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

            

            <div className="mdmTableBody" style={{ overflow: "visible" }}>
              {!loading && models.length === 0 ? (
                <div style={{ padding: 12, color: "var(--muted)", fontSize: 13 }}>
                  No models found.
                </div>
              ) : (
                sortedModels.map((m) => {
                  const isJsonOpen = jsonOpenId === String(m.id || "");
                  const createdLabel = fmtDate(m.created_at);
                  const updatedLabel = fmtDate(m.updated_at || m.created_at);

                  return (
                    <React.Fragment key={m.id}>
                      <div
                        className="mdmTRow"
                        style={{
                          gridTemplateColumns: COLS,
                          overflow: "visible",
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
                          <button
                            className="mdmBtn mdmBtn--run mdmBtn--xs"
                            type="button"
                            onClick={() => runModel(m)}
                            disabled={runningId === String(m.id || "")}
                          >
                            {runningId === String(m.id || "") ? "Running..." : "Run"}
                          </button>

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

          <div style={{ color: "var(--muted)", fontSize: 12 }}>
            Delete is a soft-delete (hidden from this list).
          </div>
        </div>
      </div>
    </div>
  );
}
