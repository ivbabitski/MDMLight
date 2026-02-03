"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./mdm.css";
import MdmWizard from "./MdmWizard.jsx";
import Login from "./Login.jsx";
import MdmModelsOverlay from "./MdmModelsOverlay.jsx";
import { loadAuth, clearAuth } from "../authStorage";


const LOGO_CANDIDATES = ["/Logo.png", "/Logo.svg"];
const LS_SELECTED_MODEL_ID = "mdm_selected_model_id";


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


const SOURCE_INPUT_FIELD_KEYS = Array.from({ length: 20 }, (_, i) => `f${String(i + 1).padStart(2, "0")}`);


/** UI-only mocks. Replace with SQL-backed data later. */
const MOCK_JOB = {
  job_id: "job_demo_001",
  status: "completed",
  total_records: 5,
  model_json: {
    matching_model: "Model v1",
    match_threshold: 0.85,
    survivorship_strategy: "Strategy 0",
    match_fields: ["name", "email", "phone"],
  },
  exceptions_json: {
    rules: [{ key: "missing_mandatory", action: "review" }],
  },
};


const UI_FIELDS = [
  { key: "name", label: "Name" },
  { key: "email", label: "Email" },
  { key: "phone", label: "Phone" },
  { key: "address", label: "Address" },
];

const USER_FIELD_KEYS = Array.from({ length: 20 }, (_, i) => `f${i + 1}`);

const MOCK_RECON = [

  {
    cluster_id: "CL-00041",
    record_id: "rec_000901",
    source_name: "CRM",
    source_id: "CRM-10901",
    cluster_size: 4,
    is_representative: 1,
    fields: { name: "Acme Inc", email: "billing@acme.io", phone: "+1 (415) 555-0199", address: "1 Market St, SF" },
  },
  {
    cluster_id: "CL-00041",
    record_id: "rec_000902",
    source_name: "ERP",
    source_id: "ERP-88012",
    cluster_size: 4,
    is_representative: 0,
    fields: { name: "ACME Incorporated", email: "billing@acme.io", phone: "", address: "1 Market Street, San Francisco" },
  },
  {
    cluster_id: "CL-00041",
    record_id: "rec_000903",
    source_name: "Support",
    source_id: "SUP-45019",
    cluster_size: 4,
    is_representative: 0,
    fields: { name: "Acme", email: "", phone: "+1 (415) 555-0199", address: "" },
  },
  {
    cluster_id: "CL-00012",
    record_id: "rec_000210",
    source_name: "ERP",
    source_id: "ERP-77001",
    cluster_size: 3,
    is_representative: 1,
    fields: { name: "Northwind Traders", email: "ap@northwind.com", phone: "+1 (212) 555-0101", address: "125 Madison Ave, NY" },
  },
  {
    cluster_id: "CL-00012",
    record_id: "rec_000212",
    source_name: "Support",
    source_id: "SUP-11801",
    cluster_size: 3,
    is_representative: 0,
    fields: { name: "", email: "support@northwind.com", phone: "", address: "" },
  },
];

function fmtInt(n) {
  const x = Number(n || 0);
  return x.toLocaleString();
}

function safeJson(v) {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v ?? "");
  }
}


async function fetchSourceInputSummary(appUserId, modelId) {
  const base = String(API_BASE || "").trim();
  const mid = String(modelId || "").trim();
  const url = `${base}/api/source-input/summary?model_id=${encodeURIComponent(mid)}&t=${Date.now()}`;

  const userId = String(appUserId || "").trim();

  const headers = {
    Accept: "application/json",
  };
  if (userId) headers["X-User-Id"] = userId;

  const res = await fetch(url, {
    cache: "no-store",
    headers,
  });


  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`source-input summary failed (HTTP ${res.status}). ${txt.slice(0, 120)}`);
  }

  const ct = String(res.headers.get("content-type") || "");
  if (!ct.toLowerCase().includes("application/json")) {
    const txt = await res.text().catch(() => "");
    throw new Error(
      `source-input summary expected JSON but got "${ct || "unknown"}". ` +
        `URL="${url}". First bytes: ${txt.slice(0, 120)}`
    );
  }

  return res.json();
}


export default function MdmPage() {


  // Logo fallback
  const [logoIdx, setLogoIdx] = useState(0);
  const logoSrc = logoIdx < LOGO_CANDIDATES.length ? LOGO_CANDIDATES[logoIdx] : null;

  // Wizard open/close only
  const [wizardOpen, setWizardOpen] = useState(false);

  const [loginOpen, setLoginOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  const [currentUser, setCurrentUser] = useState("");
  const [currentUserId, setCurrentUserId] = useState("");
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const accountMenuRef = useRef(null);
  const prevSelectedModelIdRef = useRef("");

  const [modelsOpen, setModelsOpen] = useState(false);

  const [sourceSummary, setSourceSummary] = useState(null);
  const [sourceSummaryErr, setSourceSummaryErr] = useState("");

  const refreshSourceSummary = useCallback(async () => {
    const userId = String(currentUserId || "").trim();
    if (!userId) {
      setSourceSummary(null);
      setSourceSummaryErr("");
      return;
    }

    let modelId = "";
    try {
      modelId = String(localStorage.getItem(LS_SELECTED_MODEL_ID) || "").trim();
    } catch {}

    if (!modelId) {
      setSourceSummary(null);
      setSourceSummaryErr("model_id is required (select a model)");
      return;
    }

    setSourceSummaryErr("");
    try {
      const data = await fetchSourceInputSummary(userId, modelId);
      setSourceSummary(data);
    } catch (e) {
      setSourceSummary(null);
      setSourceSummaryErr(String(e?.message || e));
    }
  }, [currentUserId]);


  useEffect(() => {
    try {
      const a = loadAuth?.();
      const uid = a?.user_id != null ? String(a.user_id).trim() : "";
      const uname = a?.username != null ? String(a.username).trim() : "";

      setCurrentUserId(uid);
      setCurrentUser(uname);
    } catch {
      setCurrentUserId("");
      setCurrentUser("");
    }
  }, []);


  useEffect(() => {
    function onMouseDown(e) {
      if (!accountMenuOpen) return;
      if (accountMenuRef.current && !accountMenuRef.current.contains(e.target)) {
        setAccountMenuOpen(false);
      }
    }

    function onKeyDown(e) {
      if (!accountMenuOpen) return;
      if (e.key === "Escape") setAccountMenuOpen(false);
    }

    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [accountMenuOpen]);


  useEffect(() => {
    refreshSourceSummary();
  }, [refreshSourceSummary]);

  useEffect(() => {
    function onUpdated() {
      refreshSourceSummary();
    }

    window.addEventListener("mdm:source_input_updated", onUpdated);

    return () => window.removeEventListener("mdm:source_input_updated", onUpdated);
  }, [refreshSourceSummary]);



  // NEW (UI only): dashboard + records review
  const [view, setView] = useState("golden"); // golden | exceptions
  const [q, setQ] = useState("");
  const [modelOpen, setModelOpen] = useState(false);

  const [promoted, setPromoted] = useState({}); // uiKey -> boolean (exceptions view)



  const job = MOCK_JOB;

  const totalSourceRecords = useMemo(() => Number(sourceSummary?.total_records || 0), [sourceSummary]);
  const fieldsWithData = useMemo(() => Number(sourceSummary?.fields_with_data || 0), [sourceSummary]);

  const sourceFieldKeys = useMemo(() => {
    if (Array.isArray(sourceSummary?.field_keys) && sourceSummary.field_keys.length) return sourceSummary.field_keys;
    return SOURCE_INPUT_FIELD_KEYS;
  }, [sourceSummary]);

  const sourceFieldLabels = useMemo(() => {
    const pills = Array.isArray(sourceSummary?.field_pills) ? sourceSummary.field_pills : [];
    return pills.map((x) => String(x || "").trim()).filter(Boolean);
  }, [sourceSummary]);


  const matchKeys = useMemo(() => job?.model_json?.match_fields || [], [job]);

  const survivorshipStrategy = useMemo(
    () => job?.model_json?.survivorship_strategy || "Strategy 0",
    [job]
  );


  const rows = useMemo(() => {
    return (MOCK_RECON || []).map((r) => {
      const f1 = String(r.fields?.name || "");
      const f2 = String(r.fields?.email || "");
      const f3 = String(r.fields?.phone || "");
      const f4 = String(r.fields?.address || "");

      const out = {
        ...r,
        job_id: job.job_id,
        uiKey: `${r.cluster_id}::${r.record_id}`,

        matching_model: String(job?.model_json?.matching_model || "Model v1"),
        master_id: String(r.cluster_id),
        match_threshold: job?.model_json?.match_threshold ?? "",
        survivorship_strategy: String(job?.model_json?.survivorship_strategy || "Strategy 0"),

        f1, f2, f3, f4,

        created_at: "2026-01-29T00:00:00Z",
        created_by: "system",
        updated_at: "",
        updated_by: "",
      };

      for (const k of USER_FIELD_KEYS) {
        if (out[k] == null) out[k] = "";
      }

      return out;
    });
  }, [job]);


  const recordsPerSource = useMemo(() => {
    if (Array.isArray(sourceSummary?.sources)) return sourceSummary.sources;
    return [];
  }, [sourceSummary]);


  const maxPerSource = useMemo(() => Math.max(...recordsPerSource.map((x) => x.count), 1), [recordsPerSource]);

  const totalClusters = useMemo(() => new Set(rows.map((r) => r.cluster_id)).size, [rows]);

  const goldenRows = useMemo(() => rows.filter((r) => r.is_representative === 1), [rows]);
  const exceptionRows = useMemo(() => rows.filter((r) => r.is_representative !== 1), [rows]);

  const listRows = useMemo(() => {
    const base = view === "golden" ? goldenRows : exceptionRows;
    const query = q.trim().toLowerCase();

    if (!query) return base;

    return base.filter((r) => {
      const hay = [
        r.matching_model,
        r.master_id,
        String(r.match_threshold),
        r.survivorship_strategy,
        ...USER_FIELD_KEYS.map((k) => r[k]),
        r.created_at,
        r.created_by,
        r.updated_at,
        r.updated_by,
      ].filter(Boolean).join(" | ").toLowerCase();

      return hay.includes(query);
    });
  }, [exceptionRows, goldenRows, q, view]);

  function togglePromote(row) {
    setPromoted((prev) => ({ ...prev, [row.uiKey]: !prev[row.uiKey] }));
    console.log("[MDM ACTION] promote_to_master_toggle", {
      job_id: row.job_id,
      master_id: row.master_id,
      record_id: row.record_id,
      source_name: row.source_name,
      source_id: row.source_id,
    });
  }

  function approveMatch(row) {
    console.log("[MDM ACTION] approve_match", {
      job_id: row.job_id,
      master_id: row.master_id,
      record_id: row.record_id,
      source_name: row.source_name,
      source_id: row.source_id,
    });
  }

  function rejectMatch(row) {
    console.log("[MDM ACTION] reject_match", {
      job_id: row.job_id,
      master_id: row.master_id,
      record_id: row.record_id,
      source_name: row.source_name,
      source_id: row.source_id,
    });
  }




  function onLogoError() {
    setLogoIdx((i) => i + 1);
  }

  function openWizard() {
    try {
      const prev = String(localStorage.getItem(LS_SELECTED_MODEL_ID) || "").trim();
      prevSelectedModelIdRef.current = prev;
      localStorage.removeItem(LS_SELECTED_MODEL_ID);
    } catch {
      prevSelectedModelIdRef.current = "";
    }
    setWizardOpen(true);
  }

  function openWizardForModel(model) {
    try {
      const id = String((model && model.id) || "");
      if (id) localStorage.setItem(LS_SELECTED_MODEL_ID, id);
      else localStorage.removeItem(LS_SELECTED_MODEL_ID);
    } catch {}
    setModelsOpen(false);
    setWizardOpen(true);
  }

  function closeWizard() {
    setWizardOpen(false);

    try {
      const current = String(localStorage.getItem(LS_SELECTED_MODEL_ID) || "").trim();
      if (!current) {
        const prev = String(prevSelectedModelIdRef.current || "").trim();
        if (prev) localStorage.setItem(LS_SELECTED_MODEL_ID, prev);
      }
      prevSelectedModelIdRef.current = "";
    } catch {}

    refreshSourceSummary();
  }




  function handleLogin({ user_id, username }) {
    const uid = String(user_id ?? "").trim();
    const u = String(username || "").trim();

    setCurrentUserId(uid);
    setCurrentUser(u);
    setAccountMenuOpen(false);
  }

  function handleLogout() {
    setCurrentUserId("");
    setCurrentUser("");
    setAccountMenuOpen(false);
    try {
      clearAuth?.();
      localStorage.removeItem("username");
    } catch {}
  }


  return (
    <div className="mdm">
      {/* TOP HEADER ONLY */}
      <div className="mdmTopbar">
        <div className="mdmBrand">
          <div className="mdmMark" aria-hidden="true">
            {logoSrc ? (
              <img
                className="mdmMark__img"
                src={logoSrc}
                alt="MDM Light logo"
                onError={onLogoError}
              />
            ) : (
              <span className="mdmMark__fallback">MDM</span>
            )}
          </div>

          <div className="mdmTitle">
            <div className="mdmTitle__main">MDM Light</div>
            <div className="mdmTitle__sub">MDM in 3 easy steps</div>
          </div>
        </div>

        {/* 2 HEADER BUTTONS ONLY */}
<div className="mdmTopbarActions">

          <div
            className="mdmTopUserLabel"
            title={currentUser ? `Logged in as: ${currentUser}` : "Not logged in"}
          >
            {currentUser ? (
              <>
                Logged in as: <span className="mdmTopUserLabel__name">{currentUser}</span>
              </>
            ) : (
              <>Not logged in</>
            )}
          </div>

          <button
            className="mdmTopIconBtn"
            type="button"
            onClick={() => setModelsOpen(true)}
            title="MDM models"
            aria-label="MDM models"
          >
            <svg viewBox="0 0 24 24" width="24" height="24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-6z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path d="M14 2v6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="11" cy="14" r="2.5" stroke="currentColor" strokeWidth="2" />
              <path d="M13 16l2 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>



          <div
            ref={accountMenuRef}
            style={{
              position: "relative",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
              <button
                className="mdmTopIconBtn"
                type="button"
                onClick={() => setAccountMenuOpen((v) => !v)}
                title={currentUser ? `Account: ${currentUser}` : "Account"}
                aria-label={currentUser ? "Account" : "Account"}
              >
                <svg viewBox="0 0 24 24" width="24" height="24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="8" r="3" stroke="currentColor" strokeWidth="2"/>
                  <path d="M5.5 20c1.6-3.2 4-4.8 6.5-4.8s4.9 1.6 6.5 4.8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>

              {accountMenuOpen ? (
                <div
                  style={{
                    position: "absolute",
                    top: "calc(100% + 10px)",
                    right: 0,
                    minWidth: 220,
                    background: "rgba(255,255,255,0.98)",
                    border: "1px solid rgba(0,0,0,0.12)",
                    borderRadius: 12,
                    padding: 12,
                    boxShadow: "0 14px 40px rgba(0,0,0,0.25)",
                    zIndex: 50,
                    color: "rgba(17,24,39,0.92)",
                  }}
                >
                  <button
                    type="button"
                    onClick={() => {
                      setAccountMenuOpen(false);
                      setLoginOpen(true);
                    }}
                    style={{
                      width: "100%",
                      padding: "9px 10px",
                      borderRadius: 10,
                      border: "1px solid rgba(0,0,0,0.14)",
                      background: "transparent",
                      cursor: "pointer",
                      fontWeight: 900,
                      marginBottom: 8,
                    }}
                  >
                    Login
                  </button>

                  <button
                    type="button"
                    onClick={handleLogout}
                    style={{
                      width: "100%",
                      padding: "9px 10px",
                      borderRadius: 10,
                      border: "1px solid rgba(0,0,0,0.14)",
                      background: "transparent",
                      cursor: "pointer",
                      fontWeight: 900,
                      marginBottom: 8,
                    }}
                  >
                    Logout
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setAccountMenuOpen(false);
                    }}
                    style={{
                      width: "100%",
                      padding: "9px 10px",
                      borderRadius: 10,
                      border: "1px solid rgba(0,0,0,0.14)",
                      background: "transparent",
                      cursor: "pointer",
                      fontWeight: 900,
                    }}
                  >
                    Help
                  </button>
                </div>
              ) : null}
          </div>
        </div>
      </div>

            

      {/* START SETUP ONLY */}
      <div className="mdmMain mdmMain--dash">
        <div className="mdmMainInner">


          <div className="mdmHero">
            <div>
              <h2>Start with setup</h2>
              <p>MDM 1 → Source, MDM 2 → Fields, MDM 3 → Matching & survivorship.</p>
            </div>
            <button className="mdmBtn mdmBtn--primary" type="button" onClick={openWizard}>
              Start setup
            </button>
          </div>

          {/* NEW: Top dashboard */}
          <div className="mdmGrid2">

            <div className="mdmCard">
              <div className="mdmCard__head">
                <div>
                  <div className="mdmCard__title">Source information</div>
                  <div className="mdmCard__sub">
                    {sourceSummaryErr ? `Source input summary error: ${sourceSummaryErr}` : "Total records, per source, fields"}
                  </div>
                </div>

                <button
                  className="mdmBtn mdmBtn--xs mdmBtn--soft mdmIconBtn"
                  type="button"
                  onClick={refreshSourceSummary}
                  title="Refresh source stats"
                  aria-label="Refresh source stats"
                >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <path d="M3 12a9 9 0 0 1 15-6.7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M3 4v6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M21 12a9 9 0 0 1-15 6.7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M21 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                </button>
              </div>
              <div className="mdmCard__body">
                <div className="mdmKpiRow">
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(totalSourceRecords)}</div>
                    <div className="mdmKpiLab">Total source records</div>
                  </div>
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(recordsPerSource.length)}</div>
                    <div className="mdmKpiLab">Sources</div>
                  </div>
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(fieldsWithData)}</div>
                    <div className="mdmKpiLab">Fields with data</div>
                  </div>
                </div>


                <div className="mdmDivider" />

                <div className="mdmSectionTitle">Records per source</div>
                <div className="mdmBarList">
                  {recordsPerSource.map((s) => {
                    const pct = Math.round((s.count / maxPerSource) * 100);
                    return (
                      <div className="mdmBarRow" key={s.source}>
                        <div className="mdmBarName">{s.source}</div>
                        <div className="mdmBarTrack" aria-hidden="true">
                          <span className="mdmBarFill" style={{ width: `${pct}%` }} />
                        </div>
                        <div className="mdmBarVal">{fmtInt(s.count)}</div>
                      </div>
                    );
                  })}
                </div>

                <div className="mdmDivider" />

                <div className="mdmSectionTitle">Fields</div>
                <div className="mdmPillRow mdmFieldsHover">
                  {sourceFieldLabels.length === 0 ? (
                    <span className="mdmPillSoft">—</span>
                  ) : sourceFieldLabels.length <= 6 ? (
                    sourceFieldLabels.map((lbl) => (
                      <span className="mdmPillSoft" key={lbl}>{lbl}</span>
                    ))
                  ) : (
                    <>
                      {sourceFieldLabels.slice(0, 4).map((lbl) => (
                        <span className="mdmPillSoft" key={lbl}>{lbl}</span>
                      ))}

                      <span className="mdmPillSoft mdmFieldsEllipsis">…</span>

                      <span className="mdmFieldsExtra">
                        {sourceFieldLabels.slice(4, -1).map((lbl) => (
                          <span className="mdmPillSoft" key={lbl}>{lbl}</span>
                        ))}
                      </span>

                      <span className="mdmPillSoft">{sourceFieldLabels[sourceFieldLabels.length - 1]}</span>
                    </>
                  )}
                </div>

              </div>
            </div>

            <div className="mdmCard">
              <div className="mdmCard__head">
                <div>
                  <div className="mdmCard__title">Matching & survivorship</div>
                  <div className="mdmCard__sub">Golden, exceptions, clusters</div>
                </div>
                <button
                  className="mdmBtn mdmBtn--xs mdmBtn--soft mdmIconBtn"
                  type="button"
                  onClick={() => setModelOpen(true)}
                  title="Matching model"
                  aria-label="Matching model"
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M4 6h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M7 6v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M4 12h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M17 12v6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M4 18h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>

              <div className="mdmCard__body">
                <div className="mdmKpiRow">
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(goldenRows.length)}</div>
                    <div className="mdmKpiLab">Golden records</div>
                  </div>
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(exceptionRows.length)}</div>
                    <div className="mdmKpiLab">Exceptions</div>
                  </div>
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(totalClusters)}</div>
                    <div className="mdmKpiLab">Match clusters</div>
                  </div>
                </div>

                <div className="mdmDivider" />

                <div className="mdmSectionTitle">Matching fields</div>
                <div className="mdmPillRow">
                  {matchKeys.map((k) => {
                    const label = UI_FIELDS.find((f) => f.key === k)?.label || k;
                    return <span className="mdmPillSoft" key={k}>{label}</span>;
                  })}
                </div>

                <div className="mdmDivider" />

                <div className="mdmSectionTitle">Survivorship strategy</div>
                <div className="mdmPillRow">
                  <span className="mdmPillSoft mdmPillSoft--strong">{survivorshipStrategy}</span>
                </div>

              </div>
            </div>
          </div>

          {/* NEW: Records table */}
          <div className="mdmCard">
            <div className="mdmCard__head">
              <div>
                <div className="mdmCard__title">Records</div>
                <div className="mdmCard__sub">Golden + exceptions (table)</div>
              </div>

              <div className="mdmBtnGroup">
                <div className="mdmSeg" role="tablist" aria-label="Records toggle">
                  <button
                    type="button"
                    className={view === "golden" ? "isActive" : ""}
                    onClick={() => { setView("golden"); setQ(""); }}
                    role="tab"
                    aria-selected={view === "golden"}
                  >
                    Golden
                  </button>
                  <button
                    type="button"
                    className={view === "exceptions" ? "isActive" : ""}
                    onClick={() => { setView("exceptions"); setQ(""); }}
                    role="tab"
                    aria-selected={view === "exceptions"}
                  >
                    Exceptions
                  </button>
                </div>

                <span className="mdmTag mdmTag--soft">{fmtInt(listRows.length)} shown</span>
              </div>
            </div>

            <div className="mdmCard__body">
              <div className="mdmWideTableWrap">
                <table className={`mdmWideTable ${view === "exceptions" ? "isExceptions" : "isGolden"}`}>

                  <thead>
                    <tr>
                      <th>matching_model</th>
                      <th>master_id</th>
                      <th>match_threshold</th>
                      <th>survivorship_strategy</th>
                      {USER_FIELD_KEYS.map((k) => (
                        <th key={k}>{k.toUpperCase()}</th>
                      ))}
                      <th>created_at</th>
                      <th>created_by</th>
                      <th>updated_at</th>
                      <th>updated_by</th>
                      <th className="mdmWideStickyCol">actions</th>

                    </tr>
                  </thead>

                  <tbody>
                    {listRows.map((r) => {
                      const isPromoted = !!promoted[r.uiKey];

                      return (
                        <tr key={r.uiKey}>
                          <td className="mdmMono">{r.matching_model}</td>
                          <td className="mdmMono">{r.master_id}</td>
                          <td className="mdmMono">{String(r.match_threshold)}</td>
                          <td>{r.survivorship_strategy}</td>

                          {USER_FIELD_KEYS.map((k) => (
                            <td key={k}>{String(r[k] || "").trim() ? r[k] : "—"}</td>
                          ))}

                          <td className="mdmMono">{r.created_at}</td>
                          <td className="mdmMono">{r.created_by}</td>
                          <td className="mdmMono">{r.updated_at}</td>
                          <td className="mdmMono">{r.updated_by}</td>

                          <td className="mdmWideStickyCol">
                            {view === "exceptions" ? (
                              <div className="mdmRowActions mdmRowActions--right">
                                <button
                                  className={`mdmBtn mdmBtn--xs mdmIconBtn ${isPromoted ? "mdmBtn--soft" : "mdmBtn--primary"}`}
                                  type="button"
                                  onClick={() => togglePromote(r)}
                                  title={isPromoted ? "Unpromote" : "Promote to master"}
                                  aria-label={isPromoted ? "Unpromote" : "Promote to master"}
                                >
                                  {isPromoted ? (
                                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                      <path d="M12 5v14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                      <path d="M7 15l5 5 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                    </svg>
                                  ) : (
                                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                      <path d="M12 19V5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                      <path d="M7 9l5-5 5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                    </svg>
                                  )}
                                </button>

                                <button
                                  className="mdmBtn mdmBtn--xs mdmBtn--soft mdmIconBtn"
                                  type="button"
                                  onClick={() => approveMatch(r)}
                                  title="Approve match"
                                  aria-label="Approve match"
                                >
                                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                    <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                  </svg>
                                </button>

                                <button
                                  className="mdmBtn mdmBtn--xs mdmBtn--danger mdmIconBtn"
                                  type="button"
                                  onClick={() => rejectMatch(r)}
                                  title="Reject match"
                                  aria-label="Reject match"
                                >
                                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                    <path d="M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                    <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                  </svg>
                                </button>
                              </div>
                            ) : (
                              <span className="mdmTiny">—</span>
                            )}
                          </td>

                        </tr>
                      );
                    })}

                    {listRows.length === 0 ? (
                      <tr>
                        <td
                          colSpan={4 + USER_FIELD_KEYS.length + 4 + 1}
                          className="mdmWideEmpty"
                        >
                          No results.

                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

        </div>

      </div>

      {/* MDM models (NEW) */}
      {modelOpen ? (
        <div className="mdmOverlay" role="dialog" aria-modal="true" aria-label="MDM models">
          <div className="mdmDialog">
            <div className="mdmDialog__head">
              <div>
                <div className="mdmDialog__title">MDM models</div>
                <div className="mdmDialog__sub">model_json + exceptions_json</div>
              </div>
              <button className="mdmX" type="button" onClick={() => setModelOpen(false)} aria-label="Close">
                ✕
              </button>
            </div>

            <div className="mdmDialog__body">
              <div>
                <div className="mdmLabel">model_json</div>
                <pre className="mdmPre">{safeJson(job.model_json)}</pre>
              </div>
              <div>
                <div className="mdmLabel">exceptions_json</div>
                <pre className="mdmPre">{safeJson(job.exceptions_json)}</pre>
              </div>
            </div>
          </div>
        </div>
      ) : null}


      {/* Help */}
      {helpOpen ? (
        <div className="mdmOverlay" role="dialog" aria-modal="true" aria-label="Help">
          <div className="mdmDialog" style={{ width: 560 }}>
            <div className="mdmDialog__head">
              <div>
                <div className="mdmDialog__title">Help</div>
                <div className="mdmDialog__sub">Quick usage notes</div>
              </div>
              <button className="mdmX" type="button" onClick={() => setHelpOpen(false)} aria-label="Close">
                ✕
              </button>
            </div>

            <div className="mdmDialog__body">
              <div style={{ padding: 18, display: "grid", gap: 10 }}>
                <div>
                  <div className="mdmLabel">Setup</div>
                  <div className="mdmTiny">Use <b>Start setup</b> to run the 3-step wizard.</div>
                </div>
                <div>
                  <div className="mdmLabel">Models</div>
                  <div className="mdmTiny">Use the <b>document</b> button in the header to open model-related screens.</div>
                </div>
                <div>
                  <div className="mdmLabel">Account</div>
                  <div className="mdmTiny">Use the <b>Account</b> menu for Login/Logout.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}


      <MdmModelsOverlay
        open={modelsOpen}
        onClose={() => setModelsOpen(false)}
        currentUser={currentUser}
        onRequireLogin={() => setLoginOpen(true)}
        onOpenModel={openWizardForModel}
      />


      {/* Wizard lives here */}
      <MdmWizard open={wizardOpen} onClose={closeWizard} />
      <Login open={loginOpen} onClose={() => setLoginOpen(false)} onLogin={handleLogin} />

    </div>
  );
}


