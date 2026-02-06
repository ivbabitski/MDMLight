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


async function fetchMatchingSummary(appUserId, modelId) {
  const base = String(API_BASE || "").trim();
  const mid = String(modelId || "").trim();
  const url = `${base}/api/matching/summary?model_id=${encodeURIComponent(mid)}&t=${Date.now()}`;

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
    throw new Error(`matching summary failed (HTTP ${res.status}). ${txt.slice(0, 120)}`);
  }

  const ct = String(res.headers.get("content-type") || "");
  if (!ct.toLowerCase().includes("application/json")) {
    const txt = await res.text().catch(() => "");
    throw new Error(
      `matching summary expected JSON but got "${ct || "unknown"}". ` +
        `URL="${url}". First bytes: ${txt.slice(0, 120)}`
    );
  }

  return res.json();
}


async function fetchReconClusterRecords(appUserId, modelId, status) {
  const base = String(API_BASE || "").trim();
  const mid = String(modelId || "").trim();
  const st = String(status || "match").trim();
  const url =
    `${base}/api/recon-cluster/records?model_id=${encodeURIComponent(mid)}` +
    `&status=${encodeURIComponent(st)}` +
    `&t=${Date.now()}`;

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
    throw new Error(`recon-cluster records failed (HTTP ${res.status}). ${txt.slice(0, 120)}`);
  }

  const ct = String(res.headers.get("content-type") || "");
  if (!ct.toLowerCase().includes("application/json")) {
    const txt = await res.text().catch(() => "");
    throw new Error(
      `recon-cluster records expected JSON but got "${ct || "unknown"}". ` +
        `URL="${url}". First bytes: ${txt.slice(0, 120)}`
    );
  }

  return res.json();
}


async function fetchGoldenRecordRecords(appUserId, modelId) {
  const base = String(API_BASE || "").trim();
  const mid = String(modelId || "").trim();
  const url = `${base}/api/golden-record/records?model_id=${encodeURIComponent(mid)}&t=${Date.now()}`;

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
    throw new Error(`golden-record records failed (HTTP ${res.status}). ${txt.slice(0, 120)}`);
  }

  const ct = String(res.headers.get("content-type") || "");
  if (!ct.toLowerCase().includes("application/json")) {
    const txt = await res.text().catch(() => "");
    throw new Error(
      `golden-record records expected JSON but got "${ct || "unknown"}". ` +
        `URL="${url}". First bytes: ${txt.slice(0, 120)}`
    );
  }

  return res.json();
}


async function cleanupReconCluster(appUserId, modelId) {
  const base = String(API_BASE || "").trim();
  const mid = String(modelId || "").trim();
  const url = `${base}/api/cleanup/recon-cluster?t=${Date.now()}`;

  const userId = String(appUserId || "").trim();

  const headers = {
    Accept: "application/json",
    "Content-Type": "application/json",
  };
  if (userId) headers["X-User-Id"] = userId;

  const res = await fetch(url, {
    method: "DELETE",
    cache: "no-store",
    headers,
    body: JSON.stringify({ mdm_model_id: mid }),
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`cleanup recon-cluster failed (HTTP ${res.status}). ${txt.slice(0, 120)}`);
  }

  const ct = String(res.headers.get("content-type") || "");
  if (!ct.toLowerCase().includes("application/json")) {
    const txt = await res.text().catch(() => "");
    throw new Error(
      `cleanup recon-cluster expected JSON but got "${ct || "unknown"}". ` +
        `URL="${url}". First bytes: ${txt.slice(0, 120)}`
    );
  }

  return res.json();
}


async function cleanupGoldenRecord(appUserId, modelId) {
  const base = String(API_BASE || "").trim();
  const mid = String(modelId || "").trim();
  const url = `${base}/api/cleanup/golden-record?t=${Date.now()}`;

  const userId = String(appUserId || "").trim();

  const headers = {
    Accept: "application/json",
    "Content-Type": "application/json",
  };
  if (userId) headers["X-User-Id"] = userId;

  const res = await fetch(url, {
    method: "DELETE",
    cache: "no-store",
    headers,
    body: JSON.stringify({ mdm_model_id: mid }),
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`cleanup golden-record failed (HTTP ${res.status}). ${txt.slice(0, 120)}`);
  }

  const ct = String(res.headers.get("content-type") || "");
  if (!ct.toLowerCase().includes("application/json")) {
    const txt = await res.text().catch(() => "");
    throw new Error(
      `cleanup golden-record expected JSON but got "${ct || "unknown"}". ` +
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

  const [selectedModelId, setSelectedModelId] = useState("");

  const [sourceSummary, setSourceSummary] = useState(null);
  const [sourceSummaryErr, setSourceSummaryErr] = useState("");

  const [matchingSummary, setMatchingSummary] = useState(null);
  const [matchingSummaryErr, setMatchingSummaryErr] = useState("");

  const [matchActionsOpen, setMatchActionsOpen] = useState(false);
  const matchActionsRef = useRef(null);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmKind, setConfirmKind] = useState(""); // "recon" | "golden"
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [confirmErr, setConfirmErr] = useState("");

  const [cleanupNotice, setCleanupNotice] = useState("");

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

    setSelectedModelId(modelId);

    setSourceSummaryErr("");
    try {
      const data = await fetchSourceInputSummary(userId, modelId);
      setSourceSummary(data);
    } catch (e) {
      setSourceSummary(null);
      setSourceSummaryErr(String(e?.message || e));
    }
  }, [currentUserId]);

  const refreshMatchingSummary = useCallback(async () => {
    const userId = String(currentUserId || "").trim();
    if (!userId) {
      setMatchingSummary(null);
      setMatchingSummaryErr("");
      return;
    }

    let modelId = "";
    try {
      modelId = String(localStorage.getItem(LS_SELECTED_MODEL_ID) || "").trim();
    } catch {}

    if (!modelId) {
      setMatchingSummary(null);
      setMatchingSummaryErr("model_id is required (select a model)");
      return;
    }

    setSelectedModelId(modelId);

    setMatchingSummaryErr("");
    try {
      const data = await fetchMatchingSummary(userId, modelId);
      setMatchingSummary(data);
    } catch (e) {
      setMatchingSummary(null);
      setMatchingSummaryErr(String(e?.message || e));
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
    function onMouseDown(e) {
      if (!matchActionsOpen) return;
      if (matchActionsRef.current && !matchActionsRef.current.contains(e.target)) {
        setMatchActionsOpen(false);
      }
    }

    function onKeyDown(e) {
      if (!matchActionsOpen) return;
      if (e.key === "Escape") setMatchActionsOpen(false);
    }

    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [matchActionsOpen]);


  useEffect(() => {
    refreshSourceSummary();
  }, [refreshSourceSummary]);

  useEffect(() => {
    refreshMatchingSummary();
  }, [refreshMatchingSummary]);

  useEffect(() => {
    function onUpdated() {
      refreshSourceSummary();
    }

    window.addEventListener("mdm:source_input_updated", onUpdated);

    return () => window.removeEventListener("mdm:source_input_updated", onUpdated);
  }, [refreshSourceSummary]);

  useEffect(() => {
    function onSelectedModelChanged() {
      setCleanupNotice("");
      refreshSourceSummary();
      refreshMatchingSummary();
    }

    window.addEventListener("mdm:selected_model_changed", onSelectedModelChanged);

    return () => window.removeEventListener("mdm:selected_model_changed", onSelectedModelChanged);
  }, [refreshMatchingSummary, refreshSourceSummary]);



  // NEW (UI only): dashboard + records review
  const [view, setView] = useState("golden"); // golden | match | exceptions
  const [q, setQ] = useState("");
  const [masterIdFilter, setMasterIdFilter] = useState("");
  const [pageSize, setPageSize] = useState(25);
  const [page, setPage] = useState(0);
  const [modelOpen, setModelOpen] = useState(false);

  const [records, setRecords] = useState([]);
  const [recordsErr, setRecordsErr] = useState("");
  const [recordsBusy, setRecordsBusy] = useState(false);

  const [promoted, setPromoted] = useState({}); // uiKey -> boolean (exceptions view)

  const job = MOCK_JOB;


  function normalizeUiFieldsFromDbRow(row) {
    const out = {};

    for (let i = 1; i <= 20; i += 1) {
      const dbKey = `f${String(i).padStart(2, "0")}`;
      const uiKey = `f${i}`;
      const v = row && row[dbKey] != null ? row[dbKey] : "";
      out[uiKey] = String(v ?? "");
    }

    return out;
  }


  function toReconUiRow(row) {
    const clusterId = row?.cluster_id != null ? String(row.cluster_id) : "";
    const sourceName = row?.source_name != null ? String(row.source_name) : "";
    const sourceId = row?.source_id != null ? String(row.source_id) : "";

    const uiFields = normalizeUiFieldsFromDbRow(row);

    return {
      ...row,
      uiKey: row?.id != null ? String(row.id) : `${clusterId}::${sourceName}::${sourceId}`,

      matching_model: String(row?.model_name || row?.model_id || ""),
      master_id: clusterId,
      match_threshold: "",
      survivorship_strategy: "",

      cluster_id: clusterId,

      created_at: row?.created_at != null ? String(row.created_at) : "",
      created_by: row?.created_by != null ? String(row.created_by) : "",
      updated_at: row?.updated_at != null ? String(row.updated_at) : "",
      updated_by: row?.updated_by != null ? String(row.updated_by) : "",

      ...uiFields,
    };
  }


  function toGoldenUiRow(row) {
    const masterId = row?.master_id != null ? String(row.master_id) : "";
    const uiFields = normalizeUiFieldsFromDbRow(row);

    return {
      ...row,
      uiKey: masterId,

      matching_model: String(row?.model_id || ""),
      master_id: masterId,
      match_threshold: row?.match_threshold ?? "",
      survivorship_strategy: String(row?.survivorship_json || ""),

      cluster_id: masterId,

      created_at: row?.created_at != null ? String(row.created_at) : "",
      created_by: row?.created_by != null ? String(row.created_by) : "",
      updated_at: row?.updated_at != null ? String(row.updated_at) : "",
      updated_by: row?.updated_by != null ? String(row.updated_by) : "",

      ...uiFields,
    };
  }


  const refreshRecords = useCallback(async () => {
    const userId = String(currentUserId || "").trim();
    if (!userId) {
      setRecords([]);
      setRecordsErr("");
      setRecordsBusy(false);
      return;
    }

    let modelId = "";
    try {
      modelId = String(localStorage.getItem(LS_SELECTED_MODEL_ID) || "").trim();
    } catch {}

    if (!modelId) {
      setRecords([]);
      setRecordsErr("model_id is required (select a model)");
      setRecordsBusy(false);
      return;
    }

    if (modelId !== String(selectedModelId || "").trim()) {
      setSelectedModelId(modelId);
    }

    setRecordsBusy(true);
    setRecordsErr("");

    try {
      if (view === "golden") {
        const data = await fetchGoldenRecordRecords(userId, modelId);
        const recs = Array.isArray(data?.records) ? data.records : [];
        setRecords(recs.map(toGoldenUiRow));
      } else {
        const status = view === "exceptions" ? "exceptions" : "match";
        const data = await fetchReconClusterRecords(userId, modelId, status);
        const recs = Array.isArray(data?.records) ? data.records : [];
        setRecords(recs.map(toReconUiRow));
      }
    } catch (e) {
      setRecords([]);
      setRecordsErr(String(e?.message || e));
    } finally {
      setRecordsBusy(false);
    }
  }, [currentUserId, selectedModelId, view]);


  useEffect(() => {
    refreshRecords();
  }, [refreshRecords]);


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


  const matchingFields = useMemo(() => {
    const arr = Array.isArray(matchingSummary?.matching_fields) ? matchingSummary.matching_fields : [];
    return arr.filter((x) => x && typeof x === "object");
  }, [matchingSummary]);



  const matchKeys = useMemo(() => job?.model_json?.match_fields || [], [job]);

  const survivorshipStrategy = useMemo(
    () => job?.model_json?.survivorship_strategy || "Strategy 0",
    [job]
  );


  const rows = useMemo(() => {
    if (!Array.isArray(records)) return [];
    return records;
  }, [records]);


  const recordsPerSource = useMemo(() => {
    if (Array.isArray(sourceSummary?.sources)) return sourceSummary.sources;
    return [];
  }, [sourceSummary]);


  const maxPerSource = useMemo(() => Math.max(...recordsPerSource.map((x) => x.count), 1), [recordsPerSource]);

  const totalClusters = useMemo(() => new Set(rows.map((r) => r.cluster_id)).size, [rows]);

  const listRows = useMemo(() => {
    const query = q.trim().toLowerCase();
    const master = masterIdFilter.trim().toLowerCase();

    const base = rows;

    const narrowed = master
      ? base.filter((r) => String(r.master_id || "").toLowerCase().includes(master))
      : base;

    if (!query) return narrowed;

    return narrowed.filter((r) => {
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
  }, [masterIdFilter, q, rows]);


  useEffect(() => {
    setPage(0);
  }, [masterIdFilter, q, view, pageSize]);

  const pageCount = useMemo(() => {
    const total = listRows.length;
    return total ? Math.ceil(total / pageSize) : 1;
  }, [listRows.length, pageSize]);

  useEffect(() => {
    setPage((p) => Math.min(p, Math.max(0, pageCount - 1)));
  }, [pageCount]);

  const pagedRows = useMemo(() => {
    const start = page * pageSize;
    return listRows.slice(start, start + pageSize);
  }, [listRows, page, pageSize]);

  const pageStart = useMemo(() => {
    if (listRows.length === 0) return 0;
    return page * pageSize + 1;
  }, [listRows.length, page, pageSize]);

  const pageEnd = useMemo(() => {
    if (listRows.length === 0) return 0;
    return Math.min(listRows.length, (page + 1) * pageSize);
  }, [listRows.length, page, pageSize]);


  function csvEscapeCell(v) {
    const s = String(v ?? "");
    const needsQuotes = /[",\n\r]/.test(s);
    const escaped = s.replace(/"/g, '""');
    return needsQuotes ? `"${escaped}"` : escaped;
  }

  function downloadCsv() {
    const cols = [
      "matching_model",
      "master_id",
      "match_threshold",
      "survivorship_strategy",
      ...USER_FIELD_KEYS,
      "created_at",
      "created_by",
      "updated_at",
      "updated_by",
    ];

    const header = cols.join(",");
    const lines = listRows.map((r) => cols.map((c) => csvEscapeCell(r[c])).join(","));
    const csv = [header, ...lines].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `mdm_${view}_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
  }



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


  async function runCleanup(kind) {
    const userId = String(currentUserId || "").trim();
    if (!userId) {
      setConfirmErr("X-User-Id is required (login)");
      return;
    }

    let modelId = "";
    try {
      modelId = String(localStorage.getItem(LS_SELECTED_MODEL_ID) || "").trim();
    } catch {}

    if (!modelId) {
      setConfirmErr("model_id is required (select a model)");
      return;
    }

    if (kind !== "recon" && kind !== "golden") {
      setConfirmErr("Invalid action");
      return;
    }

    setConfirmBusy(true);
    setConfirmErr("");

    try {
      if (kind === "recon") {
        await cleanupReconCluster(userId, modelId);
      } else {
        await cleanupGoldenRecord(userId, modelId);
      }

      setConfirmOpen(false);
      setConfirmKind("");
      setMatchActionsOpen(false);

      await refreshMatchingSummary();
      await refreshRecords();
    } catch (e) {
      setConfirmErr(String(e?.message || e));
    } finally {
      setConfirmBusy(false);
    }
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
    refreshMatchingSummary();
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
                  <div className="mdmCard__sub">
                    {matchingSummaryErr
                      ? `Matching summary error: ${matchingSummaryErr}`
                      : `Model: ${String(matchingSummary?.model_name || "—")} (id: ${String(selectedModelId || "—")})`}
                  </div>
                </div>

                <div
                  ref={matchActionsRef}
                  onMouseLeave={() => setMatchActionsOpen(false)}
                  style={{
                    position: "relative",
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    justifyContent: "center",
                  }}
                >
                  <button
                    className="mdmBtn mdmBtn--xs mdmBtn--soft mdmIconBtn"
                    type="button"
                    onClick={refreshMatchingSummary}
                    title="Refresh matching stats"
                    aria-label="Refresh matching stats"
                  >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M3 12a9 9 0 0 1 15-6.7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M3 4v6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M21 12a9 9 0 0 1-15 6.7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M21 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  </button>

                  <button
                    className="mdmBtn mdmBtn--xs mdmBtn--soft mdmIconBtn"
                    type="button"
                    onClick={() => {
                      setAccountMenuOpen(false);
                      setMatchActionsOpen((v) => !v);
                    }}
                    title="Actions"
                    aria-label="Actions"
                  >
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                      <circle cx="12" cy="5" r="1.8" fill="currentColor" />
                      <circle cx="12" cy="12" r="1.8" fill="currentColor" />
                      <circle cx="12" cy="19" r="1.8" fill="currentColor" />
                    </svg>
                  </button>

                  {matchActionsOpen ? (
                    <div
                      style={{
                        position: "absolute",
                        top: "calc(100% + 10px)",
                        right: 0,
                        minWidth: 270,
                        background: "rgba(255,255,255,0.98)",
                        border: "1px solid rgba(0,0,0,0.12)",
                        borderRadius: 12,
                        padding: 6,
                        boxShadow: "0 14px 40px rgba(0,0,0,0.25)",
                        zIndex: 60,
                        color: "#000",
                        display: "flex",
                        flexDirection: "column",
                        gap: 4,
                        boxSizing: "border-box",
                      }}
                    >
                      <button
                        type="button"
                        title="Clear matches"
                        disabled={!currentUserId || !selectedModelId || Number(matchingSummary?.match_clusters || 0) <= 0}
                        onClick={() => {
                          setMatchActionsOpen(false);
                          setConfirmErr("");
                          setConfirmKind("recon");
                          setConfirmOpen(true);
                        }}
                        style={{
                          width: "100%",
                          textAlign: "left",
                          padding: "8px 10px",
                          borderRadius: 10,
                          border: 0,
                          background: "transparent",
                          cursor: "pointer",
                          color: "#000",
                          fontWeight: 800,
                          fontSize: 13,
                          lineHeight: "16px",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          opacity: (!currentUserId || !selectedModelId || Number(matchingSummary?.match_clusters || 0) <= 0) ? 0.45 : 1,
                        }}
                      >
                        Clear matches
                      </button>

                      <button
                        type="button"
                        title="Clear golden records"
                        disabled={!currentUserId || !selectedModelId || Number(matchingSummary?.golden_records || 0) <= 0}
                        onClick={() => {
                          setMatchActionsOpen(false);
                          setConfirmErr("");
                          setConfirmKind("golden");
                          setConfirmOpen(true);
                        }}
                        style={{
                          width: "100%",
                          textAlign: "left",
                          padding: "8px 10px",
                          borderRadius: 10,
                          border: 0,
                          background: "transparent",
                          cursor: "pointer",
                          color: "#000",
                          fontWeight: 800,
                          fontSize: 13,
                          lineHeight: "16px",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          opacity: (!currentUserId || !selectedModelId || Number(matchingSummary?.golden_records || 0) <= 0) ? 0.45 : 1,
                        }}
                      >
                        Clear golden records
                      </button>
                    </div>
                  ) : null}

                </div>
              </div>

              <div className="mdmCard__body">
                <div className="mdmKpiRow">
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(Number(matchingSummary?.golden_records || 0))}</div>
                    <div className="mdmKpiLab">Golden records</div>
                  </div>
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(Number(matchingSummary?.exceptions || 0))}</div>
                    <div className="mdmKpiLab">Exceptions</div>
                  </div>
                  <div className="mdmKpi">
                    <div className="mdmKpiNum">{fmtInt(Number(matchingSummary?.match_clusters || 0))}</div>
                    <div className="mdmKpiLab">Match clusters</div>
                  </div>
                </div>

                <div className="mdmDivider" />

                <div className="mdmSectionTitle">Matching fields</div>

                {matchingFields.length ? (
                  <div className="mdmBarList">
                    {matchingFields.map((f) => {
                      const name = String(f?.label || f?.code || "").trim() || "—";

                      const wRaw = Number(f?.weight);
                      const weightStr = Number.isFinite(wRaw) ? wRaw.toFixed(2) : "—";

                      let pctNum = Number(f?.weight_pct);
                      if (!Number.isFinite(pctNum)) {
                        pctNum = Number.isFinite(wRaw) ? (wRaw * 100.0) : 0;
                      }
                      const pct = Math.max(0, Math.min(100, Math.round(pctNum)));

                      return (
                        <div
                          className="mdmBarRow"
                          key={String(f?.code || name)}
                          style={{
                            gridTemplateColumns: "minmax(0, max-content) minmax(140px, 1fr) auto",
                            alignItems: "center",
                          }}
                        >
                          <div
                            className="mdmBarName"
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 10,
                              minWidth: 0,
                            }}
                          >
                            <span
                              style={{
                                flex: "1 1 auto",
                                minWidth: 0,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                              }}
                            >
                              {name}
                            </span>

                            <span
                              className="mdmTiny"
                              style={{
                                flex: "0 0 auto",
                                whiteSpace: "nowrap",
                              }}
                            >
                              w: {weightStr}
                            </span>
                          </div>

                          <div className="mdmBarTrack" aria-hidden="true">
                            <span className="mdmBarFill" style={{ width: `${pct}%` }} />
                          </div>

                          <div className="mdmBarVal">{fmtInt(pct)}%</div>
                        </div>
                      );


                    })}
                  </div>
                ) : (
                  <div className="mdmPillRow">
                    <span className="mdmPillSoft">—</span>
                  </div>
                )}

                <div className="mdmDivider" />

                <div className="mdmSectionTitle">Survivorship strategy</div>
                <div className="mdmPillRow">
                  <span className="mdmPillSoft mdmPillSoft--strong">{String(matchingSummary?.survivorship_label || "—")}</span>
                </div>

              </div>
            </div>
          </div>

          {/* NEW: Records table */}
          <div className="mdmCard">
            <div className="mdmCard__head mdmRecordsHead">
              <div>
                <div className="mdmCard__title">Records</div>
                <div className="mdmCard__sub">Golden / Match / Exceptions (table)</div>
              </div>

              <div className="mdmInputWithIcon mdmRecordsSearch">
                <input
                  className="mdmInput mdmInput--withIcon"
                  type="text"
                  value={masterIdFilter}
                  onChange={(e) => setMasterIdFilter(e.target.value)}
                  placeholder="Filter master_id"
                  aria-label="Filter by master_id"
                />
                {masterIdFilter ? (
                  <button
                    type="button"
                    className="mdmInputIconBtn"
                    onClick={() => setMasterIdFilter("")}
                    title="Clear master_id filter"
                    aria-label="Clear master_id filter"
                  >
                    ✕
                  </button>
                ) : null}
              </div>

              <div className="mdmBtnGroup mdmRecordsActions">
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
                    className={view === "match" ? "isActive" : ""}
                    onClick={() => { setView("match"); setQ(""); }}
                    role="tab"
                    aria-selected={view === "match"}
                  >
                    Match
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

                <button
                  className="mdmBtn mdmBtn--xs mdmBtn--soft mdmRecordsDownloadBtn"
                  type="button"
                  onClick={downloadCsv}
                  title="Download CSV"
                  aria-label="Download CSV"
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M12 3v10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M8 11l4 4 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M4 17v4h16v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
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
                    {pagedRows.map((r) => {
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
                                  className={`mdmBtn mdmBtn--xs mdmIconBtn ${isPromoted ? "mdmBtn--soft" : "mdmBtn--gold"}`}
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
                                  className="mdmBtn mdmBtn--xs mdmBtn--run mdmIconBtn"
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

              <div className="mdmRecordsFooter" role="navigation" aria-label="Records pagination">
                <div className="mdmRecordsFooterLeft">
                  <span className="mdmTiny">Rows per page</span>
                  <select
                    className="mdmSelect mdmSelect--xs"
                    value={pageSize}
                    onChange={(e) => setPageSize(Number(e.target.value))}
                    aria-label="Rows per page"
                  >
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                  </select>
                </div>

                <div className="mdmRecordsFooterMid">
                  <span className="mdmTiny">
                    Showing {fmtInt(pageStart)}–{fmtInt(pageEnd)} of {fmtInt(listRows.length)}
                  </span>
                </div>

                <div className="mdmRecordsFooterRight">
                  <button
                    className="mdmBtn mdmBtn--xs mdmBtn--soft"
                    type="button"
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    aria-label="Previous page"
                    title="Previous page"
                  >
                    ‹
                  </button>

                  <span className="mdmTag mdmTag--soft">{fmtInt(page + 1)} / {fmtInt(pageCount)}</span>

                  <button
                    className="mdmBtn mdmBtn--xs mdmBtn--soft"
                    type="button"
                    onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                    disabled={page >= pageCount - 1}
                    aria-label="Next page"
                    title="Next page"
                  >
                    ›
                  </button>
                </div>
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


      {confirmOpen ? (
        <div className="mdmOverlay" role="dialog" aria-modal="true" aria-label="Confirm cleanup">
          <div className="mdmDialog" style={{ width: 460 }}>
            <div className="mdmDialog__head">
              <div>
                <div className="mdmDialog__title">
                  {confirmKind === "recon" ? "Clear matches?" : "Clear golden records?"}
                </div>
                <div className="mdmDialog__sub">
                  {confirmKind === "recon"
                    ? "This deletes recon_cluster rows for the selected model and user."
                    : "This deletes golden_record rows for the selected model and user."}
                </div>
              </div>
              <button
                className="mdmX"
                type="button"
                onClick={() => {
                  if (!confirmBusy) setConfirmOpen(false);
                }}
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            <div className="mdmDialog__body">
              <div style={{ padding: 18, display: "grid", gap: 12 }}>
                <div className="mdmTiny">
                  Model id: <span className="mdmMono">{String(selectedModelId || "—")}</span>
                </div>

                {confirmErr ? (
                  <div className="mdmTiny" style={{ color: "rgba(220,38,38,0.95)", fontWeight: 900 }}>
                    {confirmErr}
                  </div>
                ) : null}

                <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
                  <button
                    className="mdmBtn mdmBtn--soft"
                    type="button"
                    onClick={() => setConfirmOpen(false)}
                    disabled={confirmBusy}
                  >
                    Cancel
                  </button>

                  <button
                    className="mdmBtn mdmBtn--danger"
                    type="button"
                    onClick={() => runCleanup(confirmKind)}
                    disabled={confirmBusy}
                  >
                    {confirmBusy ? "Clearing..." : "Clear"}
                  </button>
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


