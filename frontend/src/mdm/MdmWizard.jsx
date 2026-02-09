"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { getUserId } from "../authStorage";

const LS_SELECTED_MODEL_ID = "mdm_selected_model_id";



const TYPE_OPTIONS = [
  { value: "email", label: "Email" },
  { value: "first_name", label: "First Name" },
  { value: "last_name", label: "Last Name" },
  { value: "business_name", label: "Business Name" },
  { value: "address_line", label: "Address Line" },
  { value: "text", label: "Text" },
  { value: "number", label: "Number" },
  { value: "date", label: "Date" },
  { value: "phone", label: "Phone" },
  { value: "tax_id", label: "Tax ID" },
  { value: "id", label: "ID" },
];

const SURVIVORSHIP_RULES = [
  { value: "recency_created_date", label: "Recency (most recently created)" },
  { value: "recency_updated_date", label: "Recency (most recently updated)" },
  { value: "first_created_date", label: "First Created" },
  { value: "first_updated_date", label: "First Updated" },
  { value: "system", label: "By System Priority" },
  { value: "specific_value_priority", label: "By Specific Value" },
];


const DEFAULT_FIELDS = [
  { code: "source_name", label: "source_name", kind: "required", include: true, key: false, type: "text", weight: 0, rule: "recency_updated_date" },
  { code: "source_id", label: "source_id", kind: "required", include: true, key: true, type: "id", weight: 0, rule: "recency_updated_date" },

  { code: "f01", label: "", kind: "flex", include: false, key: false, type: "text", weight: 0.25, rule: "recency_updated_date" },
  { code: "f02", label: "", kind: "flex", include: false, key: false, type: "text", weight: 0.25, rule: "recency_updated_date" },
  { code: "f03", label: "", kind: "flex", include: false, key: false, type: "text", weight: 0.25, rule: "recency_updated_date" },
  { code: "f04", label: "", kind: "flex", include: false, key: false, type: "text", weight: 0.25, rule: "recency_updated_date" },
  { code: "f05", label: "", kind: "flex", include: false, key: false, type: "text", weight: 0.25, rule: "recency_updated_date" },

  { code: "created_at", label: "created_at", kind: "reserved", include: false, key: false, type: "date", weight: 0, rule: "recency_updated_date" },
  { code: "created_by", label: "created_by", kind: "reserved", include: false, key: false, type: "text", weight: 0, rule: "recency_updated_date" },
  { code: "updated_at", label: "updated_at", kind: "reserved", include: false, key: false, type: "date", weight: 0, rule: "recency_updated_date" },
  { code: "updated_by", label: "updated_by", kind: "reserved", include: false, key: false, type: "text", weight: 0, rule: "recency_updated_date" },
];

function uid(prefix = "id") {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function pct01(n) {
  return `${Math.round(clamp(n, 0, 1) * 100)}%`;
}

function guessType(fieldName) {
  const x = fieldName.toLowerCase();
  if (x.includes("email")) return "email";
  if (x.includes("phone") || x.includes("mobile")) return "phone";
  if (x.includes("date") || x.includes("_at")) return "date";
  if (x.includes("id")) return "id";
  if (x.includes("zip") || x.includes("postal")) return "number";
  return "text";
}

function randomToken(bytes = 16) {
  const buf = new Uint8Array(bytes);
  if (typeof crypto !== "undefined" && crypto.getRandomValues) crypto.getRandomValues(buf);
  else for (let i = 0; i < bytes; i++) buf[i] = Math.floor(Math.random() * 256);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

function wheelToScrollContainer(e) {
  const el = e.currentTarget;
  if (!el) return;

  const max = el.scrollHeight - el.clientHeight;
  if (max <= 0) return;

  const delta = e.deltaY;
  if (!delta) return;

  const t = e.target;
  const tag = t?.tagName;
  const type = tag === "INPUT" ? t.type : "";

  const shouldHijack =
    tag === "SELECT" || (tag === "INPUT" && (type === "number" || type === "range"));

  if (!shouldHijack) return;

  const top = el.scrollTop;
  const atTop = top <= 0;
  const atBottom = top >= max - 1;

  if ((delta < 0 && atTop) || (delta > 0 && atBottom)) return;

  if (t && typeof t.blur === "function") t.blur();
  el.scrollTop = clamp(top + delta, 0, max);
}



function StepButton({ active, done, num, title, hint, onClick }) {
  return (
    <button
      className={`mdmStep ${active ? "mdmStep--active" : ""} ${done ? "mdmStep--done" : ""}`}
      type="button"
      onClick={onClick}
    >
      <div className="mdmStep__num">{num}</div>
      <div className="mdmStep__meta">
        <div className="mdmStep__title">{title}</div>
        <div className="mdmStep__hint">{hint}</div>
      </div>
    </button>
  );
}

function reorderArray(list, from, to) {
  if (from === to) return list;
  const next = [...list];
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next;
}

function setUniqueValue(list, idx, raw) {
  const next = [...list];
  const v = String(raw ?? "").trim();

  if (v) {
    for (let i = 0; i < next.length; i++) {
      if (i === idx) continue;
      if (String(next[i] ?? "").trim() === v) next[i] = "";
    }
  }

  next[idx] = v;
  return next;
}

function PriorityPickerList({
  kind,
  items,
  options,
  placeholder,
  cap,
  onPick,
  onAdd,
  onRemove,
  onReorder,
}) {
  const dragFromRef = useRef(null);

  function onDragStart(idx) {
    dragFromRef.current = idx;
  }

  function onDrop(idx) {
    const from = dragFromRef.current;
    dragFromRef.current = null;
    if (from === null || from === undefined) return;
    if (from === idx) return;
    onReorder(from, idx);
  }

  function onDragEnd() {
    dragFromRef.current = null;
  }

  return (
    <>
      <div className="mdmPriorityList" onWheelCapture={wheelToScrollContainer}>
        {items.map((value, idx) => {
          const current = String(value ?? "").trim();

          const baseOptionsRaw =
            (Array.isArray(options) && options.length > 0)
              ? options
              : items.map((x) => String(x ?? "").trim()).filter(Boolean);

          const baseOptions = [];
          for (const o of baseOptionsRaw) {
            const v = String(o ?? "").trim();
            if (!v) continue;
            if (!baseOptions.includes(v)) baseOptions.push(v);
          }

          const used = new Set(
            items
              .map((x, i) => (i === idx ? "" : String(x ?? "").trim()))
              .filter(Boolean)
          );

          const rowOptions = baseOptions.filter((o) => !used.has(o));
          if (current && !rowOptions.includes(current)) rowOptions.unshift(current);

          return (
            <div
              key={`${kind}-${idx}`}
              className="mdmPriorityRow"
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => onDrop(idx)}
            >
              <div className="mdmPriorityLeft">
                <div
                  className="mdmDragHandle"
                  draggable
                  onDragStart={() => onDragStart(idx)}
                  onDragEnd={onDragEnd}
                  title="Drag to reorder"
                  aria-label="Drag to reorder"
                >
                  <svg viewBox="0 0 12 12" width="14" height="14" aria-hidden="true">
                    <circle cx="3" cy="2" r="1" fill="currentColor" />
                    <circle cx="9" cy="2" r="1" fill="currentColor" />
                    <circle cx="3" cy="6" r="1" fill="currentColor" />
                    <circle cx="9" cy="6" r="1" fill="currentColor" />
                    <circle cx="3" cy="10" r="1" fill="currentColor" />
                    <circle cx="9" cy="10" r="1" fill="currentColor" />
                  </svg>
                </div>

                <div className="mdmPriorityName">
                  <div className="mdmPriorityPickWrap">
                    <select
                      className="mdmSelect mdmPriorityPickSelect mdmMono"
                      value={String(value ?? "")}
                      onChange={(e) => onPick(idx, e.target.value)}
                    >
                      <option value="">{placeholder}</option>
                      {rowOptions.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div className="mdmPriorityActions">
                <button
                  className="mdmBtn mdmIconBtn mdmBtn--ghost"
                  type="button"
                  onClick={() => onRemove(idx)}
                  disabled={items.length <= 1}
                  aria-label="Remove"
                  title="Remove"
                >
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true">
                    <path d="M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {kind !== "system" && (
        <div className="mdmPriorityAddRow">
          <button
            className="mdmBtn mdmBtn--soft mdmBtn--small"
            type="button"
            onClick={onAdd}
            disabled={items.length >= cap}
            title={items.length >= cap ? `Max ${cap}` : "Add"}
          >
            Add {kind === "user" ? "user" : "system"}
          </button>
        </div>
      )}
    </>
  );
}

export default function MdmWizard({ open, onClose, actor: actorProp, apiBaseUrl = "", onSaved }) {
  const fileRef = useRef(null);

  // reset wizard state whenever it opens (Start Setup should be blank)
  useEffect(() => {
    if (!open) return;

    setStep(0);
    setConfigured(false);

    setAiExceptions(false);

    setDomainModelName("");
    setEditingModelId(null);

    setSourceType("csv");
    setCsvFileName("");
    setCsvFile(null);
    setCsvUploading(false);
    setCsvUploaded(false);
    setCsvUploadError("");
    setCsvUploadRows(0);
    setRecordCount(12500);

    setApiEndpoint("http://localhost:5000/api/ingest/api");
    setApiToken(randomToken(16));
    setCopied(null);
    setShowToken(false);

    setSavingModel(false);
    setSaveError("");
    setSaveSuccess(null);

    setFields(DEFAULT_FIELDS.map((f) => ({ ...f, id: uid("f") })));

    setMatchFieldCodes([]);
    setMatchFieldPick("");
    setRunModelAfterSave(false);
    setWeightDraftById({});
    setWeightsTouched(false);

    setMatchThreshold(0.85);
    setPossibleThreshold(0.7);

    setAdvanced(false);
    setGlobalRule("recency_updated_date");

    setSourceSystemsLoading(false);
    setSourceSystemsError("");
    setSourceSystemOptions([]);

    setSystemPriority([""]);
    setUserPriority([""]);

    setSpecificValuePriority([{ id: uid("sv"), fieldCode: "", value: "" }]);
    setSpecificValueError("");

    try {
      if (fileRef.current) fileRef.current.value = "";
    } catch {}
  }, [open]);


  useEffect(() => {
    if (!open) return;

    const prevHtml = document.documentElement.style.overflow;
    const prevBody = document.body.style.overflow;

    document.documentElement.style.overflow = "hidden";
    document.body.style.overflow = "hidden";

    return () => {
      document.documentElement.style.overflow = prevHtml;
      document.body.style.overflow = prevBody;
    };
  }, [open]);


  // Stepper + finish
  const [step, setStep] = useState(0);
  const [configured, setConfigured] = useState(false);

  // Ai assisted exceptions
  const [aiExceptions, setAiExceptions] = useState(false);

  // Model meta (label + stable id)
  const [domainModelName, setDomainModelName] = useState("");
  const [editingModelId, setEditingModelId] = useState(null); // string | null


  // Step 1 — source
  const [sourceType, setSourceType] = useState("csv"); // csv | api
  const [csvFileName, setCsvFileName] = useState("");
  const [csvFile, setCsvFile] = useState(null);
  const [csvUploading, setCsvUploading] = useState(false);
  const [csvUploaded, setCsvUploaded] = useState(false);
  const [csvUploadError, setCsvUploadError] = useState("");
  const [csvUploadRows, setCsvUploadRows] = useState(0);
  const [recordCount, setRecordCount] = useState(12500);



  // API key mode (inbound)
  const [apiEndpoint, setApiEndpoint] = useState("http://localhost:5000/api/ingest/api");
  const [apiToken, setApiToken] = useState(() => randomToken(16));
  const [copied, setCopied] = useState(null); // "endpoint" | "token" | null
  const [showToken, setShowToken] = useState(false);

  const [savingModel, setSavingModel] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [saveSuccess, setSaveSuccess] = useState(null); // { name, id } | null


  // Step 2 — fields
  const [fields, setFields] = useState(() =>
    DEFAULT_FIELDS.map((f) => ({ ...f, id: uid("f") }))
  );

  // Step 3 — match fields (explicit selection)
  const [matchFieldCodes, setMatchFieldCodes] = useState([]);
  const [matchFieldPick, setMatchFieldPick] = useState("");

  const [runModelAfterSave, setRunModelAfterSave] = useState(false);
  const [weightDraftById, setWeightDraftById] = useState(() => ({}));
  const [weightsTouched, setWeightsTouched] = useState(false);

  // Step 3 — model
  const [matchThreshold, setMatchThreshold] = useState(0.85);
  const [possibleThreshold, setPossibleThreshold] = useState(0.7);


  const [advanced, setAdvanced] = useState(false);
  const [globalRule, setGlobalRule] = useState("recency_updated_date");
  const PRIORITY_CAP = 15;

  const [sourceSystemsLoading, setSourceSystemsLoading] = useState(false);
  const [sourceSystemsError, setSourceSystemsError] = useState("");
  const [sourceSystemOptions, setSourceSystemOptions] = useState([]);

  const [systemPriority, setSystemPriority] = useState([""]);
  const [userPriority, setUserPriority] = useState([""]);



  const [specificValuePriority, setSpecificValuePriority] = useState(() => [
    { id: uid("sv"), fieldCode: "", value: "" },
  ]);


  const [specificValueError, setSpecificValueError] = useState("");
  const specificValueDragFromRef = useRef(null);
  const prevMatchFieldKeyRef = useRef("");

  // computed
  const includedFields = useMemo(() => fields.filter((f) => f.include), [fields]);
  const keyField = useMemo(() => fields.find((f) => f.key), [fields]);


  const eligibleMatchFields = useMemo(
    () =>
      fields.filter((f) => {
        if (f.kind !== "flex") return false;
        if (!f.include) return false;
        return Boolean((f.label ?? "").trim());
      }),
    [fields]
  );

  const matchFieldOptions = useMemo(
    () => eligibleMatchFields.filter((f) => !matchFieldCodes.includes(f.code)),
    [eligibleMatchFields, matchFieldCodes]
  );

  const matchFields = useMemo(() => {
    const byCode = new Map(fields.map((f) => [f.code, f]));
    return matchFieldCodes
      .map((code) => byCode.get(code))
      .filter((f) => f && f.kind === "flex" && f.include && Boolean((f.label ?? "").trim()));
  }, [fields, matchFieldCodes]);

  const matchFieldKey = useMemo(() => matchFields.map((f) => f.id).join("|"), [matchFields]);

  const totalWeightPct = useMemo(() => {
    return matchFields.reduce((sum, f) => sum + Math.round((Number(f.weight) || 0) * 100), 0);
  }, [matchFields]);

  function equalWeightPcts(count) {
    if (!count || count <= 0) return [];
    const base = Math.floor(100 / count);
    let rem = 100 - base * count;
    const out = Array.from({ length: count }, () => base);
    for (let i = 0; i < out.length && rem > 0; i++) {
      out[i] = out[i] + 1;
      rem--;
    }
    return out;
  }

  // auto equalize weights when match fields selection changes (until user edits weights)
  useEffect(() => {
    const keyChanged = prevMatchFieldKeyRef.current !== matchFieldKey;
    prevMatchFieldKeyRef.current = matchFieldKey;

    if (keyChanged) {
      setWeightDraftById({});
    }

    const ids = matchFields.map((f) => f.id);
    const n = ids.length;

    if (n === 0) {
      if (weightsTouched) setWeightsTouched(false);
      return;
    }

    // Once the user edits any weight, stop auto-rebalancing.
    if (weightsTouched) {
      if (n === 1) {
        const onlyId = ids[0];
        setFields((prev) =>
          prev.map((f) => (f.id === onlyId ? { ...f, weight: 1 } : f))
        );
      }
      return;
    }

    const targetPcts = equalWeightPcts(n);

    setFields((prev) => {
      const currentPcts = ids.map((id) => {
        const ff = prev.find((x) => x.id === id);
        const w = Number(ff?.weight) || 0;
        return Math.max(0, Math.min(100, Math.round(w * 100)));
      });

      const same =
        currentPcts.length === targetPcts.length &&
        currentPcts.every((p, i) => p === targetPcts[i]);

      const sum = currentPcts.reduce((a, b) => a + b, 0);

      if (same && sum === 100) return prev;

      return prev.map((f) => {
        const idx = ids.indexOf(f.id);
        if (idx === -1) return f;
        return { ...f, weight: targetPcts[idx] / 100 };
      });
    });
  }, [matchFieldKey, weightsTouched]);


  // keep matchFieldCodes valid (must remain: flex + included + labeled)
  useEffect(() => {
    setMatchFieldCodes((prev) => {
      const eligible = new Set(
        fields
          .filter((f) => f.kind === "flex" && f.include && Boolean((f.label ?? "").trim()))
          .map((f) => f.code)
      );

      const next = prev.filter((c) => eligible.has(c));

      const same =
        next.length === prev.length &&
        next.every((v, i) => v === prev[i]);

      return same ? prev : next;
    });

    setMatchFieldPick((prev) => {
      if (!prev) return prev;
      const ok = fields.some(
        (f) => f.code === prev && f.kind === "flex" && f.include && Boolean((f.label ?? "").trim())
      );
      return ok ? prev : "";
    });
  }, [fields]);

  const selectableFlex = useMemo(
    () =>
      fields.filter((f) => {
        if (f.kind !== "flex") return false;
        const hasLabel = Boolean((f.label ?? "").trim());
        return hasLabel || f.include;
      }),
    [fields]
  );

  const allFlexIncluded =
    selectableFlex.length > 0 && selectableFlex.every((f) => !!f.include);

  const someFlexIncluded = selectableFlex.some((f) => !!f.include);

  const sortedFields = useMemo(() => {
    const requiredOrder = ["source_id", "source_name"];
    const reservedOrder = ["created_at", "created_by", "updated_at", "updated_by"];

    const req = fields
      .filter((f) => f.kind === "required")
      .sort((a, b) => requiredOrder.indexOf(a.code) - requiredOrder.indexOf(b.code));

    const flex = fields
      .filter((f) => f.kind === "flex")
      .sort((a, b) => (a.code || "").localeCompare(b.code || ""));

    const res = fields
      .filter((f) => f.kind === "reserved")
      .sort((a, b) => reservedOrder.indexOf(a.code) - reservedOrder.indexOf(b.code));

    return [...req, ...flex, ...res];
  }, [fields]);

  const systemOptions = useMemo(() => {
    const fromApi = (Array.isArray(sourceSystemOptions) ? sourceSystemOptions : [])
      .map((s) => String(s ?? "").trim())
      .filter(Boolean);

    const fromPriority = (Array.isArray(systemPriority) ? systemPriority : [])
      .map((s) => String(s ?? "").trim())
      .filter(Boolean);

    return Array.from(new Set([...fromApi, ...fromPriority]));
  }, [sourceSystemOptions, systemPriority]);


  const userOptions = useMemo(
    () => Array.from(new Set(userPriority.map((u) => String(u ?? "").trim()).filter(Boolean))),
    [userPriority]
  );

  const modelNameTrimmed = String(domainModelName || "").trim();

  const modelNameOk =
    modelNameTrimmed.length > 0 && /^[A-Za-z0-9_-]+$/.test(modelNameTrimmed);

const sourceOk =
  sourceType === "csv"
    ? Boolean(csvFileName) && (csvUploaded || !!csvFile)
    : apiEndpoint.trim().length > 0 && apiToken.trim().length > 0;


  const step1Ok = modelNameOk && sourceOk;

  const step2Ok = Boolean(keyField) && eligibleMatchFields.length > 0;

  const step3Ok = matchFieldCodes.length > 0;
  const weightsOk = totalWeightPct === 100;



  const actor = useMemo(() => {
    const p = String(actorProp || "").trim();
    if (p) return p;
    if (typeof window !== "undefined") {
      const v = String(window.localStorage?.getItem("username") || "").trim();
      if (v) return v;
    }
    return "";
  }, [actorProp]);

const apiBase = useMemo(() => {
  const env = (import.meta.env.VITE_API_URL || "").trim();
  const raw = String(apiBaseUrl || env || "").trim();

  if (raw) return raw.endsWith("/") ? raw.slice(0, -1) : raw;

  if (typeof window === "undefined") return "";
  const proto = window.location.protocol || "http:";
  const host = window.location.hostname || "localhost";
  return `${proto}//${host}:5000`;
}, [apiBaseUrl]);


function applyLoadedModelConfig(modelId, cfg) {
  const safe = (v) => (v === null || v === undefined ? "" : String(v));

  setEditingModelId(String(modelId || "").trim() || null);

  setConfigured(false);
  setSaveSuccess(null);
  setSaveError("");

  setDomainModelName(safe(cfg?.domainModelName).trim());

  const st = String(cfg?.sourceType || "csv").trim().toLowerCase();
  setSourceType(st === "api" ? "api" : "csv");

  setCsvFileName(safe(cfg?.csvFileName));
  setCsvFile(null);
  setCsvUploading(false);
  setCsvUploaded(st !== "api" && Boolean(safe(cfg?.csvFileName)));
  setCsvUploadError("");
  setCsvUploadRows(0);

  setRecordCount(Number(cfg?.recordCount || 0) || 0);

  setApiEndpoint(safe(cfg?.apiEndpoint) || "http://localhost:5000/api/ingest/api");
  setApiToken(safe(cfg?.apiToken));

  setAiExceptions(Boolean(cfg?.aiExceptions));

  setMatchThreshold(Number(cfg?.matchThreshold || 0.85) || 0.85);
  setPossibleThreshold(Number(cfg?.possibleThreshold || 0.7) || 0.7);

  setAdvanced(Boolean(cfg?.advanced));
  setGlobalRule(safe(cfg?.globalRule) || "recency_updated_date");

  setSystemPriority(
    Array.isArray(cfg?.systemPriority)
      ? cfg.systemPriority.map((x) => safe(x).trim()).filter(Boolean)
      : [""]
  );
  setUserPriority(
    Array.isArray(cfg?.userPriority)
      ? cfg.userPriority.map((x) => safe(x).trim()).filter(Boolean)
      : [""]
  );

  setSpecificValuePriority(() => {
    const rows = Array.isArray(cfg?.specificValuePriority) ? cfg.specificValuePriority : [];
    const norm = rows
      .map((r) => ({
        id: uid("sv"),
        fieldCode: safe(r?.fieldCode).trim(),
        value: safe(r?.value),
      }))
      .filter((r) => r.fieldCode || r.value);
    return norm.length > 0 ? norm : [{ id: uid("sv"), fieldCode: "", value: "" }];
  });

  setRunModelAfterSave(Boolean(cfg?.runModelAfterSave));

  setMatchFieldCodes(
    Array.isArray(cfg?.matchFieldCodes)
      ? cfg.matchFieldCodes.map((c) => safe(c).trim()).filter(Boolean)
      : []
  );
  setMatchFieldPick("");
  setWeightDraftById({});
  setWeightsTouched(true);

  // Fields: merge defaults with loaded fields (by code)
  const loadedFields = Array.isArray(cfg?.fields) ? cfg.fields : [];
  const byCode = new Map(
    loadedFields
      .map((f) => ({ ...f, code: safe(f?.code).trim() }))
      .filter((f) => f.code)
      .map((f) => [f.code, f])
  );

  const used = new Set();

  function mapField(f) {
    const code = safe(f?.code).trim();
    return {
      id: safe(f?.id).trim() || uid("f"),
      code,
      label: safe(f?.label),
      kind: safe(f?.kind).trim() || "flex",
      include: Boolean(f?.include),
      key: Boolean(f?.key),
      type: safe(f?.type).trim() || "text",
      weight: Number(f?.weight || 0) || 0,
      rule: f?.rule,
      matchThreshold: f?.matchThreshold,
    };
  }

  const merged = [];

  for (const df of DEFAULT_FIELDS) {
    const hit = byCode.get(df.code);
    if (hit) {
      merged.push(mapField(hit));
      used.add(df.code);
    } else {
      merged.push({ ...df, id: uid("f") });
      used.add(df.code);
    }
  }

  for (const lf of loadedFields) {
    const code = safe(lf?.code).trim();
    if (!code || used.has(code)) continue;
    merged.push(mapField(lf));
    used.add(code);
  }

  setFields(merged);
}

useEffect(() => {
  if (!open) return;

  let id = "";
  try {
    id = String(window.localStorage?.getItem(LS_SELECTED_MODEL_ID) || "").trim();
  } catch {
    id = "";
  }

  if (!id) {
    setEditingModelId(null);
    return;
  }

  const userId = getUserId();
  if (!userId) {
    setSaveError("No user id found in authStorage (treat as logged out)");
    return;
  }

  let cancelled = false;

  (async () => {
    try {
      const url = `${apiBase}/mdm/models/${encodeURIComponent(id)}`;
      const res = await fetch(url, {
        method: "GET",
        headers: {
          "X-User-Id": String(userId),
          Accept: "application/json",
        },
      });

      const out = await res.json().catch(() => ({}));

      if (!res.ok || !out?.ok) {
        if (!cancelled) setSaveError(out?.error || `Failed to load model (${res.status})`);
        return;
      }

      const cfg = out?.model?.config;
      if (!cfg || typeof cfg !== "object") {
        if (!cancelled) setSaveError("Loaded model is missing config");
        return;
      }

      if (cancelled) return;
      applyLoadedModelConfig(id, cfg);
    } catch (e) {
      if (!cancelled) setSaveError(String(e?.message || e));
    }
  })();

  return () => {
    cancelled = true;
  };
}, [open, apiBase]);

useEffect(() => {
  if (!open) return;
  if (step !== 2) return;
  if (globalRule !== "system") return;

    const userId = getUserId();
    if (!userId) {
      setSourceSystemsError("No user id found in authStorage (treat as logged out)");
      setSourceSystemOptions([]);
      return;
    }

    const url = `${apiBase}/mdm/source-systems?include_stats=0`;

    let cancelled = false;

    setSourceSystemsLoading(true);
    setSourceSystemsError("");

    (async () => {
      try {
        const res = await fetch(url, {
          method: "GET",
          headers: {
            "X-User-Id": String(userId),
          },
        });

        const out = await res.json().catch(() => ({}));

        if (!res.ok) {
          if (!cancelled) {
            setSourceSystemsError(out?.error || `Source systems load failed (${res.status})`);
            setSourceSystemOptions([]);
          }
          return;
        }

        const rows = Array.isArray(out?.source_systems) ? out.source_systems : [];
        const names = rows
          .map((x) => String(x?.source_system ?? "").trim())
          .filter(Boolean);

        const uniq = Array.from(new Set(names));

        if (cancelled) return;

        setSourceSystemOptions(uniq);

        setSystemPriority((prev) => {
          const existing = (Array.isArray(prev) ? prev : [])
            .map((s) => String(s ?? "").trim())
            .filter(Boolean);

          if (existing.length > 0) return prev;

          return uniq.length > 0 ? uniq.slice(0, PRIORITY_CAP) : prev;
        });
      } catch (e) {
        if (!cancelled) {
          setSourceSystemsError(String(e?.message || e));
          setSourceSystemOptions([]);
        }
      } finally {
        if (!cancelled) setSourceSystemsLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [open, step, globalRule, apiBase, PRIORITY_CAP]);

  function buildMdmConfig(effectiveFields) {
    return {
      domainModelName: domainModelName,


      sourceType: sourceType,
      csvFileName: csvFileName,
      recordCount: recordCount,

      apiEndpoint: apiEndpoint,
      apiToken: apiToken,

      aiExceptions: aiExceptions,

      matchThreshold: matchThreshold,
      possibleThreshold: possibleThreshold,

      advanced: advanced,
      globalRule: globalRule,

      systemPriority: systemPriority,
      userPriority: userPriority,

      specificValuePriority: specificValuePriority.map((x) => ({
        fieldCode: x.fieldCode,
        value: x.value,
      })),

      runModelAfterSave: runModelAfterSave,
      matchFieldCodes: matchFieldCodes,

      fields: effectiveFields.map((f) => ({
        id: f.id,
        code: f.code,
        label: f.label,
        kind: f.kind,
        include: !!f.include,
        key: !!f.key,
        type: f.type,

        weight: matchFieldCodes.includes(f.code) ? (Number(f.weight) || 0) : 0,

        rule: f.rule,

        matchThreshold: f.matchThreshold,
      })),
    };
  }


  async function saveMdmModelToApi() {
    setSaveError("");

    const name = String(domainModelName || "").trim();
    if (!name) {
      setSaveError("Model name is required");
      return null;
    }

    if (!/^[A-Za-z0-9_-]+$/.test(name)) {
      setSaveError("Model name must be alphanumeric and may include _ or - only");
      return null;
    }

    if (!actor) {
      setSaveError("No actor found (pass actor prop or set localStorage.username)");
      return null;
    }

    const userId = getUserId();
    if (!userId) {
      setSaveError("No user id found in authStorage (treat as logged out)");
      return null;
    }

    // Apply same “non-advanced” survivorship behavior as UI does on finish.
    const effectiveFields = !advanced
      ? fields.map((f) => (f.include ? { ...f, rule: globalRule } : f))
      : fields;

    const config = buildMdmConfig(effectiveFields);

    const url = editingModelId
      ? `${apiBase}/mdm/models/${encodeURIComponent(String(editingModelId))}`
      : `${apiBase}/mdm/models`;

    setSavingModel(true);
    try {
      const res = await fetch(url, {
        method: editingModelId ? "PUT" : "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": String(userId),
        },
        body: JSON.stringify({ actor, model_name: name, config }),
      });

      const out = await res.json().catch(() => ({}));

      if (!res.ok) {
        setSaveError(out?.error || `Save failed (${res.status})`);
        return null;
      }

      onSaved?.(out);
      return out;
    } catch (e) {
      setSaveError(String(e?.message || e));
      return null;
    } finally {
      setSavingModel(false);
    }
  }


  async function copyToClipboard(text, key) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(key);
      window.setTimeout(() => setCopied(null), 1200);
    } catch {
      // ignore
    }
  }


  async function rotateToken() {
    setCopied(null);

    const userId = getUserId();
    if (!userId) {
      setSaveError("No user id found in authStorage (treat as logged out)");
      return;
    }

    const mid = String(editingModelId || "").trim();
    if (!mid) {
      setSaveError("Save the model first before rotating an API token.");
      return;
    }

    try {
      const res = await fetch(`${apiBase}/api/api-keys/rotate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": String(userId),
        },
        body: JSON.stringify({ model_id: mid, name: domainModelName || null }),
      });

      const out = await res.json().catch(() => ({}));

      if (!res.ok) {
        setSaveError(out?.error || `Token rotate failed (${res.status})`);
        return;
      }

      const token = String(out?.token || "").trim();
      if (!token) {
        setSaveError("Token rotate succeeded but no token was returned by the API.");
        return;
      }

      setApiToken(token);
    } catch (e) {
      setSaveError(String(e?.message || e));
    }
  }

  function pickCsv() {
    fileRef.current?.click();
  }

  function onCsvPicked(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setCsvUploadError("");
    setCsvUploaded(false);
    setCsvUploadRows(0);

    setCsvFile(file);
    setCsvFileName(file.name);

    setRecordCount((v) => (v > 0 ? v : 12500));
    e.target.value = "";
  }


  function setKey(id) {
    setFields((prev) => {
      const tgt = prev.find((x) => x.id === id);
      if (!tgt) return prev;
      if (tgt.kind === "reserved") return prev;
      if (tgt.code === "source_name") return prev;
      return prev.map((f) => ({ ...f, key: f.id === id }));
    });
  }

  function toggleInclude(id) {
    setFields((prev) =>
      prev.map((f) => {
        if (f.id !== id) return f;
        if (f.kind === "required") return f;

        if (f.kind === "flex") {
          const hasLabel = Boolean((f.label ?? "").trim());
          if (!hasLabel && !f.include) return f;
        }

        return { ...f, include: !f.include };
      })
    );
  }

  function setAllFlexIncludes(next) {
    setFields((prev) =>
      prev.map((f) => {
        if (f.kind !== "flex") return f;

        if (next) {
          const hasLabel = Boolean((f.label ?? "").trim());
          if (!hasLabel && !f.include) return f;
        }

        return { ...f, include: next };
      })
    );
  }

  function renameField(id, label) {
    setFields((prev) =>
      prev.map((f) => {
        if (f.id !== id) return f;
        if (f.kind !== "flex") return f;

        const nextLabel = label;
        const nextHasLabel = Boolean(String(nextLabel ?? "").trim());
        const prevHasLabel = Boolean(String(f.label ?? "").trim());

        if (!f.include && !prevHasLabel && nextHasLabel) {
          return { ...f, label: nextLabel, include: true };
        }

        return { ...f, label: nextLabel };
      })
    );
  }


  function setType(id, type) {
    setFields((prev) => prev.map((f) => (f.id === id ? { ...f, type } : f)));
  }

  function autoDetectTypes() {
    setFields((prev) =>
      prev.map((f) => {
        if (f.kind !== "flex") return f;
        const src = (typeof f.label === "string" && f.label.trim()) ? f.label : (f.code || "");
        return { ...f, type: guessType(src) };
      })
    );
  }

  function addField() {
    setFields((prev) => {
      const used = new Set(prev.filter((f) => f.kind === "flex").map((f) => f.code));
      let nextCode = null;

      for (let i = 1; i <= 20; i++) {
        const code = `f${String(i).padStart(2, "0")}`;
        if (!used.has(code)) {
          nextCode = code;
          break;
        }
      }

      if (!nextCode) return prev;

      return [
        ...prev,
        {
          id: uid("f"),
          code: nextCode,
          label: "",
          kind: "flex",
          include: true,
          key: false,
          type: "text",
          weight: 0,
          rule: globalRule,
        },
      ];
    });
  }

  function setThresholds(nextMatch, nextPossible) {
    const mt = clamp(nextMatch, 0.5, 0.99);
    let pt = clamp(nextPossible, 0.3, 0.95);
    if (pt >= mt) pt = Math.max(0.3, mt - 0.05);
    setMatchThreshold(mt);
    setPossibleThreshold(pt);
  }

  function setFieldMatchThreshold(id, t) {
    setFields((prev) =>
      prev.map((f) => (f.id === id ? { ...f, matchThreshold: clamp(t, 0, 1) } : f))
    );
  }

  function setWeightPct(id, pct) {
    setWeightsTouched(true);
    setFields((prev) => {
      const ids = matchFieldCodes
        .map((code) => prev.find((f) => f.code === code)?.id)
        .filter(Boolean);
      if (!ids.includes(id)) return prev;

      const n = ids.length;
      if (n === 1) return prev.map((f) => (f.id === id ? { ...f, weight: 1 } : f));

      const targetPct = Math.max(1, Math.min(100, Math.round(Number(pct) || 0)));

      return prev.map((f) => (f.id === id ? { ...f, weight: targetPct / 100 } : f));
    });
  }

  function addMatchField(nextCode) {
    const code = String((nextCode ?? matchFieldPick) || "").trim();
    if (!code) return;

    setMatchFieldCodes((prev) => (prev.includes(code) ? prev : [...prev, code]));
    setMatchFieldPick("");
  }


  function removeMatchField(code) {
    setMatchFieldCodes((prev) => prev.filter((c) => c !== code));
    setMatchFieldPick((prev) => (prev === code ? "" : prev));
  }


  function applyGlobalRule(rule) {
    setGlobalRule(rule);
    if (!advanced) {
      setFields((prev) => prev.map((f) => (f.include ? { ...f, rule } : f)));
    }
  }

  function setFieldRule(id, rule) {
    setFields((prev) => prev.map((f) => (f.id === id ? { ...f, rule } : f)));
  }

  function toggleAdvanced(v) {
    setAdvanced(v);
    if (!v) {
      setFields((prev) => prev.map((f) => (f.include ? { ...f, rule: globalRule } : f)));
    }
  }

  function pickSystem(idx, value) {
    setSystemPriority((prev) => setUniqueValue(prev, idx, value));
  }

  function pickUser(idx, value) {
    setUserPriority((prev) => setUniqueValue(prev, idx, value));
  }

  function addSystem() {
    setSystemPriority((prev) => (prev.length >= PRIORITY_CAP ? prev : [...prev, ""]));
  }

  function addUser() {
    setUserPriority((prev) => (prev.length >= PRIORITY_CAP ? prev : [...prev, ""]));
  }

  function removeSystem(idx) {
    setSystemPriority((prev) => {
      const next = prev.filter((_, i) => i !== idx);
      return next.length ? next : [""];
    });
  }

  function removeUser(idx) {
    setUserPriority((prev) => {
      const next = prev.filter((_, i) => i !== idx);
      return next.length ? next : [""];
    });
  }

  function reorderSystem(from, to) {
    setSystemPriority((prev) => reorderArray(prev, from, to));
  }

  function reorderUser(from, to) {
    setUserPriority((prev) => reorderArray(prev, from, to));
  }

  async function finishWizard() {
    if (!step3Ok) {
      setSaveError("Select at least 1 match field");
      return;
    }
    if (!weightsOk) {
      setSaveError("Total weight must be 100%");
      return;
    }


    if (!advanced && globalRule === "specific_value_priority") {
      const invalid = specificValuePriority.some((row) => {
        const fc = String(row?.fieldCode ?? "");
        const v = String(row?.value ?? "");
        return fc.trim().length === 0 || v.trim().length === 0;
      });

      if (invalid) {
        setSpecificValueError("Field and Value are required for each row.");
        return;
      }
    }

    const wasUpdate = Boolean(editingModelId);
    const saved = await saveMdmModelToApi();
    if (!saved) return;

    const createdName = String(saved?.model_name || domainModelName || "").trim() || "Model";
    const createdId = saved?.id ?? editingModelId ?? null;

    try {
      const sid = createdId != null ? String(createdId).trim() : "";
      if (sid) window.localStorage?.setItem(LS_SELECTED_MODEL_ID, sid);
    } catch {}

    if (!advanced) {
      setFields((prev) => prev.map((f) => (f.include ? { ...f, rule: globalRule } : f)));
    }

    setConfigured(true);
    setSaveSuccess({ name: createdName, id: createdId, action: wasUpdate ? "updated" : "created" });

  }


  async function next() {
    if (step === 0 && !step1Ok) return;
    if (step === 1 && !step2Ok) return;

    if (step === 0 && sourceType === "csv") {
      if (csvUploading) return;

      // If we already uploaded a CSV for this wizard session (or loaded an existing model),
      // don't force a re-upload unless the user picked a new file.
      if (csvUploaded && !csvFile) {
        setStep(1);
        return;
      }

      setCsvUploadError("");
      setCsvUploaded(false);
      setCsvUploadRows(0);

      if (!csvFile) {
        setCsvUploadError("No CSV file selected");
        return;
      }


      const userId = getUserId();
      if (!userId) {
        setCsvUploadError("No user id found in authStorage (treat as logged out)");
        return;
      }

      const url = `${apiBase}/ingest/csv`;

      const fd = new FormData();
      fd.append("file", csvFile, csvFile.name);

      setCsvUploading(true);
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: {
            "X-User-Id": String(userId),
          },
          body: fd,
        });

        const out = await res.json().catch(() => ({}));

        if (!res.ok) {
          setCsvUploadError(out?.error || `CSV upload failed (${res.status})`);
          return;
        }

        const schemaFields = out?.schema?.fields;
        if (!Array.isArray(schemaFields)) {
          setCsvUploadError("CSV upload succeeded but schema.fields is missing");
          return;
        }

        const schemaByCode = new Map(
          schemaFields
            .map((x) => ({
              code: String(x?.code || "").trim(),
              header: x?.header,
              type: x?.type,
            }))
            .filter((x) => x.code)
            .map((x) => [x.code, x])
        );

        setFields((prev) => {
          const next = [...prev];
          const existingCodes = new Set(next.map((f) => f.code));

          for (const [code] of schemaByCode) {
            if (existingCodes.has(code)) continue;
            next.push({
              id: uid("f"),
              code: code,
              label: "",
              kind: "flex",
              include: true,
              key: false,
              type: "text",
              weight: 0,
              rule: globalRule,
            });
            existingCodes.add(code);
          }

          return next.map((f) => {
            if (f.kind !== "flex") return f;
            const hit = schemaByCode.get(f.code);
            if (!hit) return f;
            return {
              ...f,
              label: String(hit.header ?? ""),
              include: true,
              type: String(hit.type ?? "text"),
            };
          });
        });

        setCsvUploaded(true);
        setCsvUploadRows(
          Number(out?.received_rows || 0) || Number(out?.upserted_rows || 0) || 0
        );

        setStep(1);
        return;
      } catch (e) {
        setCsvUploadError(String(e?.message || e));
        return;
      } finally {
        setCsvUploading(false);
      }
    }

    setStep((s) => Math.min(2, s + 1));
  }


  function back() {
    setStep((s) => Math.max(0, s - 1));
  }

  if (!open) return null;

  return (
    <div className="mdmOverlay" role="dialog" aria-modal="true">
      <div
        className={[
          "mdmWizard",
          step === 2 && "mdmWizard--step3"
        ].filter(Boolean).join(" ")}
      >
        <div className="mdmWizard__head">
          <div>
            <h1>MDM Setup</h1>
            <p>MDM in 3 easy steps</p>
          </div>
          <button className="mdmX" type="button" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

      <div className="mdmStepper">
        {/* ... your StepButton components that reference `step` ... */}
          <StepButton
            num={1}
            title="MDM 1 — Source"
            hint="CSV or API key"
            active={step === 0}
            done={configured || step > 0}
            onClick={() => setStep(0)}
          />
          <StepButton
            num={2}
            title="MDM 2 — Fields"
            hint="Include + types + key"
            active={step === 1}
            done={configured || step > 1}
            onClick={() => {
              if (!step1Ok) return;
              if (step === 0) {
                next();
                return;
              }
              setStep(1);
            }}
          />

          <StepButton
            num={3}
            title="MDM 3 — Matching"
            hint="Thresholds + survivorship"
            active={step === 2}
            done={configured}
            onClick={() => {
              if (!step1Ok) return;
              if (!step2Ok) return;
              setStep(2);
            }}
          />
        </div>


        <div
          className="mdmWizard__body"
          onWheelCapture={
            step === 2
              ? (e) => {
                  const t = e.target;
                  if (t && typeof t.closest === "function") {
                    const inner = t.closest(".mdmWeightsList, .mdmPriorityList, .mdmSurvivorshipList, .mdmTableBody");
                    if (inner) {
                      const max = inner.scrollHeight - inner.clientHeight;
                      if (max > 0) {
                        const delta = e.deltaY || 0;
                        const top = inner.scrollTop;
                        const atTop = top <= 0;
                        const atBottom = top >= max - 1;
                        if ((delta < 0 && !atTop) || (delta > 0 && !atBottom)) return;
                      }
                    }
                  }

                  wheelToScrollContainer(e);
                }
              : undefined
          }
        >

          {/* STEP 1 */}
          {step === 0 && (
            <>
              <div className="mdmCard">
                <div className="mdmCard__head">
                  <div>
                    <div className="mdmCard__title">Domain — Model name</div>
                    <div className="mdmCard__sub">Name this model so you can track it later.</div>
                  </div>
                </div>

                <div className="mdmCard__body">
                  <div>
                    <div className="mdmLabel">Domain — Model name</div>
                    <input
                      className="mdmInput"
                      value={domainModelName}
                      placeholder="ADD MODEL NAME"
                      onChange={(e) => setDomainModelName(e.target.value)}
                      required
                      pattern="[A-Za-z0-9_\-]+"
                      aria-invalid={modelNameTrimmed.length > 0 && !modelNameOk}
                    />
                  </div>
                </div>
              </div>

              <div className="mdmCard">
                <div className="mdmCard__head">
                  <div>
                    <div className="mdmCard__title">Pick a source</div>
                    <div className="mdmCard__sub">CSV upload or inbound API key.</div>
                  </div>
                </div>



              <div className="mdmCard__body">
                <div className="mdmSeg">
                  <button
                    type="button"
                    className={sourceType === "csv" ? "isActive" : ""}
                    onClick={() => setSourceType("csv")}
                  >
                    Load CSV
                  </button>
                  <button
                    type="button"
                    className={sourceType === "api" ? "isActive" : ""}
                    onClick={() => setSourceType("api")}
                  >
                    API key
                  </button>
                </div>

                <div style={{ height: 12 }} />

                {sourceType === "csv" ? (
                  <div className="mdmRow2">
                    <div>
                      <div className="mdmLabel">CSV file</div>
                      <button className="mdmBtn mdmBtn--soft" type="button" onClick={pickCsv}>
                        Choose file
                      </button>
                      <div style={{ marginTop: 8 }} className="mdmTiny">
                        Selected: <span className="mdmMono">{csvFileName || "—"}</span>
                      </div>

                      {csvUploadError ? (
                        <div className="mdmTiny" style={{ marginTop: 8, color: "var(--coral1)" }}>
                          {csvUploadError}
                        </div>
                      ) : null}
                    </div>


                    <div>
                      <div className="mdmLabel">Record volume (estimate)</div>
                      <input
                        className="mdmInput"
                        type="number"
                        min={0}
                        value={recordCount}
                        onChange={(e) => setRecordCount(Number(e.target.value))}
                      />
                      <div className="mdmTiny" style={{ marginTop: 8 }}>
                        Used for sizing + performance expectations.
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mdmRow2">
                      <div>
                        <div className="mdmLabel">Endpoint URL</div>
                        <div className="mdmInputAction">
                          <input
                            className="mdmInput"
                            value={apiEndpoint}
                            onChange={(e) => setApiEndpoint(e.target.value)}
                          />
                          <button
                            className="mdmBtn mdmBtn--soft mdmBtn--small"
                            type="button"
                            onClick={() => copyToClipboard(apiEndpoint, "endpoint")}
                          >
                            {copied === "endpoint" ? "Copied" : "Copy"}
                          </button>
                        </div>
                        <div className="mdmTiny" style={{ marginTop: 8 }}>
                          Your systems POST records to this URL.
                        </div>
                      </div>

                      <div>
                        <div className="mdmLabel">API token</div>
                        <div className="mdmInputAction">
                          <div className="mdmInputWithIcon">
                            <input
                              className="mdmInput mdmInput--withIcon"
                              type={showToken ? "text" : "password"}
                              value={apiToken}
                              onChange={(e) => setApiToken(e.target.value)}
                            />
                            <button
                              className="mdmInputIconBtn"
                              type="button"
                              onClick={() => setShowToken((v) => !v)}
                              aria-label={showToken ? "Hide token" : "Show token"}
                              title={showToken ? "Hide token" : "Show token"}
                            >
                              {showToken ? (
                                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">
                                  <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                                  <path d="M4 4l16 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                </svg>
                              ) : (
                                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">
                                  <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                                </svg>
                              )}
                            </button>
                          </div>

                          <div className="mdmBtnGroup">
                            <button
                              className="mdmBtn mdmBtn--soft mdmBtn--small"
                              type="button"
                              onClick={rotateToken}
                              title="Generate a new token"
                            >
                              Rotate
                            </button>
                            <button
                              className="mdmBtn mdmBtn--soft mdmBtn--small"
                              type="button"
                              onClick={() => copyToClipboard(apiToken, "token")}
                            >
                              {copied === "token" ? "Copied" : "Copy"}
                            </button>
                          </div>
                        </div>

                        <div className="mdmTiny" style={{ marginTop: 8 }}>
                          Send as <span className="mdmMono">Authorization: Bearer &lt;token&gt;</span>.
                        </div>
                      </div>
                    </div>

                    <div style={{ height: 12 }} />

                    <div>
                      <div className="mdmLabel">Record volume (estimate)</div>
                      <input
                        className="mdmInput"
                        type="number"
                        min={0}
                        value={recordCount}
                        onChange={(e) => setRecordCount(Number(e.target.value))}
                      />
                      <div className="mdmTiny" style={{ marginTop: 8 }}>
                        Used for sizing + performance expectations.
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
            </>
          )}


          {/* STEP 2 */}
          {step === 1 && (
            <>
              <div className="mdmHero" style={{ margin: 0 }}>
                <div>
                  <h2>Fields</h2>
                  <p>Pick key + include fields. Keep it simple.</p>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button className="mdmBtn mdmBtn--soft" type="button" onClick={autoDetectTypes}>
                    Auto-detect types
                  </button>
                  <button className="mdmBtn mdmBtn--primary" type="button" onClick={addField}>
                    Add field
                  </button>
                </div>
              </div>

              <div className="mdmTable">
                <div className="mdmTHead">
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <input
                      className="mdmCheck"
                      type="checkbox"
                      checked={allFlexIncluded}
                      disabled={selectableFlex.length === 0}
                      ref={(el) => {
                        if (el) el.indeterminate = !allFlexIncluded && someFlexIncluded;
                      }}
                      onChange={(e) => setAllFlexIncludes(e.target.checked)}
                      title="Select all named flex fields"
                    />
                    <span>All</span>
                  </div>
                  <div>Key</div>
                  <div>Field</div>
                  <div>Type</div>
                </div>

                <div className="mdmTableBody" onWheelCapture={wheelToScrollContainer}>
                  {sortedFields.map((f) => {
                    const isRequired = f.kind === "required";
                    const isReserved = f.kind === "reserved";
                    const isFlex = f.kind === "flex";

                    return (
                      <div
                        className={[
                          "mdmTRow",
                          isRequired ? "mdmTRow--required" : "",
                          isReserved ? "mdmTRow--reserved" : "",
                        ].join(" ")}
                        key={f.id}
                      >
                        <div>
                          <input
                            className="mdmCheck"
                            type="checkbox"
                            checked={!!f.include}
                            disabled={
                              isRequired ||
                              (isFlex && !Boolean((f.label ?? "").trim()) && !f.include)
                            }
                            onChange={() => toggleInclude(f.id)}
                          />
                        </div>

                        <div>
                          <input
                            className="mdmCheck"
                            type="radio"
                            name="pk"
                            checked={!!f.key}
                            disabled={isReserved || f.code === "source_name"}
                            onChange={() => setKey(f.id)}
                          />
                        </div>

                        <div>
                          <input
                            className={`mdmInput ${isFlex ? "" : "mdmInput--locked"}`}
                            value={isFlex ? (f.label ?? "") : f.code}
                            placeholder={isFlex ? "Source field name" : ""}
                            disabled={!isFlex}
                            onChange={(e) => renameField(f.id, e.target.value)}
                          />
                        </div>

                        <div>
                          <select
                            className="mdmSelect"
                            value={f.type}
                            disabled={isReserved}
                            onChange={(e) => setType(f.id, e.target.value)}
                          >
                            {TYPE_OPTIONS.map((o) => (
                              <option key={o.value} value={o.value}>
                                {o.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="mdmTiny">
                Required: <b>source_id</b>, <b>source_name</b>. Reserved (optional): <b>created_at</b>, <b>created_by</b>, <b>updated_at</b>, <b>updated_by</b>.
                <br />
                Guardrail: pick <b>one Key</b> + include at least <b>1 flex field</b>.
              </div>
            </>
          )}

          {/* STEP 3 */}
          {step === 2 && (
            <>
              <div className="mdmGrid2">
                <div className="mdmCard">
                  <div className="mdmCard__head">
                    <div>
                      <div className="mdmCard__title">Matching thresholds</div>
                      <div className="mdmCard__sub">Match / Exception / No match</div>
                    </div>

                    {step3Ok && (
                      <label className="mdmHeadToggle">
                        <input
                          className="mdmCheck"
                          type="checkbox"
                          checked={aiExceptions}
                          onChange={(e) => setAiExceptions(e.target.checked)}
                        />
                        Enable Exceptions Agent
                      </label>
                    )}
                  </div>

                  <div className="mdmCard__body">
                    {!step3Ok && (
                      <div className="mdmTiny">Add match fields below to configure thresholds.</div>
                    )}

                    <div style={!step3Ok ? { display: "none" } : undefined}>
                      <div className="mdmLabel">Match threshold</div>
                      <div className="mdmSliderRow">
                        <input
                          type="range"
                          min={0.5}
                          max={0.99}
                          step={0.01}
                          value={matchThreshold}
                          onChange={(e) => setThresholds(Number(e.target.value), possibleThreshold)}
                        />
                        <div className="mdmMono">{pct01(matchThreshold)}</div>
                      </div>

                      <div style={{ height: 10 }} />

                      <div className="mdmLabel">Exception threshold</div>
                      <div className="mdmSliderRow">
                        <input
                          type="range"
                          min={0.3}
                          max={0.95}
                          step={0.01}
                          value={possibleThreshold}
                          onChange={(e) => setThresholds(matchThreshold, Number(e.target.value))}
                        />
                        <div className="mdmMono">{pct01(possibleThreshold)}</div>
                      </div>

                      <div style={{ marginTop: 10 }} className="mdmTiny">
                        Exceptions are between <span className="mdmMono">{pct01(possibleThreshold)}</span> and{" "}
                        <span className="mdmMono">{pct01(matchThreshold)}</span>.
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mdmCard">

                  <div className="mdmCard__head">
                    <div>
                      <div className="mdmCard__title">Survivorship</div>
                      <div className="mdmCard__sub">Simple by default. Advanced optional.</div>
                    </div>
                    {step3Ok && (
                      <div>
                        <label className="mdmTiny" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <input
                            className="mdmCheck"
                            type="checkbox"
                            checked={advanced}
                            onChange={(e) => toggleAdvanced(e.target.checked)}
                          />
                          Advanced
                        </label>
                      </div>
                    )}
                  </div>


                  <div className="mdmCard__body">
                    {!step3Ok && (
                      <div className="mdmTiny">Add match fields below to configure survivorship.</div>
                    )}

                    <div style={!step3Ok ? { display: "none" } : undefined}>
                      {!advanced ? (
                        <>
                          <div className="mdmLabel">Default rule (applies to included fields)</div>
                          <select
                            className="mdmSelect"
                            value={globalRule}
                            onChange={(e) => applyGlobalRule(e.target.value)}
                          >
                            {SURVIVORSHIP_RULES.map((r) => (
                              <option key={r.value} value={r.value}>
                                {r.label}
                              </option>
                            ))}
                          </select>


                        {globalRule === "system" && (
                          <>
                            <div style={{ height: 12 }} />
                            <div className="mdmLabel">System priority</div>

                            {sourceSystemsLoading && (
                              <div className="mdmTiny">Loading source systems…</div>
                            )}

                            {sourceSystemsError && (
                              <div className="mdmTiny" style={{ color: "var(--coral1)", fontWeight: 900 }}>
                                {sourceSystemsError}
                              </div>
                            )}

                            <PriorityPickerList
                              kind="system"
                              items={systemPriority}
                              options={systemOptions}
                              placeholder="Pick a system…"
                              cap={PRIORITY_CAP}
                              onPick={pickSystem}
                              onAdd={addSystem}
                              onRemove={removeSystem}
                              onReorder={reorderSystem}
                            />
                          </>
                        )}


                        {globalRule === "specific_value_priority" && (
                          <>
                            <div style={{ height: 12 }} />
                            <div className="mdmLabel">Specific value priority</div>

                            <div className="mdmPriorityList" onWheelCapture={wheelToScrollContainer}>
                              {specificValuePriority.map((row, idx) => {
                                const fieldOptions = includedFields.filter((f) => f.code !== "source_id");

                                return (
                                  <div
                                    key={row.id}
                                    className="mdmPriorityRow"
                                    onDragOver={(e) => e.preventDefault()}
                                    onDrop={() => {
                                      const from = specificValueDragFromRef.current;
                                      specificValueDragFromRef.current = null;
                                      if (from === null || from === undefined) return;
                                      if (from === idx) return;
                                      setSpecificValueError("");
                                      setSpecificValuePriority((prev) => reorderArray(prev, from, idx));
                                    }}
                                  >
                                    <div className="mdmPriorityLeft">
                                      <div
                                        className="mdmDragHandle"
                                        draggable
                                        onDragStart={() => { specificValueDragFromRef.current = idx; }}
                                        onDragEnd={() => { specificValueDragFromRef.current = null; }}
                                        title="Drag to reorder"
                                        aria-label="Drag to reorder"
                                      >

                                        <svg viewBox="0 0 12 12" width="14" height="14" aria-hidden="true">
                                          <circle cx="3" cy="2" r="1" fill="currentColor" />
                                          <circle cx="9" cy="2" r="1" fill="currentColor" />
                                          <circle cx="3" cy="6" r="1" fill="currentColor" />
                                          <circle cx="9" cy="6" r="1" fill="currentColor" />
                                          <circle cx="3" cy="10" r="1" fill="currentColor" />
                                          <circle cx="9" cy="10" r="1" fill="currentColor" />
                                        </svg>
                                      </div>

                                      <div className="mdmPriorityName">
                                        <div className="mdmPriorityPickWrap" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                          <select
                                            className="mdmSelect mdmPriorityPickSelect mdmMono"
                                            value={String(row.fieldCode ?? "")}
                                            onChange={(e) => {
                                              const fieldCode = e.target.value;
                                              setSpecificValueError("");
                                              setSpecificValuePriority((prev) =>
                                                prev.map((r, i) => (i === idx ? { ...r, fieldCode } : r))
                                              );
                                            }}
                                          >
                                            <option value="">Pick a field…</option>
                                            {fieldOptions.map((f) => {
                                              const label = (f.label ?? "").trim() || f.code;
                                              return (
                                                <option key={f.code} value={f.code}>
                                                  {label}
                                                </option>
                                              );
                                            })}
                                          </select>

                                          <input
                                            className="mdmInput mdmMono"
                                            value={String(row.value ?? "")}
                                            placeholder="Enter exact value…"
                                            onChange={(e) => {
                                              const value = e.target.value;
                                              setSpecificValueError("");
                                              setSpecificValuePriority((prev) =>
                                                prev.map((r, i) => (i === idx ? { ...r, value } : r))
                                              );
                                            }}
                                          />
                                        </div>
                                      </div>
                                    </div>

                                    <div className="mdmPriorityActions">
                                      <button
                                        className="mdmBtn mdmIconBtn mdmBtn--ghost"
                                        type="button"
                                        onClick={() => {
                                          setSpecificValueError("");
                                          setSpecificValuePriority((prev) => {
                                            const next = prev.filter((_, i) => i !== idx);
                                            return next.length ? next : [{ id: uid("sv"), fieldCode: "", value: "" }];
                                          });
                                        }}
                                        disabled={specificValuePriority.length <= 1}
                                        aria-label="Remove"
                                        title="Remove"
                                      >
                                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true">
                                          <path d="M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                          <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                        </svg>
                                      </button>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>

                            <div className="mdmPriorityAddRow">
                              <button
                                className="mdmBtn mdmBtn--soft mdmBtn--small"
                                type="button"
                                onClick={() => {
                                  setSpecificValueError("");
                                  setSpecificValuePriority((prev) => {
                                    if (prev.length >= PRIORITY_CAP) return prev;
                                    return [...prev, { id: uid("sv"), fieldCode: "", value: "" }];
                                  });
                                }}
                                disabled={specificValuePriority.length >= PRIORITY_CAP}
                                title={specificValuePriority.length >= PRIORITY_CAP ? `Max ${PRIORITY_CAP}` : "Add"}
                              >
                                Add value
                              </button>
                            </div>

                            {specificValueError && (
                              <div className="mdmTiny" style={{ color: "var(--coral1)", fontWeight: 900 }}>
                                {specificValueError}
                              </div>
                            )}
                          </>
                        )}

                      </>
                    ) : (
                      <>
                        <div className="mdmTiny">Advanced = rule per field (included fields only).</div>
                        <div style={{ height: 10 }} />
                        <div className="mdmSurvivorshipList" onWheelCapture={wheelToScrollContainer}>
                          {includedFields
                            .filter((f) => f.code !== "source_id")
                            .map((f) => (
                              <div key={f.id} className="mdmRow2" style={{ marginBottom: 10 }}>
                                <div>
                                  <div className="mdmLabel">Field</div>
                                  <div
                                    className="mdmMono"
                                    style={{
                                      padding: "10px 12px",
                                      border: "1px solid var(--border)",
                                      borderRadius: 14,
                                      background: "var(--panel2)",
                                    }}
                                  >
                                    {((f.label ?? "").trim() || f.code)}
                                  </div>
                                </div>
                                <div>
                                  <div className="mdmLabel">Rule</div>
                                  <select
                                    className="mdmSelect"
                                    value={f.rule}
                                    onChange={(e) => setFieldRule(f.id, e.target.value)}
                                  >
                                    {SURVIVORSHIP_RULES.filter((r) => r.value !== "specific_value_priority").map((r) => (
                                      <option key={r.value} value={r.value}>
                                        {r.label}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                              </div>
                            ))}
                        </div>
                      </>
                    )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mdmCard">

                <div className="mdmCard__head">
                  <div>
                    <div className="mdmCard__title">Per-field matching</div>
                    <div className="mdmCard__sub">Match % threshold + weight % per field.</div>
                  </div>
                </div>

                <div className="mdmCard__body">
                  <div style={{ display: "flex", gap: 10, alignItems: "flex-end", flexWrap: "wrap" }}>
                    <div style={{ flex: "1 1 320px" }}>
                      <div className="mdmLabel">Add match field</div>
                      <select
                        className="mdmSelect"
                        value={matchFieldPick}
                        onChange={(e) => addMatchField(e.target.value)}
                      >
                        <option value="">Select a field…</option>
                        {matchFieldOptions.map((f) => (
                          <option key={f.code} value={f.code}>
                            {(f.label ?? "").trim()}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div style={{ height: 10 }} />

                  {matchFields.length === 0 ? (
                    <div className="mdmTiny">No match fields selected. Select a match field above.</div>
                  ) : (
                    <>
                      <div className="mdmPerFieldHeader">
                        <div className="mdmTiny mdmPerFieldHeaderLabel">Field</div>
                        <div className="mdmTiny mdmPerFieldHeaderLabel">Match %</div>
                        <div className="mdmTiny mdmPerFieldHeaderLabel">Weight</div>
                        <div aria-hidden="true" />


                        <div
                          className="mdmTiny mdmPerFieldTotalWeight"
                          style={totalWeightPct === 100 ? undefined : { color: "var(--coral1)", fontWeight: 900 }}
                        >
                          Total weight: <span className="mdmMono">{totalWeightPct}%</span>
                          {totalWeightPct === 100 ? "" : " (must be 100%)"}
                        </div>
                      </div>

                      <div className="mdmWeightsList" onWheelCapture={wheelToScrollContainer}>
                        {matchFields.map((f) => {
                          const label =
                            (typeof f.label === "string" && f.label.trim()) ||
                            "Source field name";

                          const thresholdPct = Math.max(
                            1,
                            Math.min(100, Math.round((Number(f.matchThreshold) || 0.8) * 100))
                          );

                          const weightPct = Math.max(
                            0,
                            Math.min(100, Math.round((Number(f.weight) || 0) * 100))
                          );

                          const weightDraft = weightDraftById[f.id];

                          const maxWeightPct = matchFields.length > 1 ? (100 - (matchFields.length - 1)) : 100;

                          return (
                            <div className="mdmWeightRow" key={f.id}>
                              <div className="mdmFieldName">
                                <div style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                  {label}
                                </div>
                              </div>

                              <div className="mdmSliderRow">
                                <input
                                  type="range"
                                  min={1}
                                  max={100}
                                  step={1}
                                  value={thresholdPct}
                                  onChange={(e) => {
                                    const v = Number(e.target.value);
                                    if (Number.isNaN(v)) return;
                                    const clamped = Math.max(1, Math.min(100, v));
                                    setFieldMatchThreshold(f.id, clamped / 100);
                                  }}
                                />
                                <div className="mdmMono">{thresholdPct}%</div>
                              </div>

                              <div className="mdmWeightRight">
                                <input
                                  className="mdmInput mdmWeightInput"
                                  type="number"
                                  min={1}
                                  max={maxWeightPct}
                                  value={weightDraft !== undefined ? weightDraft : weightPct}
                                  disabled={matchFields.length === 1}
                                  onChange={(e) => {
                                    setWeightsTouched(true);
                                    setWeightDraftById((prev) => ({ ...prev, [f.id]: e.target.value }));
                                  }}
                                  onKeyDown={(e) => {

                                    if (e.key === "Enter") e.currentTarget.blur();
                                  }}
                                  onBlur={(e) => {
                                    const raw = String(e.target.value ?? "").trim();

                                    setWeightDraftById((prev) => {
                                      if (!Object.prototype.hasOwnProperty.call(prev, f.id)) return prev;
                                      const next = { ...prev };
                                      delete next[f.id];
                                      return next;
                                    });

                                    if (!raw) return;

                                    const v = Number(raw);
                                    if (Number.isNaN(v)) return;

                                    const clamped = Math.max(1, Math.min(maxWeightPct, v));
                                    if (clamped === weightPct) return;
                                    setWeightPct(f.id, clamped);
                                  }}
                                />
                                <div className="mdmMono">%</div>
                              </div>

                              <div className="mdmWeightRemove">
                                <button
                                  className="mdmBtn mdmIconBtn mdmBtn--ghost"
                                  type="button"
                                  onClick={() => removeMatchField(f.code)}
                                  aria-label="Remove match field"
                                  title="Remove"
                                >
                                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true">
                                    <path d="M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                    <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  )}

                </div>
              </div>
            </>

          )}
        </div>

        <div className="mdmWizard__footer">
          <div className="mdmFooterLeft">
            Step <span className="mdmMono">{step + 1}</span> / <span className="mdmMono">3</span>
            {saveError && (
              <>
                {" • "}
                <span className="mdmMono" style={{ color: "var(--coral1)" }}>{saveError}</span>
              </>
            )}

            {step === 0 && !modelNameOk && (
              <>
                {" • "}
                <span className="mdmMono">Model name is required (letters/numbers, _ and - only)</span>
              </>
            )}

            {step === 0 && !sourceOk && sourceType === "csv" && (
              <>
                {" • "}
                <span className="mdmMono">Choose a CSV (or switch to API key)</span>
              </>
            )}
            {step === 1 && !step2Ok && (
              <>
                {" • "}
                <span className="mdmMono">Pick Key + include at least 1 field</span>
              </>
            )}

            {step === 2 && !step3Ok && (
              <>
                {" • "}
                <span className="mdmMono">Select at least 1 match field</span>
              </>
            )}

            {step === 2 && step3Ok && !weightsOk && (
              <>
                {" • "}
                <span className="mdmMono">Total weight must be 100%</span>
              </>
            )}
          </div>


          <div style={{ display: "flex", gap: 10 }}>
            {step === 2 && (
              <label className="mdmTiny" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input
                  className="mdmCheck"
                  type="checkbox"
                  checked={runModelAfterSave}
                  onChange={(e) => setRunModelAfterSave(e.target.checked)}
                  disabled={savingModel}
                />
                Run model
              </label>
            )}

            <button className="mdmBtn mdmBtn--ghost" type="button" onClick={back} disabled={step === 0}>
              Back
            </button>

            {step < 2 ? (
              <button
                className="mdmBtn mdmBtn--primary"
                type="button"
                onClick={next}
                disabled={csvUploading || (step === 0 && !step1Ok) || (step === 1 && !step2Ok)}
              >
                {csvUploading && step === 0 && sourceType === "csv" ? "Uploading..." : "Next"}
              </button>
            ) : (
              <button
                className="mdmBtn mdmBtn--primary"
                type="button"
                onClick={finishWizard}
                disabled={savingModel || !step3Ok || !weightsOk}
              >
                {savingModel ? "Saving..." : "Finish setup"}
              </button>

            )}
          </div>

        </div>
      </div>

      {saveSuccess && (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.55)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10000,
            padding: 16,
          }}
        >
          <div
            style={{
              width: 520,
              maxWidth: "calc(100vw - 32px)",
              background: "#fff",
              borderRadius: 12,
              overflow: "hidden",
              boxShadow: "0 24px 80px rgba(0,0,0,0.35)",
            }}
          >
            <div
              style={{
                background: "var(--coral1)",
                color: "#fff",
                padding: "12px 14px",
                fontWeight: 900,
              }}
            >
              MDM Light
            </div>

            <div style={{ padding: 14 }}>
              <div style={{ marginBottom: 14, lineHeight: 1.35 }}>
                ✅ Your model was {saveSuccess.action}: <span className="mdmMono">{saveSuccess.name}</span>
                {saveSuccess.id != null && String(saveSuccess.id).trim() !== "" ? (
                  <>
                    {" "}
                    (id: <span className="mdmMono">{String(saveSuccess.id)}</span>)
                  </>
                ) : null}
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
                <button
                  className="mdmBtn mdmBtn--primary"
                  type="button"
                  onClick={() => {
                    setSaveSuccess(null);
                    onClose?.();
                  }}
                >
                  OK
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        accept=".csv"
        onChange={onCsvPicked}
        style={{ display: "none" }}
      />

    </div>
  );
}
