const RULES = [
  { value: "recency", label: "Recency (latest wins)" },
  { value: "system", label: "System priority" },
  { value: "created_at", label: "Created date" },
  { value: "updated_at", label: "Updated date" },
  { value: "created_by", label: "Created by (user)" },
  { value: "updated_by", label: "Updated by (user)" },
];

const TYPES = ["text", "number", "date", "email", "phone", "id"];

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function pick(arr) {
  return arr[randInt(0, arr.length - 1)];
}

function uid(prefix = "id") {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

export function getRuleOptions() {
  return RULES;
}

export function createInitialState() {
  const fields = [
    { name: "source_id", type: "id", include: true, isKey: true },
    { name: "first_name", type: "text", include: true, isKey: false },
    { name: "last_name", type: "text", include: true, isKey: false },
    { name: "email", type: "email", include: true, isKey: false },
    { name: "phone", type: "phone", include: true, isKey: false },
    { name: "address", type: "text", include: true, isKey: false },
    { name: "city", type: "text", include: true, isKey: false },
    { name: "state", type: "text", include: true, isKey: false },
    { name: "zip", type: "number", include: true, isKey: false },
    { name: "created_at", type: "date", include: false, isKey: false },
    { name: "updated_at", type: "date", include: false, isKey: false },
    { name: "source_system", type: "text", include: false, isKey: false },
  ];

  const weights = {};
  const survivorship = {};
  for (const f of fields) {
    weights[f.name] = f.isKey ? 1.0 : clamp((randInt(10, 60) / 100), 0.05, 1);
    survivorship[f.name] = f.isKey ? "system" : "recency";
  }

  const recordCount = 12450;

  return {
    source: {
      fileName: "customers.csv",
      recordCount,
      fields,
    },
    model: {
      matchThreshold: 0.85,
      possibleThreshold: 0.70,
      weights,
      survivorshipMode: "simple", // simple | advanced
      survivorship,
      systemPriority: ["CRM", "ERP", "Web", "Partners"],
    },
    routing: {
      useAI: true,
      notifyEmail: "",
    },
    run: null,
    exceptions: makeExceptions(recordCount),
    stewardship: {
      reviewed: 0,
      resolved: 0,
      approved: 0,
    },
  };
}

export function addNewField(state) {
  const idx = state.source.fields.filter((f) => f.name.startsWith("field_")).length + 1;
  const name = `field_${idx}`;
  const type = pick(TYPES);

  const next = structuredClone(state);
  next.source.fields.push({ name, type, include: true, isKey: false });
  next.model.weights[name] = clamp(randInt(10, 50) / 100, 0.05, 1);
  next.model.survivorship[name] = "recency";
  return next;
}

export function makeRunFromState(state) {
  const total = state.source.recordCount || 0;

  // basic, believable distribution
  const matched = Math.round(total * (randInt(62, 82) / 100));
  const exceptions = Math.round(total * (randInt(4, 12) / 100));
  const notMatched = Math.max(0, total - matched - exceptions);

  const precision = clamp(randInt(86, 96) / 100, 0, 1);
  const recall = clamp(randInt(78, 92) / 100, 0, 1);

  return {
    runId: uid("run"),
    startedAt: new Date().toISOString(),
    totals: { total, matched, exceptions, notMatched },
    quality: { precision, recall },
  };
}

export function makeExceptions(recordCount) {
  const n = clamp(Math.round(recordCount * 0.006), 18, 60);

  const reasons = [
    "Conflicting email",
    "Phone mismatch",
    "Address variance",
    "Name similarity borderline",
    "Missing key attributes",
    "Potential household match",
  ];

  const statuses = ["open", "in_review", "resolved", "approved"];

  const items = [];
  for (let i = 0; i < n; i++) {
    const score = clamp((randInt(62, 88) / 100), 0, 1);
    const reason = pick(reasons);
    const status = i < Math.round(n * 0.75) ? pick(statuses.slice(0, 2)) : pick(statuses.slice(2));

    items.push({
      id: uid("ex"),
      left: `SRC-${randInt(10000, 99999)}`,
      right: `SRC-${randInt(10000, 99999)}`,
      score,
      reason,
      status,
      updatedAt: new Date(Date.now() - randInt(1, 48) * 3600 * 1000).toISOString(),
    });
  }

  // keep open-ish items at top
  items.sort((a, b) => {
    const rank = (s) => (s === "open" ? 0 : s === "in_review" ? 1 : s === "resolved" ? 2 : 3);
    const ra = rank(a.status);
    const rb = rank(b.status);
    if (ra !== rb) return ra - rb;
    return b.score - a.score;
  });

  return items;
}

export function formatInt(n) {
  return new Intl.NumberFormat().format(n);
}

export function pct(n) {
  return `${Math.round(n * 100)}%`;
}

export function clamp01(n) {
  return clamp(n, 0, 1);
}
