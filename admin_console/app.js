const STORAGE_KEY = "vitality.admin_console.azure";
const LIST_PAGE_SIZE = 100;
const LIST_PAGE_LIMIT = 50;

const loginView = document.getElementById("login-view");
const mainView = document.getElementById("main-view");
const filesView = document.getElementById("files-view");
const storesView = document.getElementById("stores-view");
const storesList = document.getElementById("stores-list");
const storeDetail = document.getElementById("store-detail");
const sessionActions = document.getElementById("session-actions");
const endpointLabel = document.getElementById("endpoint-label");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const filesError = document.getElementById("files-error");
const filesLoading = document.getElementById("files-loading");
const filesBody = document.getElementById("files-body");
const filesEmpty = document.getElementById("files-empty");
const filesSummary = document.getElementById("files-summary");
const purposeFilter = document.getElementById("purpose-filter");
const storesError = document.getElementById("stores-error");
const storesLoading = document.getElementById("stores-loading");
const storesBody = document.getElementById("stores-body");
const storesEmpty = document.getElementById("stores-empty");
const storesSummary = document.getElementById("stores-summary");
const storeDetailTitle = document.getElementById("store-detail-title");
const storeFilesError = document.getElementById("store-files-error");
const storeFilesLoading = document.getElementById("store-files-loading");
const storeFilesBody = document.getElementById("store-files-body");
const storeFilesEmpty = document.getElementById("store-files-empty");
const storeFilesSummary = document.getElementById("store-files-summary");
const storeFilesFilter = document.getElementById("store-files-filter");
const backStoresBtn = document.getElementById("back-stores-btn");
const refreshBtn = document.getElementById("refresh-btn");
const logoutBtn = document.getElementById("logout-btn");
const tabButtons = [...document.querySelectorAll(".tab")];

/** @type {"files" | "stores"} */
let currentTab = "files";
/** @type {string | null} */
let selectedStoreId = null;
/** @type {{ id: string, name: string } | null} */
let selectedStoreMeta = null;
/** @type {any[]} */
let cachedStoreFiles = [];
/**
 * fileId -> [{ storeId, storeName, status }]
 * @type {Map<string, Array<{ storeId: string, storeName: string, status: string }>>}
 */
let vsMembershipByFileId = new Map();
let vsScanSummary = "";
/** @type {any[]} */
let cachedFiles = [];

function readCredentials() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed?.endpoint || !parsed?.apiKey) return null;
    return {
      endpoint: String(parsed.endpoint).replace(/\/+$/, ""),
      apiKey: String(parsed.apiKey),
      apiVersion: String(parsed.apiVersion || "2025-04-01-preview"),
    };
  } catch {
    return null;
  }
}

function writeCredentials(credentials) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(credentials));
}

function clearCredentials() {
  localStorage.removeItem(STORAGE_KEY);
}

function showLoginError(message) {
  loginError.hidden = !message;
  loginError.textContent = message || "";
}

function showFilesError(message) {
  filesError.hidden = !message;
  filesError.textContent = message || "";
}

function showStoresError(message) {
  storesError.hidden = !message;
  storesError.textContent = message || "";
}

function showStoreFilesError(message) {
  storeFilesError.hidden = !message;
  storeFilesError.textContent = message || "";
}

function formatBytes(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function formatCreatedAt(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const date = new Date(value * 1000);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
}

function formatFileCounts(counts) {
  if (!counts || typeof counts !== "object") return "—";
  const total = counts.total ?? counts.completed ?? null;
  const completed = counts.completed;
  const inProgress = counts.in_progress;
  const failed = counts.failed;
  const parts = [];
  if (typeof total === "number") parts.push(`${total} total`);
  if (typeof completed === "number") parts.push(`${completed} ok`);
  if (typeof inProgress === "number" && inProgress > 0) parts.push(`${inProgress} pending`);
  if (typeof failed === "number" && failed > 0) parts.push(`${failed} failed`);
  return parts.length ? parts.join(" · ") : "—";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setLoggedIn(credentials) {
  loginView.hidden = true;
  mainView.hidden = false;
  sessionActions.hidden = false;
  endpointLabel.textContent = credentials.endpoint;
  showLoginError("");
  setTab(currentTab);
}

function setLoggedOut() {
  loginView.hidden = false;
  mainView.hidden = true;
  sessionActions.hidden = true;
  endpointLabel.textContent = "";
  filesBody.innerHTML = "";
  filesSummary.textContent = "";
  filesEmpty.hidden = true;
  showFilesError("");
  storesBody.innerHTML = "";
  storesSummary.textContent = "";
  storesEmpty.hidden = true;
  showStoresError("");
  storeFilesBody.innerHTML = "";
  storeFilesSummary.textContent = "";
  storeFilesEmpty.hidden = true;
  showStoreFilesError("");
  selectedStoreId = null;
  selectedStoreMeta = null;
  cachedStoreFiles = [];
  cachedFiles = [];
  vsMembershipByFileId = new Map();
  vsScanSummary = "";
  storeFilesFilter.value = "";
  showStoreList();
}

function setTab(tab) {
  currentTab = tab;
  for (const button of tabButtons) {
    button.setAttribute("aria-selected", button.dataset.tab === tab ? "true" : "false");
  }
  filesView.hidden = tab !== "files";
  storesView.hidden = tab !== "stores";
}

function showStoreList() {
  selectedStoreId = null;
  selectedStoreMeta = null;
  cachedStoreFiles = [];
  storesList.hidden = false;
  storeDetail.hidden = true;
}

function showStoreDetail(store) {
  selectedStoreId = store.id;
  selectedStoreMeta = { id: store.id, name: store.name || store.id };
  storesList.hidden = true;
  storeDetail.hidden = false;
  storeDetailTitle.textContent = store.name || store.id;
  storeFilesFilter.value = "";
}

function formatAzureError(payload, status) {
  if (!payload) return `Request failed (${status})`;
  if (typeof payload.error === "string") return payload.error;
  if (payload.error?.message) return payload.error.message;
  if (payload.message) return payload.message;
  return `Request failed (${status})`;
}

async function azureGet(credentials, path, extraParams = {}) {
  const params = new URLSearchParams({
    "api-version": credentials.apiVersion,
    ...extraParams,
  });
  const url = `${credentials.endpoint}${path}?${params}`;
  let response;
  try {
    response = await fetch(url, {
      headers: {
        "api-key": credentials.apiKey,
      },
    });
  } catch {
    throw new Error(
      "Could not reach Azure from the browser (often CORS). Open DevTools → Network for details.",
    );
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(formatAzureError(payload, response.status));
  }

  return payload;
}

async function fetchFiles(credentials) {
  const purpose = purposeFilter.value.trim();
  const extra = purpose ? { purpose } : {};
  const payload = await azureGet(credentials, "/openai/files", extra);
  return Array.isArray(payload?.data) ? payload.data : [];
}

/** Vector store list endpoints cap at 100 per page and default to 20, so follow the cursor. */
async function azureListAll(credentials, path) {
  const collected = [];
  let after = null;
  for (let page = 0; page < LIST_PAGE_LIMIT; page += 1) {
    const params = { limit: String(LIST_PAGE_SIZE) };
    if (after) params.after = after;
    const payload = await azureGet(credentials, path, params);
    const items = Array.isArray(payload?.data) ? payload.data : [];
    collected.push(...items);
    if (!payload?.has_more || items.length === 0) return collected;
    after = payload.last_id || items[items.length - 1]?.id;
    if (!after) return collected;
  }
  return collected;
}

async function fetchVectorStores(credentials) {
  return azureListAll(credentials, "/openai/vector_stores");
}

async function fetchVectorStoreFiles(credentials, storeId) {
  return azureListAll(credentials, `/openai/vector_stores/${storeId}/files`);
}

/** On a vector_store.file object, `id` is the Files API id; `file_id` only appears on some payloads. */
function resolveAzureFileId(vsFile) {
  if (typeof vsFile?.file_id === "string" && vsFile.file_id) return vsFile.file_id;
  if (typeof vsFile?.id === "string" && vsFile.id) return vsFile.id;
  return "";
}

function paperIdFromAttributes(vsFile) {
  const attrs = vsFile?.attributes;
  if (!attrs || typeof attrs !== "object") return "—";
  if (attrs.paper_id != null) return String(attrs.paper_id);
  return "—";
}

function lastErrorText(vsFile) {
  const err = vsFile?.last_error;
  if (!err) return "—";
  if (typeof err === "string") return err;
  if (err.message) return String(err.message);
  return JSON.stringify(err);
}

function formatMembershipCell(fileId) {
  const entries = vsMembershipByFileId.get(fileId);
  if (!entries || entries.length === 0) {
    return vsScanSummary ? "not attached" : "not checked";
  }
  return entries
    .map((entry) => `${entry.storeName} (${entry.status || "?"})`)
    .join("; ");
}

function renderFiles(files) {
  cachedFiles = files;
  filesBody.innerHTML = "";
  filesEmpty.hidden = files.length > 0;
  const totalBytes = files.reduce(
    (sum, file) => sum + (typeof file.bytes === "number" ? file.bytes : 0),
    0,
  );
  const linked = files.filter((file) => (vsMembershipByFileId.get(file.id) || []).length > 0).length;
  const membershipNote = vsScanSummary
    ? ` · ${linked} linked to a vector store · ${vsScanSummary}`
    : "";
  filesSummary.textContent = `${files.length} file${files.length === 1 ? "" : "s"} · ${formatBytes(totalBytes)} total${membershipNote}`;

  for (const file of files) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="filename" title="${escapeHtml(file.filename || "—")}">${escapeHtml(file.filename || "—")}</td>
      <td class="mono">${escapeHtml(file.id || "—")}</td>
      <td>${escapeHtml(file.purpose || "—")}</td>
      <td>${escapeHtml(file.status || "—")}</td>
      <td>${escapeHtml(formatBytes(file.bytes))}</td>
      <td>${escapeHtml(formatCreatedAt(file.created_at))}</td>
      <td>${escapeHtml(formatMembershipCell(file.id))}</td>
    `;
    filesBody.appendChild(row);
  }
}

function renderStores(stores) {
  storesBody.innerHTML = "";
  storesEmpty.hidden = stores.length > 0;
  storesSummary.textContent = `${stores.length} vector store${stores.length === 1 ? "" : "s"}`;

  for (const store of stores) {
    const row = document.createElement("tr");
    row.className = "clickable-row";
    row.tabIndex = 0;
    row.innerHTML = `
      <td>${escapeHtml(store.name || "—")}</td>
      <td class="mono">${escapeHtml(store.id || "—")}</td>
      <td>${escapeHtml(store.status || "—")}</td>
      <td>${escapeHtml(formatFileCounts(store.file_counts))}</td>
      <td>${escapeHtml(formatBytes(store.usage_bytes))}</td>
      <td>${escapeHtml(formatCreatedAt(store.created_at))}</td>
    `;
    const open = () => {
      showStoreDetail(store);
      void loadStoreFiles();
    };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
    storesBody.appendChild(row);
  }
}

function renderStoreFiles(files, filterText = "") {
  const needle = filterText.trim().toLowerCase();
  const filtered = needle
    ? files.filter((file) => {
        const fileId = resolveAzureFileId(file).toLowerCase();
        const paperId = paperIdFromAttributes(file).toLowerCase();
        const status = String(file.status || "").toLowerCase();
        return fileId.includes(needle) || paperId.includes(needle) || status.includes(needle);
      })
    : files;

  storeFilesBody.innerHTML = "";
  storeFilesEmpty.hidden = filtered.length > 0;
  const completed = files.filter((file) => file.status === "completed").length;
  storeFilesSummary.textContent = `${files.length} file${files.length === 1 ? "" : "s"} · ${completed} completed${
    needle ? ` · showing ${filtered.length}` : ""
  }`;

  for (const file of filtered) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="mono">${escapeHtml(resolveAzureFileId(file) || "—")}</td>
      <td>${escapeHtml(file.status || "—")}</td>
      <td class="mono">${escapeHtml(paperIdFromAttributes(file))}</td>
      <td>${escapeHtml(formatBytes(file.usage_bytes))}</td>
      <td>${escapeHtml(formatCreatedAt(file.created_at))}</td>
      <td>${escapeHtml(lastErrorText(file))}</td>
    `;
    storeFilesBody.appendChild(row);
  }
}

async function loadFiles() {
  const credentials = readCredentials();
  if (!credentials) {
    setLoggedOut();
    return;
  }

  setLoggedIn(credentials);
  filesLoading.hidden = false;
  showFilesError("");
  try {
    const files = await fetchFiles(credentials);
    renderFiles(files);
  } catch (error) {
    filesBody.innerHTML = "";
    filesEmpty.hidden = true;
    filesSummary.textContent = "";
    showFilesError(error instanceof Error ? error.message : "Could not load files");
    filesLoading.hidden = true;
    return;
  }

  try {
    await refreshVsMembership(credentials);
    renderFiles(cachedFiles);
  } catch (error) {
    showFilesError(
      `Files loaded, but vector store membership could not be checked: ${
        error instanceof Error ? error.message : "unknown error"
      }`,
    );
  } finally {
    filesLoading.hidden = true;
  }
}

async function loadStores() {
  const credentials = readCredentials();
  if (!credentials) {
    setLoggedOut();
    return;
  }

  setLoggedIn(credentials);
  showStoreList();
  storesLoading.hidden = false;
  showStoresError("");
  try {
    const stores = await fetchVectorStores(credentials);
    renderStores(stores);
  } catch (error) {
    storesBody.innerHTML = "";
    storesEmpty.hidden = true;
    storesSummary.textContent = "";
    showStoresError(error instanceof Error ? error.message : "Could not load vector stores");
  } finally {
    storesLoading.hidden = true;
  }
}

async function loadStoreFiles() {
  const credentials = readCredentials();
  if (!credentials || !selectedStoreId) {
    return;
  }

  storeFilesLoading.hidden = false;
  showStoreFilesError("");
  try {
    cachedStoreFiles = await fetchVectorStoreFiles(credentials, selectedStoreId);
    renderStoreFiles(cachedStoreFiles, storeFilesFilter.value);
  } catch (error) {
    cachedStoreFiles = [];
    storeFilesBody.innerHTML = "";
    storeFilesEmpty.hidden = true;
    storeFilesSummary.textContent = "";
    showStoreFilesError(error instanceof Error ? error.message : "Could not load store files");
  } finally {
    storeFilesLoading.hidden = true;
  }
}

/** Scan every store's files so the Files table can report where each file is attached. */
async function refreshVsMembership(credentials) {
  const stores = await fetchVectorStores(credentials);
  const membership = new Map();
  const results = await Promise.allSettled(
    stores.map(async (store) => {
      const files = await fetchVectorStoreFiles(credentials, store.id);
      return { store, files };
    }),
  );
  let scannedFiles = 0;
  const failedStores = [];
  for (const [index, result] of results.entries()) {
    if (result.status === "rejected") {
      failedStores.push(stores[index]?.name || stores[index]?.id || "unknown");
      continue;
    }
    const { store, files } = result.value;
    scannedFiles += files.length;
    for (const vsFile of files) {
      const fileId = resolveAzureFileId(vsFile);
      if (!fileId) continue;
      const list = membership.get(fileId) || [];
      list.push({
        storeId: store.id,
        storeName: store.name || store.id,
        status: vsFile.status || "unknown",
      });
      membership.set(fileId, list);
    }
  }
  vsMembershipByFileId = membership;
  vsScanSummary = `checked ${stores.length} store${stores.length === 1 ? "" : "s"} · ${scannedFiles} attached file${scannedFiles === 1 ? "" : "s"}`;
  if (failedStores.length > 0) {
    throw new Error(`could not read files for ${failedStores.join(", ")}`);
  }
}

async function refreshCurrent() {
  if (currentTab === "files") {
    await loadFiles();
    return;
  }
  if (selectedStoreId) {
    await loadStoreFiles();
    return;
  }
  await loadStores();
}

for (const button of tabButtons) {
  button.addEventListener("click", () => {
    const tab = button.dataset.tab === "stores" ? "stores" : "files";
    setTab(tab);
    if (tab === "files") {
      void loadFiles();
    } else if (selectedStoreId) {
      void loadStoreFiles();
    } else {
      void loadStores();
    }
  });
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const endpoint = document.getElementById("endpoint-input").value.trim().replace(/\/+$/, "");
  const apiKey = document.getElementById("api-key-input").value.trim();
  const apiVersion =
    document.getElementById("api-version-input").value.trim() || "2025-04-01-preview";

  if (!endpoint || !apiKey) {
    showLoginError("Endpoint and API key are required.");
    return;
  }

  const credentials = { endpoint, apiKey, apiVersion };
  writeCredentials(credentials);
  currentTab = "files";
  await loadFiles();
});

refreshBtn.addEventListener("click", () => {
  void refreshCurrent();
});

purposeFilter.addEventListener("change", () => {
  void loadFiles();
});

backStoresBtn.addEventListener("click", () => {
  void loadStores();
});

storeFilesFilter.addEventListener("input", () => {
  renderStoreFiles(cachedStoreFiles, storeFilesFilter.value);
});

logoutBtn.addEventListener("click", () => {
  clearCredentials();
  loginForm.reset();
  document.getElementById("api-version-input").value = "2025-04-01-preview";
  setLoggedOut();
});

const existing = readCredentials();
if (existing) {
  document.getElementById("endpoint-input").value = existing.endpoint;
  document.getElementById("api-key-input").value = existing.apiKey;
  document.getElementById("api-version-input").value = existing.apiVersion;
  void loadFiles();
} else {
  setLoggedOut();
}
