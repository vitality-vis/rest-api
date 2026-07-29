const STORAGE_KEY = "vitality.admin_console.azure";

const loginView = document.getElementById("login-view");
const filesView = document.getElementById("files-view");
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
const refreshBtn = document.getElementById("refresh-btn");
const logoutBtn = document.getElementById("logout-btn");

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

function setLoggedIn(credentials) {
  loginView.hidden = true;
  filesView.hidden = false;
  sessionActions.hidden = false;
  endpointLabel.textContent = credentials.endpoint;
  showLoginError("");
}

function setLoggedOut() {
  loginView.hidden = false;
  filesView.hidden = true;
  sessionActions.hidden = true;
  endpointLabel.textContent = "";
  filesBody.innerHTML = "";
  filesSummary.textContent = "";
  filesEmpty.hidden = true;
  showFilesError("");
}

function formatAzureError(payload, status) {
  if (!payload) return `Request failed (${status})`;
  if (typeof payload.error === "string") return payload.error;
  if (payload.error?.message) return payload.error.message;
  if (payload.message) return payload.message;
  return `Request failed (${status})`;
}

async function fetchFiles(credentials) {
  const params = new URLSearchParams({ "api-version": credentials.apiVersion });
  const purpose = purposeFilter.value.trim();
  if (purpose) params.set("purpose", purpose);

  const url = `${credentials.endpoint}/openai/files?${params}`;
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

  return Array.isArray(payload?.data) ? payload.data : [];
}

function renderFiles(files) {
  filesBody.innerHTML = "";
  filesEmpty.hidden = files.length > 0;
  const totalBytes = files.reduce(
    (sum, file) => sum + (typeof file.bytes === "number" ? file.bytes : 0),
    0,
  );
  filesSummary.textContent = `${files.length} file${files.length === 1 ? "" : "s"} · ${formatBytes(totalBytes)} total`;

  for (const file of files) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(file.filename || "—")}</td>
      <td class="mono">${escapeHtml(file.id || "—")}</td>
      <td>${escapeHtml(file.purpose || "—")}</td>
      <td>${escapeHtml(file.status || "—")}</td>
      <td>${escapeHtml(formatBytes(file.bytes))}</td>
      <td>${escapeHtml(formatCreatedAt(file.created_at))}</td>
    `;
    filesBody.appendChild(row);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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
  } finally {
    filesLoading.hidden = true;
  }
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
  await loadFiles();
});

refreshBtn.addEventListener("click", () => {
  void loadFiles();
});

purposeFilter.addEventListener("change", () => {
  void loadFiles();
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
