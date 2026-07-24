"use strict";

const $ = (id) => document.getElementById(id);
let staged = [];      // [{id, name, pages}]
let pollTimer = null;
let activeJobId = null;  // the running job, for the Cancel button

// Single always-present live region: route transient status (key saved/cleared,
// estimate ready, job done/failed, errors) through here so screen-reader users
// hear the outcome reliably, even when a per-section box toggles `hidden`.
let _announceTimer = null;
function announce(msg) {
  const el = $("a11y-status");
  if (!el) return;
  // Clear, then set after a tick, so repeating an identical message still
  // re-announces (a live region set to the same text is ignored).
  el.textContent = "";
  clearTimeout(_announceTimer);
  _announceTimer = setTimeout(() => { el.textContent = msg; }, 60);
}

async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

// ---- settings ----------------------------------------------------------- //
const NUMERIC = ["temperature", "thinking_budget", "pdf_dpi", "max_dimension", "word_confidence"];
const TEXT = ["thinking_level", "media_resolution", "tesseract_language"];
const BOOL = ["preprocess", "refine_word_boxes"];

function gatherSettings() {
  const s = {};
  for (const id of TEXT) s[id] = $(id).value;
  for (const id of NUMERIC) {
    const v = parseFloat($(id).value);
    if (!Number.isNaN(v)) s[id] = v;
  }
  for (const id of BOOL) s[id] = $(id).checked;
  // One picker drives both models: detection (two-pass) follows transcription.
  const model = $("model").value;
  if (model) { s.transcription_model = model; s.detection_model = model; }
  const mode = document.querySelector("input[name=mode]:checked");
  if (mode) s.mode = mode.value;
  const ct = document.querySelector("input[name=content_type]:checked");
  if (ct) s.content_type = ct.value;
  return s;
}

function applyKeyStatus(data) {
  const badge = $("key-status");
  const info = $("key-info");
  info.hidden = false;
  if (data.api_key_set) {
    badge.textContent = "Key saved"; badge.className = "badge ok";
    info.className = "key-info";
    const where = data.api_key_source === "env"
      ? `from <span class="mono">GEMINI_API_KEY</span> environment variable`
      : "stored locally on this machine";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">✓</span><span>Gemini key ${where}: <span class="mono">${escapeHtml(data.api_key_hint || "")}</span></span>`;
  } else {
    badge.textContent = "No API key"; badge.className = "badge warn";
    info.className = "key-info warn";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">!</span><span>No Gemini key stored. Paste one above or set <span class="mono">GEMINI_API_KEY</span>.</span>`;
  }
}

// Free, proactive check that a stored key still works. ListModels costs no
// generation tokens/quota, so a revoked/expired key is caught here in Settings
// instead of failing mid-transcription. Only "valid"/"invalid" change the
// badge; "unknown" (offline/transient) and "no_key" leave it untouched.
async function verifyKey(announceValid = false) {
  let data;
  try { data = await api("GET", "/api/key-status"); }
  catch (e) { return; } // our own server unreachable — leave the local badge
  const badge = $("key-status");
  const info = $("key-info");
  const key = $("api_key");
  if (data.state === "valid") {
    badge.textContent = "Key valid"; badge.className = "badge ok";
    if (key) key.removeAttribute("aria-invalid");
    if (announceValid) announce("API key verified and working.");
  } else if (data.state === "invalid") {
    badge.textContent = "Key rejected"; badge.className = "badge warn";
    info.className = "key-info warn";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">!</span><span>${escapeHtml(data.message)} Transcription will fail until you paste a current key.</span>`;
    // Flag the field for assistive tech and announce the rejection (this also
    // fires on load for a bad stored key, which the user needs to know).
    if (key) key.setAttribute("aria-invalid", "true");
    announce("API key rejected. " + data.message);
  }
}

async function loadSettings() {
  const s = await api("GET", "/api/settings");
  for (const id of [...TEXT, ...NUMERIC]) if (s[id] !== undefined) $(id).value = s[id];
  for (const id of BOOL) if (s[id] !== undefined) $(id).checked = s[id];
  // Model dropdown (options populated by loadModelCatalog before this runs).
  const sel = $("model");
  if (s.transcription_model) sel.value = s.transcription_model;
  if (!sel.value && sel.options.length) {
    // A saved model that isn't in the curated catalog -> fall back to the first
    // (and persist it so the UI and the backend agree on what will run).
    sel.value = sel.options[0].value;
    saveSettings(gatherSettings());
  }
  updateModelPricingHint();
  const radio = document.querySelector(`input[name=mode][value="${s.mode}"]`);
  if (radio) radio.checked = true;
  const ctRadio = document.querySelector(`input[name=content_type][value="${s.content_type}"]`);
  if (ctRadio) ctRadio.checked = true;
  applyKeyStatus(s);
  verifyKey(); // background-verify the stored key (no-op when no key set)
}

async function loadTesseractStatus() {
  let data;
  try {
    data = await api("GET", "/api/tesseract");
  } catch (e) {
    return;
  }
  const info = $("tesseract-info");
  if (info) {
    if (data.available) {
      // Tesseract ships bundled with the app, so confirming "it's installed" is
      // just noise. Stay silent when it works; only speak up when it's missing
      // (Printed-only and word-box refinement genuinely won't run without it).
      info.hidden = true;
      info.innerHTML = "";
    } else {
      info.hidden = false;
      info.className = "key-info warn";
      let detail;
      if (data.wrapper_present === false) {
        detail = `The <span class="mono">pytesseract</span> Python package is missing. Reinstall CursBreaker (<span class="mono">pip install .</span>) and restart.`;
      } else {
        const hint = data.install_hint ? escapeHtml(data.install_hint) + " " : "";
        // No-admin path: a portable build dropped into the app-managed folder.
        const portable = data.managed_dir
          ? ` No admin rights? Unzip a portable Tesseract into <span class="mono">${escapeHtml(data.managed_dir)}</span> (binary + a <span class="mono">tessdata</span> subfolder).`
          : "";
        detail = `Tesseract OCR engine not detected. ${hint}${portable} Or set <span class="mono">TESSERACT_CMD</span> to the full path of the executable and restart.`;
      }
      info.innerHTML =
        `<span class="glyph" aria-hidden="true">!</span><span>${detail} Printed-only mode and word-box refinement need it; Handwriting mode still works as usual.</span>`;
      announce("Tesseract OCR engine not detected. Printed-only mode and word-box refinement are unavailable; handwriting mode still works.");
    }
  }
  // Populate the language datalist with whatever's actually installed.
  const dl = $("tesseract-langs");
  if (dl) {
    dl.innerHTML = "";
    for (const code of data.languages || []) {
      const opt = document.createElement("option");
      opt.value = code;
      dl.appendChild(opt);
    }
  }
}

async function saveSettings(partial) {
  const data = await api("POST", "/api/settings", partial);
  applyKeyStatus(data);
}

// Curated model catalog (with published prices) backing the dropdown + the
// automatic cost estimate. Populated once from the backend.
let modelCatalog = [];
let pricesAsOf = "";

async function loadModelCatalog() {
  try {
    const data = await api("GET", "/api/models");
    modelCatalog = data.models || [];
    pricesAsOf = data.prices_as_of || "";
    const sel = $("model");
    sel.innerHTML = "";
    for (const m of modelCatalog) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      sel.appendChild(opt);
    }
  } catch (e) { /* dropdown stays empty; settings still load */ }
}

function modelInfo(id) {
  return modelCatalog.find((m) => m.id === id) || null;
}

// Show the selected model's published price right under the picker, so the cost
// basis is visible before anyone runs anything -- and clearly an estimate.
function updateModelPricingHint() {
  const hint = $("model-pricing");
  if (!hint) return;
  const m = modelInfo($("model").value);
  if (!m) { hint.textContent = ""; return; }
  let rates = `$${m.input_per_mtok.toFixed(2)}/1M input, $${m.output_per_mtok.toFixed(2)}/1M output`;
  if (m.tier_threshold) {
    rates += ` for prompts &le;${formatTokens(m.tier_threshold)} tokens `
      + `($${m.input_per_mtok_high.toFixed(2)}/$${m.output_per_mtok_high.toFixed(2)} above)`;
  }
  hint.innerHTML =
    `<b>${escapeHtml(m.label)}</b>: ${rates}.`
    + (pricesAsOf ? ` Prices as of ${escapeHtml(pricesAsOf)}, used for the cost estimate (an estimate, not a guarantee).` : "")
    + ` <a href="${PRICING_URL}" target="_blank" rel="noopener noreferrer">Live pricing</a>.`;
}

// ---- upload / staging --------------------------------------------------- //
// Uploads are local (browser -> the bundled server on 127.0.0.1), so the time
// here is copying bytes to disk and reading page counts -- never the Gemini API.
// For big jobs (e.g. a 10 GB book) that's slow enough to need real feedback, so
// we upload in bounded batches over XHR (fetch can't report upload progress)
// and show a live byte + file count. These helpers are pure and top-level so the
// Node test harness can exercise them directly.

// Mirror the server's accepted types (images.SUPPORTED_EXT). The browse dialog
// already constrains itself via the input's `accept` list, but drag-and-drop
// does not -- so we filter here too. Without it, an all-unsupported batch (e.g.
// 20 stray .docx dropped ahead of a .png) would 400 and abort the rest of the
// upload; filtering first means valid files always go, and the byte/file totals
// reflect only what will actually upload. The server still re-checks.
const SUPPORTED_EXT = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif", ".pdf"];
function isSupportedFile(f) {
  const name = ((f && f.name) || "").toLowerCase();
  return SUPPORTED_EXT.some((ext) => name.endsWith(ext));
}

// Bound each POST by BOTH a file count and a byte size, so the server holds only
// a batch at a time, each request returns quickly (keeping the heartbeat alive
// and the staged list filling steadily), and one failed batch never sinks the
// rest. A single file bigger than the byte cap still gets its own batch.
const UPLOAD_MAX_FILES = 20;
const UPLOAD_MAX_BYTES = 256 * 1024 * 1024;  // 256 MB
function planUploadBatches(files, maxFiles = UPLOAD_MAX_FILES, maxBytes = UPLOAD_MAX_BYTES) {
  const batches = [];
  let cur = [], curBytes = 0;
  for (const f of files) {
    const size = Number(f.size) || 0;
    if (cur.length && (cur.length >= maxFiles || curBytes + size > maxBytes)) {
      batches.push(cur); cur = []; curBytes = 0;
    }
    cur.push(f); curBytes += size;
  }
  if (cur.length) batches.push(cur);
  return batches;
}

// Human-readable byte size: "812 B", "3.4 MB", "9.7 GB".
function formatBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return n + " B";
  const units = ["KB", "MB", "GB", "TB"];
  let i = -1;
  do { n /= 1024; i++; } while (n >= 1024 && i < units.length - 1);
  return (n < 10 ? n.toFixed(1) : Math.round(n)) + " " + units[i];
}

// The status line during an upload. Two phases per batch: "Uploading" while
// bytes move, then "Saving…" once they've all arrived but the server is still
// flushing them to disk -- the part that, unlabelled, looks like a freeze.
function uploadStatusText({ sentBytes, totalBytes, filesDone, filesTotal, saving }) {
  const pct = totalBytes ? Math.min(100, Math.round((sentBytes / totalBytes) * 100)) : 0;
  const head = saving ? "Saving…" : `Uploading… ${pct}%`;
  const size = totalBytes ? ` (${formatBytes(sentBytes)} of ${formatBytes(totalBytes)})` : "";
  const files = filesTotal ? ` · ${filesDone}/${filesTotal} file(s) ready` : "";
  return head + size + files;
}

// POST one batch via XMLHttpRequest so we can watch the body upload (fetch
// can't). onProgress(loadedBytes) ticks during the send; onUploaded() fires the
// moment the body is fully sent (server now saving to disk); resolves with the
// parsed {files:[...]} JSON.
function uploadBatchXHR(batchFiles, onProgress, onUploaded) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of batchFiles) fd.append("files", f);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    if (xhr.upload) {
      xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded); };
      xhr.upload.onload = () => { if (onUploaded) onUploaded(); };
    }
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || xhr.statusText || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.send(fd);
  });
}

async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const note = $("action-note");
  // Drop unsupported files up front (drag-and-drop bypasses the input's accept
  // filter), so a stray .docx can't form an all-unsupported batch that 400s and
  // strands the valid files behind it.
  const all = Array.from(fileList);
  const files = all.filter(isSupportedFile);
  const skipped = all.length - files.length;
  if (!files.length) {
    note.textContent = `No supported files. Allowed: ${SUPPORTED_EXT.join(", ")}`;
    announce("No supported files to upload. Allowed types: " + SUPPORTED_EXT.join(", "));
    return;
  }
  const skippedNote = skipped ? ` · skipped ${skipped} unsupported file(s)` : "";
  const totalBytes = files.reduce((s, f) => s + (Number(f.size) || 0), 0);
  const filesTotal = files.length;
  const batches = planUploadBatches(files);
  let sentBase = 0;   // bytes from already-completed batches
  let filesDone = 0;  // files successfully staged so far
  const show = (loaded, saving) =>
    (note.textContent = uploadStatusText({
      sentBytes: sentBase + Math.min(loaded, totalBytes - sentBase),
      totalBytes, filesDone, filesTotal, saving,
    }));
  announce(`Uploading ${filesTotal} file(s), ${formatBytes(totalBytes)}.`);
  show(0, false);
  try {
    for (const batch of batches) {
      const batchBytes = batch.reduce((s, f) => s + (Number(f.size) || 0), 0);
      const data = await uploadBatchXHR(
        batch,
        (loaded) => show(loaded, false),
        () => show(batchBytes, true),  // body fully sent -> server is saving it
      );
      sentBase += batchBytes;
      const accepted = (data && data.files) || [];
      staged.push(...accepted);
      filesDone += accepted.length;
      // Surface staged files as each batch lands. renderStaged() resets the note
      // to its resting summary, so re-apply progress while more batches remain.
      renderStaged();
      if (sentBase < totalBytes) show(0, false);
    }
    note.textContent = stagedStatus() + skippedNote;
    announce(`${filesDone} file(s) ready to transcribe.` +
      (skipped ? ` Skipped ${skipped} unsupported file(s).` : ""));
  } catch (e) {
    renderStaged();  // keep whatever staged before the failure
    const partial = filesDone ? ` (${filesDone} file(s) staged first)` : "";
    note.textContent = "Upload failed: " + e.message + partial;
    announce("Upload failed: " + e.message);
  }
  // Fill in page counts (computed lazily server-side) for everything now staged.
  pollStagedPages();
}

// The action note doubles as a transient status/error line. This is its
// resting state -- a summary of what's staged -- restored whenever a transient
// message (e.g. a prior "no API key" error) should be cleared. The total page
// count is appended once every staged file has been counted (it's computed in
// the background, server-side); until then we show just the file count, so the
// number simply appears when ready rather than churning per file.
function stagedStatus(list = staged) {
  if (!list.length) return "";
  const base = `${list.length} file(s) ready`;
  if (pendingPageCounts(list)) return base;
  const pages = list.reduce((n, f) => n + (f.pages || 0), 0);
  return `${base} · ${pages.toLocaleString()} page(s)`;
}

// A job in flight must keep Transcribe/Estimate disabled even if something that
// would normally re-enable them runs mid-job (a staged-list re-render, or a
// previously-issued cost estimate returning). Those buttons do nothing useful
// while work is underway, so they stay greyed until the job ends.
function jobRunning() { return activeJobId != null; }

function renderStaged() {
  const ul = $("staged");
  ul.innerHTML = "";
  for (const f of staged) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHtml(f.name)}</span>`;
    const rm = document.createElement("button");
    rm.className = "rm"; rm.type = "button"; rm.textContent = "×";
    rm.title = "Remove " + f.name;
    rm.setAttribute("aria-label", "Remove " + f.name);  // "×" alone says nothing
    rm.onclick = () => { staged = staged.filter((x) => x.id !== f.id); renderStaged(); };
    li.appendChild(rm);
    ul.appendChild(li);
  }
  $("transcribe").disabled = staged.length === 0 || jobRunning();
  $("estimate").disabled = staged.length === 0 || jobRunning();
  // Any change to the file set invalidates a prior cost estimate.
  const est = $("estimate-info");
  if (est) { est.hidden = true; est.innerHTML = ""; }
  $("action-note").textContent = stagedStatus();
  // Removing files can change how many already-searchable PDFs remain.
  updateSkipOverlayControl();
  updateClearButton();
}

// ---- clear the batch ---------------------------------------------------- //

// "Clear batch" appears whenever there's something to clear -- staged files, a
// run in progress, or a results set on screen -- and is disabled while a job is
// running (clearing mid-run would pull the workspace out from under the worker).
function updateClearButton() {
  const btn = $("clear-batch");
  if (!btn) return;
  const showing = staged.length > 0
    || !$("results-card").hidden
    || !$("progress-card").hidden;
  btn.hidden = !showing;
  btn.disabled = jobRunning();
}

// Wipe the current batch server-side (staged uploads + generated outputs) and
// reset the page to a clean slate so the next batch has room. The user's own
// on-disk originals (files staged by path) are only unstaged, never deleted.
async function clearBatch() {
  if (jobRunning()) return;  // guarded by the disabled button too
  const hadResults = !$("results-card").hidden;
  const ok = confirm(
    hadResults
      ? "Clear this batch and its results?\n\nFiles you've already downloaded are "
        + "kept. Staged files and the generated outputs are removed from the server."
      : "Clear the staged files?\n\nThey're removed from the server. Your original "
        + "files on disk are left untouched."
  );
  if (!ok) return;
  let freed = "";
  try {
    const r = await api("POST", "/api/clear");
    if (r && r.freed_bytes) freed = " Freed " + formatBytes(r.freed_bytes) + ".";
  } catch (e) {
    $("action-note").textContent = "Couldn't clear: " + e.message;
    return;
  }
  // Reset all batch state + UI to the clean, nothing-staged starting point.
  staged = [];
  renderStaged();                 // clears the list; resets the note + buttons
  $("progress-card").hidden = true;
  $("results-card").hidden = true;
  $("results").innerHTML = "";
  const rt = $("results-tokens"); if (rt) { rt.hidden = true; rt.textContent = ""; }
  const dn = $("download-note"); if (dn) { dn.hidden = true; dn.textContent = ""; }
  const tt = $("token-text"); if (tt) tt.textContent = "";
  const log = $("activity-log"); if (log) { log.replaceChildren(); log.dataset.shown = "0"; }
  const live = $("activity-live"); if (live) live.textContent = "";
  const pause = $("pause-banner"); if (pause) pause.hidden = true;
  $("progress-bar").style.width = "0%";
  $("progress").setAttribute("aria-valuenow", "0");
  // Set the note after renderStaged (which resets it to the empty summary).
  $("action-note").textContent = "Batch cleared." + freed;
  announce("Batch cleared." + freed);
  updateClearButton();
}

// ---- lazy page counts --------------------------------------------------- //
// The server counts pages in the background after staging; we poll for the
// numbers and, once every file is counted, append the total to the staged
// summary (no per-file pills -- the figure just appears). Cosmetic only:
// Transcribe/Estimate don't wait on it and recount independently.
let stagedPagesTimer = null;

function pendingPageCounts(list) {
  return list.some((f) => f.pages === null || f.pages === undefined);
}

// Merge server counts into the staged model; returns true if anything changed.
function applyStagedPages(list, pages) {
  let changed = false;
  for (const f of list) {
    if ((f.pages === null || f.pages === undefined) && typeof pages[f.id] === "number") {
      f.pages = pages[f.id];
      changed = true;
    }
  }
  return changed;
}

// Merge the server's "already has a searchable text layer" flags into the staged
// model (computed in the same background pass as the page counts).
function applyStagedTextLayers(list, textLayers) {
  if (!textLayers) return;
  for (const f of list) {
    if ((f.hasText === null || f.hasText === undefined)
        && typeof textLayers[f.id] === "boolean") {
      f.hasText = textLayers[f.id];
    }
  }
}

// The batch-level "skip our overlay for already-searchable PDFs" toggle is shown
// only when some staged files already carry a text layer AND a searchable PDF
// will actually be built (it only affects that output). One decision covers the
// whole batch -- never per file or per page.
function updateSkipOverlayControl() {
  const wrap = $("skip-overlay-wrap");
  if (!wrap) return;
  const n = staged.filter((f) => f.hasText === true).length;
  const outs = selectedOutputs();
  const pdfInPlay = outs.length === 0 || outs.includes("pdf");
  if (n > 0 && pdfInPlay) {
    const label = $("skip-overlay-label");
    if (label) {
      label.textContent =
        `${n} file(s) already contain searchable text — skip adding CursBreaker's `
        + `PDF overlay for them, keeping each original's existing text layer.`;
    }
    wrap.hidden = false;
  } else {
    wrap.hidden = true;
  }
}

// Whether the user has actively opted to skip the overlay: only counts when the
// control is visible (relevant) and ticked.
function skipOverlayActive() {
  const wrap = $("skip-overlay-wrap");
  const cb = $("skip-overlay");
  return !!(wrap && !wrap.hidden && cb && cb.checked);
}

function pollStagedPages() {
  clearInterval(stagedPagesTimer);
  if (!pendingPageCounts(staged)) return;
  let ticks = 0;
  const tick = async () => {
    let data;
    try { data = await api("GET", "/api/staged-pages"); }
    catch (e) { return; }  // transient; keep polling (the cap still bounds us)
    // Surface the total only once every file has been counted, and only by
    // refreshing the resting summary -- don't clobber a transient message or the
    // estimate mid-count. The number then just appears.
    if (applyStagedPages(staged, data.pages || {}) && !pendingPageCounts(staged)) {
      $("action-note").textContent = stagedStatus();
    }
    // Existing-text-layer flags land in the same pass; reveal the skip toggle
    // once any already-searchable files are known.
    applyStagedTextLayers(staged, data.text_layers);
    updateSkipOverlayControl();
    // Stop once every file has a number, or after a generous cap so a lost
    // server (e.g. a restart) can't leave us polling forever.
    if (!pendingPageCounts(staged) || ++ticks > 600) clearInterval(stagedPagesTimer);
  };
  tick();
  stagedPagesTimer = setInterval(tick, 700);
}

// ---- processing --------------------------------------------------------- //

// Output kinds ticked in the Documents card; [] means "create everything" (the
// server treats an empty/omitted list as all formats).
const OUTPUT_FORMAT_IDS = ["txt", "hocr", "alto", "pdf"];
function selectedOutputs() {
  return OUTPUT_FORMAT_IDS.filter((t) => { const el = $("out-" + t); return el && el.checked; });
}

async function transcribe() {
  if (!staged.length) return;
  // Drop any leftover error (e.g. a prior "no API key") so it can't linger
  // through this run.
  $("action-note").textContent = stagedStatus();
  // Free up screen space the moment work starts so progress + results land
  // above the fold on smaller laptops.
  setSettingsOpen(false);  // collapse Settings; remembers the collapsed state
  $("transcribe").disabled = true;
  $("estimate").disabled = true;
  const est = $("estimate-info");
  if (est) { est.hidden = true; est.innerHTML = ""; }
  $("token-text").textContent = "";
  // Reset the activity log + bar for a fresh run.
  const logReset = $("activity-log");
  if (logReset) { logReset.replaceChildren(); logReset.dataset.shown = "0"; }
  const liveReset = $("activity-live"); if (liveReset) liveReset.textContent = "";
  const pauseReset = $("pause-banner"); if (pauseReset) pauseReset.hidden = true;
  $("progress-bar").style.width = "0%";
  $("progress").setAttribute("aria-valuenow", "0");
  $("results-card").hidden = true;
  $("results").innerHTML = "";
  try {
    const { job_id } = await api("POST", "/api/process", {
      file_ids: staged.map((f) => f.id),
      outputs: selectedOutputs(),
      skip_existing_text_overlay: skipOverlayActive(),
    });
    activeJobId = job_id;
    const cancel = $("cancel-job");
    if (cancel) { cancel.hidden = false; cancel.disabled = false; cancel.textContent = "Cancel"; }
    $("progress-card").hidden = false;
    updateClearButton();  // now a job is running -> shown but disabled
    pollJob(job_id);
  } catch (e) {
    $("action-note").textContent = "Error: " + e.message;
    $("transcribe").disabled = false;
  }
}

async function cancelJob() {
  if (!activeJobId) return;
  const btn = $("cancel-job");
  if (btn) { btn.disabled = true; btn.textContent = "Cancelling…"; }
  try {
    await api("POST", `/api/jobs/${activeJobId}/cancel`);
  } catch (e) {
    // Re-enable so the user can retry if the request itself failed.
    if (btn) { btn.disabled = false; btn.textContent = "Cancel"; }
  }
}

// Disk-full pause controls. resumeJob retries the file that couldn't be saved
// (the user has freed space); endJob stops and keeps everything already saved.
async function resumeJob() {
  if (!activeJobId) return;
  const btn = $("resume-job");
  if (btn) { btn.disabled = true; btn.textContent = "Resuming…"; }
  try {
    await api("POST", `/api/jobs/${activeJobId}/resume`);
    announce("Resuming — retrying the file that ran out of space.");
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = "I've freed up space — Resume"; }
  }
}

async function endJob() {
  if (!activeJobId) return;
  const btn = $("stop-job");
  if (btn) btn.disabled = true;
  try {
    await api("POST", `/api/jobs/${activeJobId}/end`);
    announce("Stopping — keeping the files already finished.");
  } catch (e) {
    if (btn) btn.disabled = false;
  }
}

// Drive the page-driven bar + the verbose activity log from a job payload.
// Top-level so the Node test harness can exercise it directly with a fake DOM.
function renderProgress(job) {
  // Bar fills by pages completed across the whole job; clamp, and force 100%
  // once finished so it always reads full on completion.
  const total = Number(job.total_units || 0);
  const done = Number(job.done_units || 0);
  let pct = total ? Math.round((done / total) * 100) : 0;
  pct = Math.max(0, Math.min(100, pct));
  if (job.status === "done") pct = 100;
  $("progress-bar").style.width = pct + "%";
  $("progress").setAttribute("aria-valuenow", String(pct));

  // Concise headline above the log. (Cancelled keeps its partial bar fraction —
  // only "done" forces 100%.)
  const okCount = (job.results || []).filter((r) => !r.error).length;
  $("progress-text").textContent =
    job.status === "error" ? "Error: " + job.error
    : job.status === "stopped" ? `Stopped — out of disk space · ${okCount} file(s) saved`
    : job.status === "cancelled" ? `Cancelled — ${(job.results || []).length} file(s) completed`
    : job.status === "done" ? `Done — ${(job.results || []).length} file(s)`
    : job.paused ? `Paused — no disk space · page ${done}/${total}`
    : total ? `Processing — page ${done}/${total}`
    : "Processing…";

  // Disk-full pause banner. The worker is blocked until the user picks Resume or
  // Stop, so nothing else is being written or billed while this is up.
  const pauseEl = $("pause-banner");
  if (pauseEl) {
    const paused = !!job.paused;
    pauseEl.hidden = !paused;
    if (paused) {
      const reason = $("pause-reason");
      if (reason) reason.textContent = job.pause_reason || "No space left on the disk.";
      // Reset the buttons each time the banner (re)appears — e.g. a resume that
      // didn't free enough space and paused again.
      const rb = $("resume-job");
      if (rb) { rb.disabled = false; rb.textContent = "I've freed up space — Resume"; }
      const sb = $("stop-job");
      if (sb) sb.disabled = false;
    }
    // The banner owns the stop control while paused; otherwise the normal Cancel
    // button is available for the whole running job (and restored after a resume).
    const cancel = $("cancel-job");
    if (cancel && job.status === "running") cancel.hidden = paused;
  }

  // Append-only activity log. The server keeps only the last _LOG_CAP lines but
  // reports `log_total` — the running count of every line ever emitted. We track
  // how many we've already rendered (as an absolute index) and append the rest,
  // so the log keeps flowing even once the stored window starts trimming old
  // lines. (Using the rendered count as the cursor froze the log the instant
  // lines.length plateaued at the cap — about file 83 of 160 at ~6 lines/file.)
  const logEl = $("activity-log");
  if (logEl) {
    const lines = job.log || [];
    const total = Number(job.log_total != null ? job.log_total : lines.length);
    const base = total - lines.length;             // absolute index of lines[0]
    let shown = Number(logEl.dataset.shown || 0);  // absolute lines already shown
    // Decide BEFORE adding whether to follow the tail: only if the user is parked
    // at (within a few px of) the bottom. If they've scrolled up to read, leave
    // their position alone — they can scroll back down to resume following.
    const stickToBottom =
      logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight <= 4;
    // A total below what we've shown means a fresh/reset job — start the log over.
    if (total < shown) { logEl.replaceChildren(); shown = 0; }
    // Append every line we haven't shown yet that the server still has (absolute
    // index >= base). If polling fell behind by more than the cap, the unseen
    // middle is dropped rather than re-flowed — never a freeze.
    for (let i = Math.max(shown, base) - base; i < lines.length; i++) {
      const li = document.createElement("li");
      li.textContent = lines[i];   // textContent -> no escaping needed
      logEl.appendChild(li);
    }
    shown = Math.max(shown, total);
    logEl.dataset.shown = String(shown);
    // Announce only the newest line to screen-reader users.
    if (lines.length) {
      const live = $("activity-live");
      const latest = lines[lines.length - 1];
      if (live && live.textContent !== latest) live.textContent = latest;
    }
    if (stickToBottom) logEl.scrollTop = logEl.scrollHeight;
  }
}

function pollJob(jobId) {
  clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try { job = await api("GET", `/api/jobs/${jobId}`); }
    catch (e) { return; }
    renderProgress(job);
    // Live token counter: ticks up as each page's call returns.
    const tok = $("token-text");
    if (tok) {
      if (job.tokens && job.tokens.calls) {
        const prefix = job.status === "running" ? "Tokens so far — " : "Tokens — ";
        tok.textContent = prefix + tokenSummary(job.tokens) + costSuffix(job.tokens);
      } else {
        tok.textContent = "";
      }
    }
    if (job.status !== "running") {
      clearInterval(pollTimer);
      activeJobId = null;
      const cancel = $("cancel-job");
      if (cancel) { cancel.hidden = true; cancel.disabled = false; cancel.textContent = "Cancel"; }
      $("transcribe").disabled = false;
      $("estimate").disabled = staged.length === 0;
      // Show whatever finished — completed files remain downloadable on cancel
      // and on a disk-full stop.
      if (job.status === "done") {
        announce(`Transcription complete — ${(job.results || []).length} file(s) ready to download.`);
      } else if (job.status === "cancelled") {
        announce(`Transcription cancelled — ${(job.results || []).length} file(s) completed.`);
      } else if (job.status === "stopped") {
        const okCount = (job.results || []).filter((r) => !r.error).length;
        announce(`Stopped — out of disk space. ${okCount} file(s) saved and ready to download.`);
      } else if (job.status === "error") {
        announce("Transcription failed: " + (job.error || "unknown error"));
      }
      if ((job.status === "done" || job.status === "cancelled" || job.status === "stopped")
          && (job.results || []).length) {
        renderResults(jobId, job);
        if (job.status === "done") {
          // Send keyboard/screen-reader focus to the freshly-rendered results,
          // centring it so it's actually in view (not just focused off-screen).
          const h = $("results-heading");
          if (h) { h.scrollIntoView({ block: "center" }); h.focus(); }
        }
      }
      updateClearButton();  // job finished: re-enable clear (results may be up)
    }
  };
  tick();
  pollTimer = setInterval(tick, 700);
}

// ---- bulk download (type-filtered) -------------------------------------- //
const DOWNLOAD_TYPES = ["hocr", "alto", "pdf", "txt"];

// Which download-type checkboxes are currently ticked.
function selectedDownloadTypes() {
  return DOWNLOAD_TYPES.filter((t) => { const cb = $("dl-" + t); return cb && cb.checked; });
}

// "Download selected" is meaningless with nothing ticked — disable it then.
function syncDownloadSelected() {
  const btn = $("dl-selected");
  if (btn) btn.disabled = selectedDownloadTypes().length === 0;
}

// Show (or clear) a message under the download controls -- e.g. "not enough disk
// space to build the zip; use the per-file links". aria-live so it's announced.
function setDownloadNote(msg) {
  const el = $("download-note");
  if (!el) return;
  el.textContent = msg || "";
  el.hidden = !msg;
}

// Start a download without navigating away: the zip response is an attachment,
// so a transient anchor click keeps the results page intact.
function triggerDownload(url) {
  const a = document.createElement("a");
  a.href = url;
  a.rel = "noopener";
  (document.body || document.documentElement).appendChild(a);
  a.click();
  a.remove();
}

function renderResults(jobId, job) {
  $("results-card").hidden = false;
  const dlSel = $("dl-selected");
  if (dlSel) dlSel.onclick = async () => {
    const types = selectedDownloadTypes();
    if (!types.length) return;
    const url = `/api/download/${jobId}.zip?types=${types.join(",")}`;
    setDownloadNote("");          // clear any prior message
    dlSel.disabled = true;
    try {
      // Pre-flight: a full disk can't build the zip, so check before the browser
      // kicks off an attachment download that would otherwise fail silently.
      await api("GET", url + "&probe=1");
      triggerDownload(url);
    } catch (e) {
      setDownloadNote(e.message || "Couldn't prepare the download.");
    } finally {
      dlSel.disabled = false;
      syncDownloadSelected();
    }
  };
  // Unified download: one button that grabs the produced formats as a zip. Show
  // the type picker (each produced format checked) only when there's an actual
  // choice -- more than one format. With a single format the picker stays hidden
  // and the button just downloads it, so there aren't two redundant controls.
  const present = DOWNLOAD_TYPES.filter((t) => (job.results || []).some((r) => r[t]));
  for (const t of DOWNLOAD_TYPES) {
    const cb = $("dl-" + t);
    if (!cb) continue;
    const has = present.includes(t);
    cb.checked = has;              // every produced format is selected by default
    cb.disabled = !has;
    if (cb.parentElement) cb.parentElement.hidden = !has;  // drop formats not produced
  }
  const picker = $("dl-types-fieldset");
  if (picker) picker.hidden = present.length <= 1;
  syncDownloadSelected();
  // Per-job token total with the transparent dollar disclaimer.
  const totals = $("results-tokens");
  if (totals) {
    if (job.tokens && job.tokens.calls) {
      totals.hidden = false;
      totals.innerHTML = `<b>Total tokens:</b> ${tokenSummary(job.tokens)}.${costDisclaimerHtml(job.tokens)}`;
    } else {
      totals.hidden = true;
      totals.innerHTML = "";
    }
  }
  const root = $("results");
  root.innerHTML = "";
  for (const r of job.results) {
    const div = document.createElement("div");
    div.className = "result";
    const tokPill = (r.tokens && r.tokens.calls)
      ? ` <span class="pill">${formatTokens(r.tokens.total)} tokens${
          r.tokens.cost !== null && r.tokens.cost !== undefined ? ", ~" + formatCost(r.tokens.cost) : ""
        }</span>`
      : "";
    let html = `<h3>${escapeHtml(r.source_name)} <span class="pill">${r.n_pages} page(s), ${r.n_lines} lines</span>${tokPill}</h3>`;
    if (r.error) {
      html += `<p class="err">Error: ${escapeHtml(r.error)}</p>`;
      div.innerHTML = html;
    } else {
      // Only link the formats this run actually produced (the output picker may
      // have skipped some) — otherwise the href would be a broken "null".
      const pdfLink = r.pdf ? `<a class="btn small primary" href="${r.pdf}">Searchable PDF</a>` : "";
      const txtLink = r.txt ? `<a class="btn small" href="${r.txt}">Download .txt</a>` : "";
      const hocrLink = r.hocr ? `<a class="btn small" href="${r.hocr}">Download .hocr</a>` : "";
      const altoLink = r.alto ? `<a class="btn small" href="${r.alto}">Download ALTO (.xml)</a>` : "";
      html += `<div class="links">
        ${pdfLink}
        ${txtLink}
        ${hocrLink}
        ${altoLink}</div>`;
      div.innerHTML = html;
      const links = div.querySelector(".links");
      r.images.forEach((im, i) => {
        const label = r.images.length > 1 ? `Preview p${i + 1}` : "Preview boxes";
        const pageSuffix = r.images.length > 1 ? `, page ${i + 1}` : "";
        const btn = document.createElement("button");
        btn.className = "btn small"; btn.type = "button";
        btn.textContent = label;
        btn.setAttribute("aria-label", `Preview detected line boxes for ${r.source_name}${pageSuffix}`);
        btn.onclick = () => openPreview(im.preview, `${r.source_name} — detected lines${pageSuffix}`);
        links.appendChild(btn);
        // No standalone page-PNG download: the re-rendered PNG is lower quality
        // than the user's own source image. (The PNGs still back the preview, the
        // searchable PDF, the hOCR pairing, and the "Download all" zip.)
      });
    }
    root.appendChild(div);
  }
}

// ---- preview modal (native <dialog>: focus trap + Esc + inert bg) ------- //
let lastFocused = null;
function openPreview(url, title) {
  const dlg = $("modal");
  const img = $("modal-img");
  const label = title || "Image preview";  // never leave the dialog unlabelled
  img.src = url;
  img.alt = label;                 // descriptive alt for this specific preview
  $("modal-title").textContent = label;
  lastFocused = document.activeElement;  // so we can restore focus on close
  if (typeof dlg.showModal === "function" && !dlg.open) dlg.showModal();
  else dlg.setAttribute("open", "");     // fallback for very old browsers
}
function closePreview() {
  const dlg = $("modal");
  if (typeof dlg.close === "function" && dlg.open) dlg.close();
  else dlg.removeAttribute("open");
  $("modal-img").removeAttribute("src");
}

// ---- token / cost estimate --------------------------------------------- //
const PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing";

function formatTokens(n) {
  return Number(n || 0).toLocaleString();
}

// Under $1, show cents to the tenth (e.g. "12.3¢") — small jobs read better in
// cents than as "$0.123". $1 and up stays in dollars to the cent ("$1.23").
function formatCost(usd) {
  if (usd === null || usd === undefined) return "";
  const v = Number(usd);
  if (v < 1) return (v * 100).toFixed(1) + "¢";  // ¢ = cent sign
  return "$" + v.toFixed(2);
}

function priceBasis(t) {
  return `$${Number(t.price_input_per_mtok || 0).toFixed(2)}/1M input and `
    + `$${Number(t.price_output_per_mtok || 0).toFixed(2)}/1M output`;
}

// "Gemini 3.5 Flash's published price of $… (prices as of …)" — the model and
// date a dollar figure was computed from, for full transparency.
function priceSource(t) {
  const model = t.model_label ? `${escapeHtml(t.model_label)}'s ` : "";
  const asOf = t.prices_as_of ? ` (prices as of ${escapeHtml(t.prices_as_of)})` : "";
  return `${model}published price of ${priceBasis(t)}${asOf}`;
}

// Compact token line (no dollars): "12,345 tokens · 10,000 in / 2,345 out · 8 API calls".
function tokenSummary(t) {
  if (!t) return "";
  const parts = [`${formatTokens(t.total)} tokens`];
  parts.push(
    `${formatTokens(t.input)} in / ${formatTokens(t.output)} out`
    + (t.thinking ? ` (incl. ${formatTokens(t.thinking)} thinking)` : "")
  );
  if (t.calls) parts.push(`${formatTokens(t.calls)} API call${t.calls === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

// " · ~$0.0123 (est.)" when prices are set, else "".
function costSuffix(t) {
  if (!t || t.cost === null || t.cost === undefined) return "";
  return ` · ~${formatCost(t.cost)} (est.)`;
}

// The fuller, transparent dollar disclaimer shown with a final/total figure.
function costDisclaimerHtml(t) {
  if (!t || t.cost === null || t.cost === undefined) {
    return ` <span class="muted">No published price on file for this model, so there's no dollar estimate &mdash; the token counts above are exact.</span>`;
  }
  return ` <span class="muted">&mdash; <b>~${formatCost(t.cost)}</b> is a rough estimate, not a guarantee, `
    + `based on ${priceSource(t)}. Token counts are exact; Google `
    + `bills the actual usage &mdash; verify live rates at `
    + `<a href="${PRICING_URL}" target="_blank" rel="noopener noreferrer">Gemini API pricing</a>.</span>`;
}

async function estimateCost() {
  if (!staged.length) return;
  const box = $("estimate-info");
  if (!box) return;
  box.hidden = false;
  box.className = "key-info estimate-info";
  box.innerHTML = `<span>Estimating&hellip;</span>`;
  $("estimate").disabled = true;
  try {
    const data = await api("POST", "/api/estimate", { file_ids: staged.map((f) => f.id) });
    box.className = "key-info estimate-info";
    box.innerHTML = renderEstimate(data);
    announce(estimateSummary(data));
  } catch (e) {
    box.className = "key-info estimate-info warn";
    box.innerHTML =
      `<div class="estimate-line"><span class="glyph" aria-hidden="true">!</span><span>Couldn't estimate: ${escapeHtml(e.message)}</span></div>`;
    announce("Couldn't estimate cost: " + e.message);
  } finally {
    // Don't re-enable if a transcription job started while this estimate was in
    // flight -- it must stay greyed for the duration of the run.
    $("estimate").disabled = staged.length === 0 || jobRunning();
  }
}

function renderEstimate(d) {
  if (d.billable === false) {
    return `<div class="estimate-line"><span class="glyph" aria-hidden="true">●</span>`
      + `<span>No Gemini tokens for these ${d.files} file(s): ${escapeHtml(d.reason)} makes no API call, so there's no token cost.</span></div>`;
  }
  const hasCost = d.cost_low !== null && d.cost_low !== undefined;
  // Headline: an estimated cost RANGE (output scales with how much text is on
  // the page) -- or a token range when the model has no published price.
  const headline = hasCost
    ? `<div class="estimate-cost"><span class="estimate-cost-num">~${formatCost(d.cost_low)}–${formatCost(d.cost_high)}</span>`
      + `<span class="estimate-cost-label">estimated range &mdash; not a guarantee</span></div>`
    : `<div class="estimate-cost"><span class="estimate-cost-num">${formatTokens(d.total_low)}–${formatTokens(d.total_high)}</span>`
      + `<span class="estimate-cost-label">tokens &mdash; no published price for this model</span></div>`;
  // Supporting detail as bullets rather than a paragraph.
  const points = [];
  points.push(
    `<b>${d.files}</b> file(s), <b>${formatTokens(d.pages)}</b> page(s)`
    + (d.model_label ? ` with <b>${escapeHtml(d.model_label)}</b>` : "")
  );
  points.push(
    `~<b>${formatTokens(d.input)}</b> input + ~<b>${formatTokens(d.output_low)}–${formatTokens(d.output_high)}</b> output tokens`
    + ` <span class="muted">(assuming ~${formatTokens(d.per_page_low)}–${formatTokens(d.per_page_high)} output tokens/page across ${formatTokens(d.pages)} page(s))</span>`
  );
  if (hasCost) {
    points.push(
      `Priced at ${priceBasis(d)}`
      + (d.prices_as_of ? `, prices as of ${escapeHtml(d.prices_as_of)}` : "")
      + ` &mdash; <a href="${PRICING_URL}" target="_blank" rel="noopener noreferrer">check live pricing</a>`
    );
  }
  points.push(
    `<span class="muted">The range tracks how much text is on each page &mdash; pages with less writing `
    + `cost less, pages with more cost more. Both ends are estimates: input is measured from the first `
    + `page (and scaled), output is assumed. The live counter shows the real usage during the run.</span>`
  );
  const lis = points.map((p) => `<li>${p}</li>`).join("");
  return `${headline}<ul class="estimate-points">${lis}</ul>`;
}

// Plain-text version of the estimate headline, for the screen-reader announcer.
function estimateSummary(d) {
  if (d.billable === false) {
    return `No Gemini API cost for these ${d.files} file(s): ${d.reason}.`;
  }
  if (d.cost_low !== null && d.cost_low !== undefined) {
    return `Estimated cost ${formatCost(d.cost_low)} to ${formatCost(d.cost_high)} `
      + `for ${formatTokens(d.pages)} page(s).`;
  }
  return `Estimated ${formatTokens(d.total_low)} to ${formatTokens(d.total_high)} tokens `
    + `for ${formatTokens(d.pages)} page(s).`;
}

// ---- misc --------------------------------------------------------------- //
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Stage files already on this machine by path -- the server reads them in place,
// no copy and no upload. Complements drag-and-drop for big local batches.
async function addFromPath() {
  const input = $("path-input");
  const path = (input.value || "").trim();
  const note = $("action-note");
  if (!path) { input.focus(); return; }
  const btn = $("add-path");
  btn.disabled = true;
  note.textContent = "Reading files from that location…";
  try {
    const data = await api("POST", "/api/stage-path", { path });
    const accepted = (data && data.files) || [];
    staged.push(...accepted);
    renderStaged();
    input.value = "";
    const skip = data && data.skipped ? ` · skipped ${data.skipped} unsupported file(s)` : "";
    note.textContent = stagedStatus() + skip;
    announce(`${accepted.length} file(s) added. ${stagedStatus()}`);
    pollStagedPages();   // fill the page-count pills, same as an upload
  } catch (e) {
    note.textContent = "Couldn't add from path: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function wire() {
  $("save-key").onclick = () => {
    const hadKey = $("api_key").value.trim() !== "";
    saveSettings({ api_key: $("api_key").value }).then(() => {
      $("api_key").value = "";
      // A freshly-saved key invalidates any prior "no API key" transcription
      // error, so clear that stale message immediately.
      $("action-note").textContent = stagedStatus();
      if (hadKey) {
        announce("Gemini API key saved.");
        verifyKey(true); // confirm the just-pasted key works, and announce it
      }
    });
  };
  $("clear-key").onclick = async () => {
    if (!confirm("Clear the stored Gemini key from this machine?\n(If GEMINI_API_KEY is set in your environment, that will still be used.)")) return;
    await api("DELETE", "/api/settings/api_key");
    $("api_key").value = "";
    $("api_key").removeAttribute("aria-invalid");
    await loadSettings();
    announce("Gemini API key cleared.");
  };
  for (const id of [...TEXT, ...NUMERIC]) $(id).addEventListener("change", () => saveSettings(gatherSettings()));
  for (const id of BOOL) $(id).addEventListener("change", () => saveSettings(gatherSettings()));
  for (const r of document.querySelectorAll("input[name=mode]")) r.addEventListener("change", () => saveSettings(gatherSettings()));
  for (const r of document.querySelectorAll("input[name=content_type]")) r.addEventListener("change", () => saveSettings(gatherSettings()));
  // Model picker: update the visible price basis, then persist (both models).
  $("model").addEventListener("change", () => { updateModelPricingHint(); saveSettings(gatherSettings()); });

  const dz = $("dropzone");
  $("browse").onclick = () => $("file-input").click();
  $("file-input").addEventListener("change", (e) => uploadFiles(e.target.files));
  // Add files already on disk by path (no upload, no copy).
  $("add-path").onclick = addFromPath;
  $("path-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); addFromPath(); }
  });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    uploadFiles(e.dataTransfer.files);
  });

  $("transcribe").onclick = transcribe;
  $("estimate").onclick = estimateCost;
  $("clear-batch").onclick = clearBatch;
  updateClearButton();  // hidden until there's a batch to clear
  $("cancel-job").onclick = cancelJob;
  const resumeBtn = $("resume-job"); if (resumeBtn) resumeBtn.onclick = resumeJob;
  const stopBtn = $("stop-job"); if (stopBtn) stopBtn.onclick = endJob;

  // Download-by-type controls in the Results header: keep "Download selected"
  // enabled only while at least one type is ticked.
  for (const t of DOWNLOAD_TYPES) {
    const cb = $("dl-" + t);
    if (cb) cb.addEventListener("change", syncDownloadSelected);
  }
  syncDownloadSelected();

  // The "skip our overlay" toggle only matters when a searchable PDF is being
  // built, so re-evaluate it whenever the output picker changes.
  for (const t of OUTPUT_FORMAT_IDS) {
    const cb = $("out-" + t);
    if (cb) cb.addEventListener("change", updateSkipOverlayControl);
  }
  updateSkipOverlayControl();

  // Settings disclosure (heading > button) toggle + theme switcher.
  $("settings-toggle").onclick = () =>
    setSettingsOpen($("settings-toggle").getAttribute("aria-expanded") !== "true");
  // Segmented theme control: both options visible; selecting either (click or
  // arrow keys) applies it. Native radios fire 'change' on selection.
  for (const r of document.querySelectorAll('input[name="theme"]'))
    r.addEventListener("change", () => setTheme(r.value));

  const dlg = $("modal");
  $("modal-close").onclick = closePreview;
  // Clicking the backdrop (target is the dialog itself) closes it.
  dlg.addEventListener("click", (e) => { if (e.target === dlg) closePreview(); });
  // Esc (native) and the Close button both fire 'close' -> restore focus.
  dlg.addEventListener("close", () => {
    if (lastFocused && typeof lastFocused.focus === "function") lastFocused.focus();
  });
}

// ---- heartbeat: server shuts itself down when the tab stops pinging ---- //
function heartbeat() {
  fetch("/api/heartbeat", { method: "POST", keepalive: true }).catch(() => {});
}
heartbeat();
setInterval(heartbeat, 5000);
function bye() {
  try { navigator.sendBeacon("/api/heartbeat?bye=true"); } catch (e) {}
}
window.addEventListener("beforeunload", bye);
window.addEventListener("pagehide", (e) => { if (!e.persisted) bye(); });

// ---- theme + settings disclosure state (both persisted) ----------------- //
function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}
function setTheme(theme) {
  const root = document.documentElement;
  if (theme === "light") root.dataset.theme = "light";
  else delete root.dataset.theme;            // dark is the default (no attribute)
  // Reflect the choice in the always-visible segmented control.
  const radio = $(theme === "light" ? "theme-light" : "theme-dark");
  if (radio) radio.checked = true;
  try { localStorage.setItem("cb.theme", theme); } catch (e) {}
}

function setSettingsOpen(open) {
  const btn = $("settings-toggle");
  const body = $("settings-body");
  if (!btn || !body) return;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  body.hidden = !open;
  try { localStorage.setItem("cb.settings.open", open ? "1" : "0"); } catch (e) {}
}

async function init() {
  wire();
  // Sync the theme button to whatever the pre-paint inline script applied, and
  // restore the saved Settings open/closed choice (default: open).
  setTheme(currentTheme());
  let settingsOpen = "1";
  try { settingsOpen = localStorage.getItem("cb.settings.open") || "1"; } catch (e) {}
  setSettingsOpen(settingsOpen !== "0");
  // Populate the model dropdown before loading settings, so the saved model can
  // be selected and its price shown.
  await loadModelCatalog();
  await loadSettings();
  loadTesseractStatus();
}
init();
