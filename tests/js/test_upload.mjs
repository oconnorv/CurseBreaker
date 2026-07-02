// Stubbed-DOM harness for app.js upload helpers: batch planning, byte
// formatting, and the live upload status line. Run directly:
// `node tests/js/test_upload.mjs` (also run under pytest via
// tests/test_frontend_js.py). Exit 0 = all pass.
"use strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP_JS = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../src/cursbreaker/static/app.js"
);

function makeEl() {
  const el = {
    _children: [], _html: "", _attrs: {}, style: {},
    dataset: {}, classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    value: "", checked: false, hidden: false, textContent: "", className: "",
    disabled: false, href: "", onclick: null, options: [],
    scrollTop: 0, clientHeight: 200, scrollHeight: 0,
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return this._attrs[k]; },
    removeAttribute(k) { delete this._attrs[k]; },
    addEventListener() {}, focus() {},
    appendChild(c) { this._children.push(c); this.options.push(c); return c; },
    replaceChildren() { this._children = []; this.options = []; },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; },
  };
  Object.defineProperty(el, "childElementCount", { get() { return this._children.length; } });
  Object.defineProperty(el, "innerHTML", {
    get() { return this._html; },
    set(v) { this._html = v; if (v === "") { this._children = []; this.options = []; } },
  });
  return el;
}

let elements = {};
const el = (id) => (elements[id] || (elements[id] = makeEl()));

const document = {
  getElementById: el,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => makeEl(),
  documentElement: makeEl(),
  activeElement: makeEl(),
  addEventListener() {},
};
const sandbox = {
  document,
  localStorage: { getItem: () => null, setItem() {} },
  fetch: () => Promise.resolve({ ok: true, json: async () => ({}) }),
  matchMedia: () => ({ matches: false }),
  console, navigator: { sendBeacon() {} },
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0,
  confirm: () => true, FormData: class { append() {} },
  XMLHttpRequest: class { open() {} send() {} },
  addEventListener() {},
};
sandbox.globalThis = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(APP_JS, "utf-8"), sandbox, { filename: "app.js" });

let failures = 0;
const check = (name, cond, extra) => {
  if (cond) console.log("PASS", name);
  else { failures++; console.log("FAIL", name, extra !== undefined ? ":: " + extra : ""); }
};

const { planUploadBatches, formatBytes, uploadStatusText, isSupportedFile,
        stagedStatus, pendingPageCounts, applyStagedPages,
        applyStagedTextLayers } = sandbox;
check("planUploadBatches is exported", typeof planUploadBatches === "function");
check("formatBytes is exported", typeof formatBytes === "function");
check("uploadStatusText is exported", typeof uploadStatusText === "function");
check("isSupportedFile is exported", typeof isSupportedFile === "function");
check("stagedStatus is exported", typeof stagedStatus === "function");
check("pendingPageCounts is exported", typeof pendingPageCounts === "function");
check("applyStagedPages is exported", typeof applyStagedPages === "function");

const MB = 1024 * 1024;
const file = (size) => ({ size });

// --- Group A: byte formatting -------------------------------------------- //
check("formatBytes bytes", formatBytes(812) === "812 B", formatBytes(812));
check("formatBytes MB one-decimal", formatBytes(3.4 * MB) === "3.4 MB", formatBytes(3.4 * MB));
check("formatBytes GB rounds >=10", formatBytes(10 * 1024 * MB) === "10 GB", formatBytes(10 * 1024 * MB));
check("formatBytes handles junk", formatBytes(undefined) === "0 B", formatBytes(undefined));

// --- Group A2: supported-type filter ------------------------------------ //
for (const name of ["scan.png", "p.PDF", "x.Tiff", "a.tif", "b.jpg", "c.jpeg", "d.gif"]) {
  check(`isSupportedFile accepts ${name}`, isSupportedFile({ name }) === true, name);
}
for (const name of ["notes.docx", "readme.txt", "Thumbs.db", ".DS_Store", "noext"]) {
  check(`isSupportedFile rejects ${name}`, isSupportedFile({ name }) === false, name);
}
check("isSupportedFile tolerates missing name", isSupportedFile({}) === false);
check("isSupportedFile tolerates null", isSupportedFile(null) === false);
// The bug this guards: an all-unsupported lead group would otherwise become its
// own batch and 400. After filtering, only the supported files are batched.
const mixed = [{ name: "a.docx", size: 1 }, { name: "b.docx", size: 1 }, { name: "c.png", size: 1 }];
const supported = mixed.filter(isSupportedFile);
check("filter leaves only supported", supported.length === 1 && supported[0].name === "c.png");
check("planUploadBatches over filtered set has no empty/unsupported batch",
  planUploadBatches(supported, 2, 256 * MB).flat().every(isSupportedFile));

// --- Group B: batch planning -------------------------------------------- //
// Count cap: 45 files at 20/batch -> 20 + 20 + 5.
let b = planUploadBatches(Array.from({ length: 45 }, () => file(1)), 20, 256 * MB);
check("count cap splits into 3 batches", b.length === 3, b.length);
check("count cap batch sizes", b.map((x) => x.length).join(",") === "20,20,5", b.map((x) => x.length).join(","));

// Byte cap: 5 files of 100 MB, cap 256 MB -> [100,100] , [100,100], [100].
b = planUploadBatches([file(100 * MB), file(100 * MB), file(100 * MB), file(100 * MB), file(100 * MB)], 20, 256 * MB);
check("byte cap splits by size", b.map((x) => x.length).join(",") === "2,2,1", b.map((x) => x.length).join(","));

// A single oversized file still gets shipped (its own batch), never dropped.
b = planUploadBatches([file(900 * MB), file(10 * MB)], 20, 256 * MB);
check("oversized file gets its own batch", b.length === 2 && b[0].length === 1, JSON.stringify(b.map((x) => x.length)));
check("every file accounted for", b.flat().length === 2, b.flat().length);

check("empty input -> no batches", planUploadBatches([], 20, 256 * MB).length === 0);

// --- Group C: status line ------------------------------------------------ //
let s = uploadStatusText({ sentBytes: MB, totalBytes: 4 * MB, filesDone: 0, filesTotal: 10, saving: false });
check("uploading shows percent", s.includes("Uploading… 25%"), s);
check("uploading shows byte progress", s.includes("(1.0 MB of 4.0 MB)"), s);
check("uploading shows file counts", s.includes("0/10 file(s) ready"), s);

s = uploadStatusText({ sentBytes: 4 * MB, totalBytes: 4 * MB, filesDone: 4, filesTotal: 10, saving: true });
check("saving phase is labelled", s.startsWith("Saving…"), s);
check("saving still reports files", s.includes("4/10 file(s) ready"), s);

s = uploadStatusText({ sentBytes: 0, totalBytes: 0, filesDone: 0, filesTotal: 0, saving: false });
check("zero totals don't divide-by-zero", s === "Uploading… 0%", s);

// --- Group D: lazy page counts + aggregate summary ----------------------- //
check("stagedStatus empty -> ''", stagedStatus([]) === "", JSON.stringify(stagedStatus([])));
check("stagedStatus pending -> file count only (no pages yet)",
  stagedStatus([{ pages: null }, { pages: 2 }]) === "2 file(s) ready",
  stagedStatus([{ pages: null }, { pages: 2 }]));
check("stagedStatus complete -> appends the total pages",
  stagedStatus([{ pages: 2 }, { pages: 3 }]) === "2 file(s) ready · 5 page(s)",
  stagedStatus([{ pages: 2 }, { pages: 3 }]));

const list = [{ id: "a", pages: null }, { id: "b", pages: 2 }, { id: "c", pages: null }];
check("pendingPageCounts true when any null", pendingPageCounts(list) === true);
let changed = applyStagedPages(list, { a: 5 });  // only 'a' resolves this round
check("applyStagedPages reports a change", changed === true);
check("applyStagedPages fills the resolved count", list[0].pages === 5, list[0].pages);
check("applyStagedPages leaves already-known counts", list[1].pages === 2, list[1].pages);
check("applyStagedPages leaves still-pending null", list[2].pages === null, String(list[2].pages));
check("pendingPageCounts still true (c pending)", pendingPageCounts(list) === true);
changed = applyStagedPages(list, { c: 7 });
check("applyStagedPages resolves the last one", list[2].pages === 7 && changed === true, list[2].pages);
check("pendingPageCounts false once all known", pendingPageCounts(list) === false);
check("applyStagedPages no-op when nothing new", applyStagedPages(list, { a: 99 }) === false);
check("applyStagedPages didn't overwrite a known count", list[0].pages === 5, list[0].pages);

// --- Group E: existing-text-layer flags --------------------------------- //
check("applyStagedTextLayers is exported", typeof applyStagedTextLayers === "function");
const tl = [{ id: "a" }, { id: "b" }, { id: "c", hasText: true }];
applyStagedTextLayers(tl, { a: true, b: false });  // 'c' already known
check("applyStagedTextLayers records a true flag", tl[0].hasText === true, String(tl[0].hasText));
check("applyStagedTextLayers records a false flag", tl[1].hasText === false, String(tl[1].hasText));
check("applyStagedTextLayers leaves an already-known flag", tl[2].hasText === true);
check("applyStagedTextLayers leaves a still-unknown flag undefined",
  applyStagedTextLayers(tl, {}) === undefined && tl[1].hasText === false);
// A file still being scanned (null) must not be coerced to a boolean.
const pending = [{ id: "x" }];
applyStagedTextLayers(pending, { x: null });
check("applyStagedTextLayers ignores a null (still scanning)", pending[0].hasText === undefined,
  String(pending[0].hasText));
applyStagedTextLayers(pending, undefined);  // tolerates a missing map
check("applyStagedTextLayers tolerates a missing map", pending[0].hasText === undefined);

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);
