"""FastAPI app for the local CursBreaker GUI.

Everything runs on the user's machine and binds to localhost. Uploaded files are
copied into a temp staging area; each processing run writes its outputs to a
per-job temp directory, which is offered back as individual downloads or a zip.
The user's original files are never modified.
"""

from __future__ import annotations

import atexit
import errno
import io
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from lxml import etree
from PIL import Image, ImageDraw
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import __version__
from .config import load_settings, save_settings
from .gemini_client import make_provider
from .hocr import XHTML_NS
from .images import SUPPORTED_EXT, count_content_pages, is_supported, pdf_has_text_layer
from .pipeline import OUTPUT_FORMATS, estimate_usage, process_batch
from .pricing import (
    CATALOG,
    PRICES_AS_OF,
    PRICING_URL,
    cost_for,
    effective_rates,
    pricing_for,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="CursBreaker", version=__version__)

# Staged uploads and per-job outputs live under one temp workspace. Override its
# location with CURSBREAKER_WORK_DIR (e.g. a roomier drive) when the default temp
# dir -- often a small or full C: on Windows -- can't hold a large batch.
_WORK_PARENT = os.environ.get("CURSBREAKER_WORK_DIR") or None  # None -> system temp
if _WORK_PARENT:
    os.makedirs(_WORK_PARENT, exist_ok=True)
_BASE = Path(tempfile.mkdtemp(prefix="cursbreaker_", dir=_WORK_PARENT))
STAGE_DIR = _BASE / "stage"
JOBS_DIR = _BASE / "jobs"
STAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_workspace() -> None:
    """Remove this session's temp workspace (staged uploads + job outputs). Best
    effort; registered with atexit and run on the hard-exit path, so a normal quit
    doesn't leave gigabytes of copied images behind in temp."""
    shutil.rmtree(_BASE, ignore_errors=True)


def sweep_stale_workspaces() -> None:
    """Delete leftover cursbreaker_* workspaces from earlier runs that exited
    without cleaning up -- e.g. an OS OOM-kill or power loss, where neither atexit
    nor the hard-exit hook gets to run. Best effort and scoped to our own temp
    prefix; the live session's dir is skipped, and any dir with files still open
    (a concurrent instance on Windows) simply fails to delete and is left alone.

    Called once at launch (from __main__), never at import -- so parallel test
    workers, each with their own workspace, never sweep one another."""
    for d in _BASE.parent.glob("cursbreaker_*"):
        if d != _BASE and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_workspace)

STAGED: dict[str, Path] = {}
JOBS: dict[str, dict] = {}

# Page counts are computed off the upload request (a slow scan of big multi-frame
# TIFFs/PDFs shouldn't hold staging hostage). ``None`` means "still counting";
# the browser polls /api/staged-pages to fill the numbers in. They're a display
# hint only -- processing and estimating recount independently.
STAGED_PAGES: dict[str, int | None] = {}
# Whether each staged PDF already carries a searchable text layer (``None`` while
# still being scanned). Computed in the same background pass as the page counts
# and polled alongside them, so the UI can offer to skip our overlay for a batch
# whose inputs are already searchable. Non-PDFs are always ``False``.
STAGED_HAS_TEXT: dict[str, bool | None] = {}
_STAGED_PAGES_LOCK = threading.Lock()

# Browser heartbeat / auto-shutdown state. Tests don't start the watchdog; it is
# kicked off explicitly from __main__ when the CLI launches the server.
_LAST_PING_AT: float | None = None
_AUTOSHUTDOWN_STARTED = False


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@app.get("/api/settings")
def get_settings():
    return load_settings().public_dict()


@app.post("/api/settings")
async def update_settings(payload: dict):
    settings = load_settings()
    fields = type(settings).model_fields
    for key, value in payload.items():
        if key == "api_key":
            if value:  # never blank out a saved key on a no-op save
                settings.api_key = value
        elif key in fields:
            setattr(settings, key, value)
    settings.normalize_content()  # migrate a posted legacy "mixed" value
    settings.sync_models()  # detection (two-pass) always follows the chosen model
    save_settings(settings)
    return settings.public_dict()


@app.delete("/api/settings/api_key")
def clear_api_key():
    settings = load_settings()
    settings.api_key = ""
    save_settings(settings)
    return settings.public_dict()


@app.get("/api/tesseract")
def tesseract_status():
    """Detailed Tesseract status: which piece (if any) is missing and where we
    looked, so the UI can offer a fix specific to the actual failure."""
    from . import tesseract_client

    st = tesseract_client.status(load_settings(), force=True)
    return {
        "available": st.installed,  # kept for back-compat with older clients
        "languages": st.languages,
        "wrapper_present": st.wrapper_present,
        "binary_found": st.binary_found,
        "cmd_path": st.cmd_path,
        "source": st.source,
        "version": st.version,
        "error": st.error,
        "install_hint": st.install_hint,
        "managed_dir": st.managed_dir,
    }


@app.get("/api/key-status")
def key_status():
    """Cheap, generation-free check that the stored Gemini key still works, so a
    revoked/expired key surfaces in Settings before a transcription fails. Uses
    the free ListModels endpoint (no token/quota cost)."""
    from .gemini_client import check_api_key

    st = check_api_key(load_settings())
    return {"state": st.state, "message": st.message}


@app.get("/api/models")
def list_models():
    """The curated model catalog with published prices, for the dropdown +
    automatic cost estimate. A fixed list (not the key's live ListModels) so the
    selectable models and their prices stay in lockstep."""
    return {
        "models": [
            {
                "id": m.model,
                "label": m.label,
                "input_per_mtok": m.input_per_mtok,
                "output_per_mtok": m.output_per_mtok,
                "tier_threshold": m.tier_threshold,
                "input_per_mtok_high": m.input_per_mtok_high,
                "output_per_mtok_high": m.output_per_mtok_high,
            }
            for m in CATALOG
        ],
        "prices_as_of": PRICES_AS_OF,
        "pricing_url": PRICING_URL,
    }


# --------------------------------------------------------------------------- #
# Upload / process / status
# --------------------------------------------------------------------------- #
def _page_count(path: Path) -> int:
    return count_content_pages(path)


def _count_pages_bg(file_ids: list[str]) -> None:
    """Fill in page counts (and the existing-text-layer flag) for just-staged
    files, off the request thread. Recorded one at a time so the browser's poll
    sees them appear incrementally rather than all-or-nothing at the end of a big
    batch."""
    for fid in file_ids:
        path = STAGED.get(fid)
        n = _page_count(path) if path is not None else None
        has_text = pdf_has_text_layer(path) if path is not None else False
        with _STAGED_PAGES_LOCK:
            STAGED_PAGES[fid] = n
            STAGED_HAS_TEXT[fid] = has_text


_UPLOAD_CHUNK = 1024 * 1024  # stream uploads to disk a MB at a time


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    """Stage uploaded files for transcription.

    Declared **sync** (``def``) on purpose: Starlette runs a sync endpoint in a
    worker thread, so the blocking disk writes for a large batch don't stall the
    event loop — the browser heartbeat and job-status polls keep being served,
    and the auto-shutdown watchdog won't trip mid-upload. Each file is also
    *streamed* to disk in chunks rather than read whole into memory, so a
    multi-GB batch doesn't balloon RAM.

    Page counts are deliberately **not** computed here: scanning a stack of big
    multi-frame TIFFs/PDFs could add seconds per batch, and the count is only a
    display hint. The request returns the moment the bytes are on disk; a
    background worker fills counts in, and the browser polls /api/staged-pages.
    """
    staged = []
    for f in files:
        if not is_supported(f.filename or ""):
            continue
        file_id = uuid.uuid4().hex
        # Stage in a per-upload subdir so the UUID never leaks into output names.
        sub = STAGE_DIR / file_id
        sub.mkdir(parents=True, exist_ok=True)
        dest = sub / Path(f.filename).name
        f.file.seek(0)
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out, _UPLOAD_CHUNK)
        except OSError as exc:
            # Roll back this file's partial write and report cleanly instead of a
            # 500 + traceback. Running out of disk is the common case on a big batch.
            shutil.rmtree(sub, ignore_errors=True)
            if exc.errno == errno.ENOSPC:
                raise HTTPException(
                    507,
                    "Ran out of disk space while staging uploads. Free up space on "
                    "the drive holding the temp folder — or set CURSBREAKER_WORK_DIR "
                    "to a drive with more room — then try again.",
                ) from exc
            raise HTTPException(500, f"Could not save upload: {exc.strerror or exc}") from exc
        STAGED[file_id] = dest
        staged.append({"id": file_id, "name": Path(f.filename).name, "pages": None})
    if not staged:
        raise HTTPException(400, f"No supported files. Allowed: {sorted(SUPPORTED_EXT)}")
    new_ids = [s["id"] for s in staged]
    with _STAGED_PAGES_LOCK:
        for fid in new_ids:
            STAGED_PAGES[fid] = None  # "counting" until the worker records a number
            STAGED_HAS_TEXT[fid] = None
    threading.Thread(target=_count_pages_bg, args=(new_ids,), daemon=True).start()
    return {"files": staged}


@app.get("/api/staged-pages")
def staged_pages():
    """Current page counts for staged files: ``{id: int}`` once known, ``null``
    while a file is still being counted. The browser polls this after an upload
    to fill the staged list's page pills in, without ever blocking the upload (or
    Transcribe/Estimate, which work the moment files are staged) on the scan.

    ``text_layers`` reports, per file, whether it already carries a searchable
    text layer (``null`` while still scanning) so the UI can offer to skip our
    overlay for a batch that's already searchable."""
    with _STAGED_PAGES_LOCK:
        return {"pages": dict(STAGED_PAGES), "text_layers": dict(STAGED_HAS_TEXT)}


def _is_under(path: Path, base: Path) -> bool:
    """True if ``path`` lives inside ``base`` -- used to tell our own staged
    upload *copies* (safe to delete) apart from files staged by path (the user's
    originals, read in place and never ours to remove)."""
    try:
        return Path(path).resolve().is_relative_to(Path(base).resolve())
    except (OSError, ValueError):
        return False


def _tree_bytes(path: Path) -> int:
    """Total size of the files under ``path`` (0 if it's already gone). Best
    effort, only to report how much disk a clear frees."""
    total = 0
    try:
        for p in Path(path).rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


@app.post("/api/clear")
def clear_batch():
    """Clear the current batch so a fresh one can start: drop every staged file
    and finished-job output, and free the disk they held under the workspace.

    Files staged by *path* (the user's own originals, read in place) are only
    unstaged -- never deleted; we remove only copies we made (uploads) and
    outputs we generated. Refuses while a job is still running so we never pull
    the workspace out from under the worker."""
    if any(j.get("status") == "running" for j in JOBS.values()):
        raise HTTPException(409, "A job is still running — cancel it first, then clear.")

    files_cleared = len(STAGED)
    jobs_cleared = len(JOBS)
    freed = 0

    # Staged uploads each live in their own STAGE_DIR/<id>/ subdir; delete those.
    # Path-staged originals sit outside STAGE_DIR and are left on disk untouched.
    for path in list(STAGED.values()):
        if _is_under(Path(path), STAGE_DIR):
            sub = Path(path).parent
            freed += _tree_bytes(sub)
            shutil.rmtree(sub, ignore_errors=True)
    STAGED.clear()
    with _STAGED_PAGES_LOCK:
        STAGED_PAGES.clear()
        STAGED_HAS_TEXT.clear()

    # Generated outputs: one dir per job.
    for job in list(JOBS.values()):
        out = job.get("out_dir")
        if out:
            freed += _tree_bytes(Path(out))
            shutil.rmtree(out, ignore_errors=True)
    JOBS.clear()

    return {
        "files_cleared": files_cleared,
        "jobs_cleared": jobs_cleared,
        "freed_bytes": freed,
    }


class StagePathRequest(BaseModel):
    path: str


@app.post("/api/stage-path")
def stage_path(req: StagePathRequest):
    """Stage files that are already on this machine *by path, without copying
    them*: STAGED points at the originals and the pipeline reads them in place
    (the user's files are never modified). A folder stages the supported files
    directly inside it. This skips both the duplicate disk copy and the HTTP
    upload, so a big local batch is added instantly."""
    raw = (req.path or "").strip().strip('"')
    if not raw:
        raise HTTPException(400, "Enter a file or folder path.")
    try:
        p = Path(raw).expanduser()
        exists = p.exists()
    except OSError as exc:
        raise HTTPException(400, f"That path can't be read: {exc.strerror or exc}")
    if not exists:
        raise HTTPException(404, f"Path not found: {raw}")

    if p.is_dir():
        candidates = sorted(c for c in p.iterdir() if c.is_file())
    elif p.is_file():
        candidates = [p]
    else:
        raise HTTPException(400, "That path is neither a file nor a folder.")

    staged, skipped = [], 0
    for c in candidates:
        if not is_supported(c.name):
            skipped += 1
            continue
        file_id = uuid.uuid4().hex
        STAGED[file_id] = c  # the original on disk -- read in place, never copied
        staged.append({"id": file_id, "name": c.name, "pages": None})
    if not staged:
        where = "that folder" if p.is_dir() else "that file"
        raise HTTPException(
            400, f"No supported files in {where}. Allowed: {sorted(SUPPORTED_EXT)}"
        )

    new_ids = [s["id"] for s in staged]
    with _STAGED_PAGES_LOCK:
        for fid in new_ids:
            STAGED_PAGES[fid] = None
            STAGED_HAS_TEXT[fid] = None
    threading.Thread(target=_count_pages_bg, args=(new_ids,), daemon=True).start()
    return {"files": staged, "skipped": skipped}


class ProcessRequest(BaseModel):
    file_ids: list[str]
    mode: str | None = None
    # Output kinds to write (subset of pipeline.OUTPUT_FORMATS). Empty/omitted
    # means "create everything", so leaving the picker untouched is unchanged.
    outputs: list[str] | None = None
    # Batch-wide: for input PDFs that already carry a searchable text layer, keep
    # the original's layer and skip adding ours (pass the original through). Only
    # affects those files; others still get the overlay.
    skip_existing_text_overlay: bool = False


class EstimateRequest(BaseModel):
    file_ids: list[str]
    mode: str | None = None


@app.post("/api/estimate")
def estimate(req: EstimateRequest):
    """Free pre-flight estimate of the tokens, and the rough dollars, a run will
    cost -- before any transcription happens. Uses Gemini's no-charge
    ``count_tokens`` for input; output is a labelled assumption, and the cost is
    derived automatically from the selected model's published price."""
    paths = [STAGED[i] for i in req.file_ids if i in STAGED]
    if not paths:
        raise HTTPException(400, "No staged files to estimate.")

    settings = load_settings()
    if req.mode in ("one_pass", "two_pass"):
        settings.mode = req.mode

    content = (settings.content_type or "handwriting").lower()
    # Printed-only runs locally (Tesseract) with no Gemini call -> no token cost.
    if content == "text":
        return {
            "billable": False,
            "reason": "Printed-only mode",
            "files": len(paths),
            "input": 0,
            "output_low": 0,
            "output_high": 0,
            "total_low": 0,
            "total_high": 0,
            "calls": 0,
            "cost_low": None,
            "cost_high": None,
        }

    if not settings.resolved_api_key():
        raise HTTPException(
            400, "No Gemini API key set. Add one in Settings to estimate cost."
        )

    try:
        provider = make_provider(settings)
        data = estimate_usage(paths, provider, settings)
    except Exception as exc:
        raise HTTPException(502, f"Couldn't estimate right now: {exc}")
    data["billable"] = True
    return data


@app.post("/api/process")
def process(req: ProcessRequest):
    paths = [STAGED[i] for i in req.file_ids if i in STAGED]
    if not paths:
        raise HTTPException(400, "No staged files to process.")

    settings = load_settings()
    if req.mode in ("one_pass", "two_pass"):
        settings.mode = req.mode
    if not settings.resolved_api_key():
        raise HTTPException(400, "No Gemini API key set. Add one in Settings.")

    # Keep only recognized output kinds; empty (nothing picked, or all unknown)
    # falls through to "create everything" in the pipeline.
    outputs = [o for o in (req.outputs or []) if o in OUTPUT_FORMATS]

    job_id = uuid.uuid4().hex
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # The model is captured at job start so the live cost figure stays anchored
    # to the model actually used, even if the user switches models mid-run.
    model = settings.transcription_model
    JOBS[job_id] = {
        "status": "running",
        "done": 0,                 # legacy file counters (kept for back-compat)
        "total": len(paths),
        "current": "",             # latest step message
        "stage": "",               # latest step's machine tag
        "log": [],                 # append-only activity log (capped)
        "log_total": 0,            # lines ever emitted; lets the UI append past the cap
        "done_units": 0,           # pages completed across the whole job (bar)
        "total_units": 0,          # total pages across the whole job (bar)
        "results": [],
        "out_dir": str(out_dir),
        "error": None,
        "model": model,
        "tokens": _usage_to_dict(None, model),  # zeros until the provider exists
        "paused": False,           # true while the worker is blocked on a full disk
        "pause_reason": None,      # message shown in the disk-full banner
        "_cancel": False,          # private flag flipped by /api/jobs/{id}/cancel
        "_resume": threading.Event(),  # released by /api/jobs/{id}/resume or /end
        "_resume_action": None,    # "resume" | "end" -- read by the worker on wake
    }
    threading.Thread(
        target=_run_job,
        args=(job_id, paths, settings, out_dir, outputs),
        kwargs={"skip_text_overlay": req.skip_existing_text_overlay},
        daemon=True,
    ).start()
    return {"job_id": job_id}


_LOG_CAP = 500  # keep the most recent N activity-log lines (bounds memory + JSON)


def _append_log(job: dict, message: str, cap: int = _LOG_CAP) -> None:
    """Append a line to the job's activity log, keeping only the most recent
    ``cap`` entries in memory while tracking ``log_total`` -- the running count of
    every line ever emitted. The browser appends against that total, so the log
    keeps flowing even after old lines are trimmed. (Using the stored list's
    length as the cursor froze the log the moment that length plateaued at the
    cap -- about file 83 of 160 at ~6 lines/file.)"""
    log = job["log"]
    log.append(message)
    if len(log) > cap:
        del log[: len(log) - cap]
    job["log_total"] = job.get("log_total", 0) + 1


def _run_job(job_id, paths, settings, out_dir, outputs=None, *, skip_text_overlay=False):
    job = JOBS[job_id]
    try:
        provider = make_provider(settings)
        # Expose the provider so a status poll can read its running token total
        # live (per page, as each call returns), not just at file boundaries.
        job["_provider"] = provider

        # Page count up front (cheap; same function /api/upload uses) so the bar
        # is page-driven and actually fills. count_content_pages already returns
        # 1 on any read error, so the sum is always >= the file count. Keep the
        # per-file counts too: process_batch reconciles the page counter to this
        # budget at each file boundary, so a file that errors or under-renders
        # can't strand the bar below 100% partway through a big batch.
        unit_counts = [count_content_pages(p) for p in paths]
        total_units = sum(unit_counts)
        job["total_units"] = total_units

        # The worker thread writes these scalar keys and appends to job["log"];
        # the request thread reads them in _public_job. Simple int/str writes and
        # list.append are atomic under the GIL, and no keys are added after the
        # job dict is created, so no lock is needed.
        def report(ev):
            job["current"] = ev.message
            job["stage"] = ev.stage
            job["done_units"] = ev.units_done
            job["total_units"] = ev.units_total or job["total_units"]
            job["done"], job["total"] = ev.file_index, ev.file_total
            _append_log(job, ev.message)

        def on_disk_full():
            """Block the worker on a full disk -- no more files, no more API calls --
            until the browser hits /resume (space freed -> retry) or /end (stop and
            keep what's saved). The job stays ``running`` so the page keeps polling
            and shows the banner."""
            ev = job["_resume"]
            ev.clear()
            job["_resume_action"] = None
            job["pause_reason"] = (
                "No space left on the disk. Free up space, then Resume to "
                "continue — or Stop to finish with the files already saved."
            )
            job["paused"] = True  # set last: the banner shows only once we're parked
            # Wake on the event; poll _cancel too so a shutdown/cancel still frees us.
            while not ev.wait(timeout=1.0):
                if job.get("_cancel"):
                    break
            job["paused"] = False
            job["pause_reason"] = None
            return "resume" if job.get("_resume_action") == "resume" else "end"

        results = process_batch(
            paths, provider, settings, out_dir, report, units_total=total_units,
            unit_counts=unit_counts,
            should_cancel=lambda: bool(job.get("_cancel")),
            on_disk_full=on_disk_full,
            outputs=outputs,
            skip_text_overlay=skip_text_overlay,
        )
        job["results"] = [
            {
                "source_name": r.source_name,
                "n_pages": r.n_pages,
                "n_lines": r.n_lines,
                "txt": _url(job_id, r.txt_name),
                "hocr": _url(job_id, r.hocr_name),
                "alto": _url(job_id, r.alto_name),
                "pdf": _url(job_id, r.pdf_name),
                # Page images back the Preview overlay only -- no download URL
                # (derivative images aren't a deliverable).
                "images": [
                    {"name": n, "preview": f"/api/preview/{job_id}/{n}"}
                    for n in r.image_names
                ],
                "error": r.error,
                "tokens": _usage_to_dict(r.token_usage, job["model"]),
            }
            for r in results
        ]
        job["tokens"] = _usage_to_dict(provider.usage, job["model"])
        # Completed files (if any) are kept and downloadable in every stop case.
        # "stopped" (user ended at a disk-full pause) is distinct from a manual
        # "cancelled" so the UI can explain what happened.
        if job.get("stage") == "stopped":
            job["status"] = "stopped"
        elif job.get("_cancel"):
            job["status"] = "cancelled"
        else:
            job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


def _usage_to_dict(usage, model=None) -> dict:
    """Serialize a ``TokenUsage`` (or None -> zeros) for the browser. The dollar
    cost is derived automatically from the selected model's published price; it
    is ``None`` when the model isn't in the catalog (so no dollars are implied),
    and the (tier-aware) rates used are echoed back for transparency."""
    if usage is None:
        d = {"input": 0, "output": 0, "thinking": 0, "total": 0, "calls": 0}
    else:
        d = {
            "input": usage.input,
            "output": usage.output,
            "thinking": usage.thinking,
            "total": usage.total,
            "calls": usage.calls,
        }
    pricing = pricing_for(model)
    if pricing:
        if usage is not None:
            in_rate, out_rate = effective_rates(pricing, usage)
            d["cost"] = cost_for(pricing, usage)
        else:
            in_rate, out_rate = pricing.input_per_mtok, pricing.output_per_mtok
            d["cost"] = 0.0
        d["model"] = pricing.model
        d["model_label"] = pricing.label
        d["price_input_per_mtok"] = in_rate
        d["price_output_per_mtok"] = out_rate
        d["prices_as_of"] = PRICES_AS_OF
    else:
        d["price_input_per_mtok"] = 0.0
        d["price_output_per_mtok"] = 0.0
        d["cost"] = None
    return d


def _public_job(job: dict) -> dict:
    """A copy of a job's state safe to send to the browser: private keys (the
    provider handle) dropped, and the token counter refreshed live from the
    provider so it ticks up while the job runs."""
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    # The log list is shared by reference with the worker thread, which may keep
    # appending after this returns and before Starlette serializes; copy it so a
    # stable snapshot is sent.
    if isinstance(out.get("log"), list):
        out["log"] = list(out["log"])
    provider = job.get("_provider")
    usage = getattr(provider, "usage", None) if provider is not None else None
    if usage is not None:
        out["tokens"] = _usage_to_dict(usage, job.get("model"))
    return out


def _url(job_id, name):
    return f"/api/download/{job_id}/{name}" if name else None


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return _public_job(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Request cooperative cancellation of a running job. The worker stops at the
    next page/file boundary (an in-flight Gemini call can't be interrupted), so
    the status flips to ``cancelled`` shortly after; files already finished stay
    downloadable. A no-op on a job that's already finished."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if job["status"] == "running" and not job.get("_cancel"):
        # Acknowledge in the activity log immediately (before the worker reaches
        # a cancel boundary) and explain why it isn't instant. "page" stage means
        # a Gemini/Tesseract call is in flight and can't be interrupted.
        if job.get("stage") == "page":
            msg = ("Cancellation requested — the current page is already being "
                   "processed by Gemini and can't be interrupted, so this will "
                   "take effect once the current step finishes.")
        else:
            msg = "Cancellation requested — stopping after the current step finishes."
        _append_log(job, msg)
        job["current"] = msg
        job["_cancel"] = True  # set last, so the message is logged before the worker stops
    return {"status": job["status"], "cancelling": bool(job.get("_cancel"))}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Resume a job paused on a full disk -- the user has freed space and wants to
    retry the file that couldn't be saved. No-op unless the job is actually
    paused."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if job.get("paused"):
        _append_log(job, "Resuming — retrying the file that ran out of disk space.")
        job["_resume_action"] = "resume"
        job["_resume"].set()  # set last, after the action is recorded
    return {"status": job["status"], "paused": bool(job.get("paused"))}


@app.post("/api/jobs/{job_id}/end")
def end_job(job_id: str):
    """Stop a job paused on a full disk, keeping every file already saved (the
    user chose to finish rather than free space). No-op unless paused."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if job.get("paused"):
        _append_log(job, "Stopping — keeping the files already finished.")
        job["_resume_action"] = "end"
        job["_resume"].set()
    return {"status": job["status"], "paused": bool(job.get("paused"))}


# --------------------------------------------------------------------------- #
# Downloads / preview
# --------------------------------------------------------------------------- #
def _safe_output(job_id: str, name: str) -> Path:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Invalid file name.")
    path = Path(job["out_dir"]) / name
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/api/download/{job_id}/{name}")
def download(job_id: str, name: str):
    return FileResponse(_safe_output(job_id, name), filename=name)


# Friendly download-type names -> the filename suffixes they map to on disk, for
# the type-filtered zip (e.g. ?types=hocr,txt). Lets a user grab just the hOCR of
# a 160-page book instead of the whole archive. Page images aren't here: they're
# internal scaffolding (preview + PDF), never part of a download.
_TYPE_SUFFIXES = {
    "hocr": (".hocr",),
    "alto": (".alto.xml",),
    "pdf": (".pdf",),
    "txt": (".txt",),
}
# Every downloadable document suffix -- the default archive (no ?types) is all of
# these, so a zip never sweeps in the page-image PNGs sitting alongside them.
_DOC_SUFFIXES = tuple(suf for sufs in _TYPE_SUFFIXES.values() for suf in sufs)


def _human_size(num: float) -> str:
    """Bytes as a short human string, for the disk-space messages."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


@app.get("/api/download/{job_id}.zip")
def download_zip(job_id: str, types: str | None = None, probe: bool = False):
    """Download a job's document outputs as a zip. ``?types=hocr,txt`` restricts
    the archive to those types; omit it for every document type. Page images are
    never included (internal scaffolding, not a deliverable).

    The zip is built to a temp file and streamed -- never held whole in memory --
    but building it needs free disk for the whole archive. A 243-page PDF's zip is
    large, so on a full disk that write used to fail silently (or 500 mid-build),
    which read as a crash. We now check free space first and return a clear 507;
    ``?probe=1`` runs that check (and the file selection) without downloading, so
    the browser can warn before it ever starts. The per-file links stream straight
    from disk and need no extra space, so the message points there as the way out."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    out_dir = Path(job["out_dir"])

    selected = None
    suffixes = _DOC_SUFFIXES  # default ("everything") = all document types, no images
    if types is not None and types.strip():
        selected = [t for t in (s.strip().lower() for s in types.split(",")) if t in _TYPE_SUFFIXES]
        if not selected:
            raise HTTPException(400, "Unrecognized download type(s).")
        suffixes = tuple(suf for t in selected for suf in _TYPE_SUFFIXES[t])

    files = [
        f for f in sorted(out_dir.iterdir())
        if f.is_file() and f.name.endswith(suffixes)
    ]
    if not files:
        raise HTTPException(404, "No matching files to download.")

    # Building the zip needs room for the whole archive next to the outputs
    # (already-compressed PDFs/PNGs don't shrink, so ~the summed size). Check up
    # front so a full disk gets a clear message, not a silent or mid-write failure.
    total = sum(f.stat().st_size for f in files)
    needed = int(total * 1.02) + 8 * 1024 * 1024  # +2% zip overhead, +8 MB slack
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError:
        free = needed  # can't measure -> don't block; the build still guards ENOSPC
    if free < needed:
        raise HTTPException(
            507,
            f"Not enough free disk space to build this download (it needs about "
            f"{_human_size(total)}, but only {_human_size(free)} is free). Free up "
            f"space and try again — or use the per-file download links below, which "
            f"stream straight from disk and need no extra space.",
        )
    if probe:
        return {"ok": True}

    tag = "_" + "-".join(selected) if selected else ""
    tmp_path: Path | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            prefix="cursbreaker_zip_", suffix=".zip", dir=str(out_dir), delete=False
        )
        tmp.close()
        tmp_path = Path(tmp.name)
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.name)
    except OSError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if exc.errno == errno.ENOSPC:  # disk filled despite the check (a race)
            raise HTTPException(
                507,
                "Ran out of disk space while building the download. Free up space "
                "and try again, or use the per-file download links below.",
            ) from exc
        raise
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"cursbreaker_{job_id[:8]}{tag}.zip",
        background=BackgroundTask(tmp_path.unlink, missing_ok=True),
    )


@app.get("/api/preview/{job_id}/{name}")
def preview(job_id: str, name: str):
    """Draw the hOCR line boxes over the page image so localization is visible."""
    img_path = _safe_output(job_id, name)
    out_dir = img_path.parent
    boxes = _line_boxes_for_image(out_dir, name)

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im, "RGBA")
        for (x0, y0, x1, y1) in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(220, 30, 90, 255), width=3)
            draw.rectangle([x0, y0, x1, y1], fill=(220, 30, 90, 28))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


def _line_boxes_for_image(out_dir: Path, image_name: str) -> list[tuple[int, int, int, int]]:
    ns = {"x": XHTML_NS}
    for hocr in out_dir.glob("*.hocr"):
        root = etree.fromstring(hocr.read_bytes())
        for page in root.xpath("//x:div[@class='ocr_page']", namespaces=ns):
            if f'image "{image_name}"' not in (page.get("title") or ""):
                continue
            boxes = []
            for line in page.xpath(".//x:span[@class='ocr_line']", namespaces=ns):
                title = line.get("title") or ""
                for part in title.split(";"):
                    part = part.strip()
                    if part.startswith("bbox "):
                        x0, y0, x1, y1 = (int(v) for v in part[5:].split()[:4])
                        boxes.append((x0, y0, x1, y1))
            return boxes
    return []


# --------------------------------------------------------------------------- #
# Browser heartbeat / auto-shutdown
# --------------------------------------------------------------------------- #
@app.post("/api/heartbeat")
def heartbeat(bye: bool = False):
    """The page pings every few seconds while it's open. ``bye=true`` is sent
    via ``navigator.sendBeacon`` on tab close, which pulls the last-seen time
    back so the watchdog fires shortly (unless another tab is still pinging)."""
    global _LAST_PING_AT
    now = time.time()
    if bye:
        _LAST_PING_AT = now - max(0.0, _SHUTDOWN_GRACE - 3.0)
    else:
        _LAST_PING_AT = now
    return {"ok": True}


_SHUTDOWN_GRACE = 15.0  # seconds without a ping before we quit
_SHUTDOWN_POLL = 2.0  # how often the watchdog checks


def _any_jobs_running() -> bool:
    return any(j.get("status") == "running" for j in JOBS.values())


def _should_shutdown(
    last_ping: float | None,
    grace: float,
    *,
    now: float | None = None,
    jobs_running: bool = False,
) -> bool:
    if jobs_running or last_ping is None:
        return False
    return ((time.time() if now is None else now) - last_ping) > grace


def _quit_process() -> None:
    """Polite shutdown first (uvicorn handles SIGINT gracefully); hard-exit as
    a backstop if uvicorn doesn't pick it up promptly."""
    try:
        signal.raise_signal(signal.SIGINT)
    except Exception:
        pass

    def _hard_exit() -> None:
        _cleanup_workspace()  # os._exit skips atexit, so drop the workspace here
        os._exit(0)

    threading.Timer(2.5, _hard_exit).start()


def start_autoshutdown(
    grace_seconds: float = _SHUTDOWN_GRACE, poll_seconds: float = _SHUTDOWN_POLL
) -> None:
    """Begin the watchdog thread. Safe to call multiple times (no-op after the
    first call)."""
    global _LAST_PING_AT, _AUTOSHUTDOWN_STARTED, _SHUTDOWN_GRACE, _SHUTDOWN_POLL
    if _AUTOSHUTDOWN_STARTED:
        return
    _AUTOSHUTDOWN_STARTED = True
    _SHUTDOWN_GRACE = grace_seconds
    _SHUTDOWN_POLL = poll_seconds
    # Don't arm the countdown yet -- leave the last-seen time unset until a
    # browser actually pings. Otherwise a failed browser-open (no tab ever
    # connects) would quietly quit the server ~grace seconds after launch, before
    # the user can open the printed URL by hand. _should_shutdown treats None as
    # "stay up", so the server waits indefinitely for the first connection.
    _LAST_PING_AT = None

    def loop():
        while True:
            time.sleep(poll_seconds)
            if _should_shutdown(
                _LAST_PING_AT, grace_seconds, jobs_running=_any_jobs_running()
            ):
                _quit_process()
                return

    threading.Thread(target=loop, name="cb-autoshutdown", daemon=True).start()


def install_access_log_filter() -> None:
    """Drop heartbeat hits from uvicorn's access log so the CLI stays quiet
    while the browser is just keeping the server alive."""
    import logging

    class _HideHeartbeat(logging.Filter):
        _cursbreaker_heartbeat = True

        def filter(self, record):  # noqa: A003 (logging API name)
            try:
                for arg in (record.args or ()):
                    if isinstance(arg, str) and "/api/heartbeat" in arg:
                        return False
                if "/api/heartbeat" in record.getMessage():
                    return False
            except Exception:
                pass
            return True

    logger = logging.getLogger("uvicorn.access")
    if not any(getattr(f, "_cursbreaker_heartbeat", False) for f in logger.filters):
        logger.addFilter(_HideHeartbeat())


import logging as _logging


class PrettyAccessFormatter(_logging.Formatter):
    """Drop-in replacement for uvicorn's access-log formatter that styles each
    line by HTTP status class, so successful requests don't visually read like
    warnings:

      * 2xx, 3xx -> green, prefixed with ``ok``
      * 4xx       -> yellow, prefixed with ``warn``
      * 5xx       -> red, prefixed with ``err``

    Colors auto-detect a TTY; the standard ``NO_COLOR`` env var disables them.
    Non-access log records (anything that isn't a 5-arg access tuple) fall
    through to the default Formatter so we never break unrelated log lines.
    """

    _RESET = "\x1b[0m"
    _GREEN = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED = "\x1b[31m"

    def __init__(self, *, use_colors=None):
        super().__init__()
        if use_colors is None:
            use_colors = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.use_colors = bool(use_colors)

    def format(self, record):  # noqa: A003 (logging API name)
        args = record.args or ()
        try:
            client_addr, method, full_path, http_version, status_code = args
            status = int(status_code)
        except (ValueError, TypeError):
            return super().format(record)

        if 200 <= status < 400:
            color, marker = self._GREEN, " ok "
        elif 400 <= status < 500:
            color, marker = self._YELLOW, "warn"
        else:
            color, marker = self._RED, " err"

        try:
            from http import HTTPStatus

            phrase = HTTPStatus(status).phrase
        except ValueError:
            phrase = ""

        line = (
            f'{marker}     {client_addr} - '
            f'"{method} {full_path} HTTP/{http_version}" '
            f'{status} {phrase}'
        ).rstrip()
        if self.use_colors:
            return f"{color}{line}{self._RESET}"
        return line


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/favicon.ico")
def favicon():
    # Drop a favicon.ico into src/cursbreaker/static/ to use a custom one;
    # otherwise reply 204 so browsers stop asking and we don't log a 404.
    f = STATIC_DIR / "favicon.ico"
    if f.is_file():
        return FileResponse(f, media_type="image/x-icon")
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text("utf-8")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
