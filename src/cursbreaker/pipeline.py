"""Batch orchestration: input files -> .txt + .hocr (+ page PNGs).

Outputs are written to a dedicated directory (the web server uses a per-job temp
dir) so the user's source files are never touched or overwritten. Each page is
re-rendered to PNG at the dimensions the bounding boxes refer to, so the
``.hocr`` and its image always pair correctly and travel together.
"""

from __future__ import annotations

import errno
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection

from .align import align_lines, align_words
from .alto import build_alto
from .config import Settings
from .gemini_client import TranscriptionProvider
from .hocr import build_hocr, normalized_to_pixel
from .images import (
    PDF_EXT,
    blank_png,
    count_content_pages,
    iter_pages,
    load_output_images,
    load_pages,
    pdf_has_text_layer,
)
from .models import OcrWord, PageResult, PixelBox, PlacedLine, TokenUsage, TranscribedLine
from .pricing import PRICES_AS_OF, cost_for, effective_rates, pricing_for
from .searchable_pdf import (
    write_searchable_pdf_from_images,
    write_searchable_pdf_over_source,
)
from . import tesseract_client

@dataclass(frozen=True)
class ProgressEvent:
    """One step of work, reported live as a batch runs.

    ``units_done``/``units_total`` are *pages* across the whole batch (the
    progress bar is page-driven). ``stage`` is a machine tag
    ("load" | "page" | "page_done" | "write" | "file_done" | "error" |
    "batch_done") and ``message`` is the human line shown in the activity log."""

    message: str
    stage: str
    file_index: int
    file_total: int
    file_name: str
    units_done: int
    units_total: int


# Top-level reporter the server supplies; receives one ProgressEvent per step.
Reporter = Callable[["ProgressEvent"], None]
# The per-file line emitter threaded into process_file/process_page: a callable
# ``report(message, *, stage="page")`` built by process_batch (which injects the
# file context and the running page counter). Defaults to a no-op everywhere so
# direct callers (e.g. tests) need not pass one.
StepReporter = Callable[..., None]


def _noop(*_args, **_kwargs) -> None:
    pass


# Cooperative-cancellation predicate, checked at page/file boundaries (a
# synchronous Gemini call can't be interrupted mid-flight).
CancelCheck = Callable[[], bool]


class JobCancelled(Exception):
    """Raised inside ``process_file`` when ``should_cancel()`` turns true, so the
    partially-processed file is abandoned cleanly (not recorded as an error)."""


# Invoked when a file's output write fails because the disk is full. It BLOCKS
# until the user decides, then returns "resume" (space freed -> retry the file)
# or "end" (stop now, keep the files already saved). A full disk is a whole-job
# condition, not one bad file, so the batch must pause -- not silently skip every
# remaining file (writing nothing, still spending API tokens) as it used to.
DiskFullHandler = Callable[[], str]


def _is_disk_full(exc: BaseException) -> bool:
    """True if ``exc`` -- or any error it wraps -- is an ENOSPC "no space left on
    device". Output writes (page PNGs, txt/hOCR/ALTO/PDF) can surface it from deep
    inside Pillow/pikepdf, so walk the ``__cause__``/``__context__`` chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError) and cur.errno == errno.ENOSPC:
            return True
        cur = cur.__cause__ or cur.__context__
    return False

# Pre-flight output-token estimate, per page, as a LOW-HIGH range. Output is
# genuinely unpredictable before a page is read: it scales with how much text is
# on the page (sparse page -> little output -> cheaper; dense page -> lots ->
# pricier), and nothing measurable up front predicts it -- real handwriting docs
# we measured ran ~3,900-8,000 output tokens/page in two-pass at the SAME
# input/page. So rather than a single number that's ~2x off for some documents,
# we estimate a range that brackets the realistic spread; the UI shows it as a
# range and explains that denser pages cost more. One-pass makes a single
# structured call (~0.6x of two-pass, which also emits the plain text on its own
# first call).
_EST_OUTPUT_PER_PAGE_LOW = {"two_pass": 3000, "one_pass": 1800}
_EST_OUTPUT_PER_PAGE_HIGH = {"two_pass": 9000, "one_pass": 5400}


@dataclass
class FileResult:
    source_name: str
    n_pages: int = 0
    n_lines: int = 0
    txt_name: str | None = None
    hocr_name: str | None = None
    alto_name: str | None = None
    pdf_name: str | None = None
    image_names: list[str] = field(default_factory=list)
    error: str | None = None
    # Tokens this file billed to Gemini (zero for Printed-only mode).
    token_usage: TokenUsage = field(default_factory=TokenUsage)


def process_page(
    loaded,
    provider: TranscriptionProvider,
    settings: Settings,
    *,
    report: StepReporter | None = None,
    page_no: int = 0,
    page_total: int = 0,
    should_cancel: CancelCheck | None = None,
) -> PageResult:
    report = report or _noop
    # A cancel that landed while this page was rendering is caught here, before
    # the expensive transcription call is made.
    if should_cancel and should_cancel():
        raise JobCancelled()
    content = (settings.content_type or "handwriting").lower()
    if content == "text":
        return _process_page_text_only(
            loaded, settings, report=report, page_no=page_no, page_total=page_total
        )
    if content == "mixed":
        # Retired alias: "Gemini text + Tesseract" is now handwriting with word-
        # box refinement. Defensive, in case an un-migrated value reaches here.
        settings = settings.model_copy(
            update={"content_type": "handwriting", "refine_word_boxes": True}
        )
    return _process_page_handwriting(
        loaded, provider, settings,
        report=report, page_no=page_no, page_total=page_total,
        should_cancel=should_cancel,
    )


def _process_page_handwriting(
    loaded,
    provider: TranscriptionProvider,
    settings: Settings,
    *,
    report: StepReporter | None = None,
    page_no: int = 0,
    page_total: int = 0,
    should_cancel: CancelCheck | None = None,
) -> PageResult:
    """Gemini transcription (one-pass, or two-pass with line alignment). The
    Gemini text is always authoritative; when ``refine_word_boxes`` is set and
    Tesseract is available, real per-word boxes are layered on afterwards."""
    report = report or _noop
    pg = f"Page {page_no}/{page_total}"
    png = loaded.to_png_bytes()
    if settings.mode == "one_pass":
        report(f"{pg} · transcribing + locating (Gemini)…", stage="page")
        items = provider.transcribe_with_boxes(png)
        plain_text = "\n".join(i.text for i in items)
        placed: list[PlacedLine] = [
            PlacedLine(text=i.text, box_2d=list(i.box_2d), is_interpolated=False)
            for i in items
        ]
    else:
        report(f"{pg} · transcribing (Gemini)…", stage="page")
        text = provider.transcribe_text(png)
        # Cancelled during the first pass? Stop before the second (billed) call
        # so the user isn't charged for a request they no longer want.
        if should_cancel and should_cancel():
            raise JobCancelled()
        report(f"{pg} · locating lines (Gemini)…", stage="page")
        detected = provider.detect_lines(png)
        placed = align_lines(text.splitlines(), detected)
        plain_text = text

    # Skip the (local but per-line) Tesseract refinement if cancelled by now.
    if should_cancel and should_cancel():
        raise JobCancelled()
    w, h = loaded.sent_width, loaded.sent_height
    lines: list[TranscribedLine] = []
    for p in placed:
        conf = (
            settings.interpolated_confidence if p.is_interpolated
            else settings.word_confidence
        )
        lines.append(
            TranscribedLine(
                text=p.text,
                box=normalized_to_pixel(p.box_2d, w, h),
                confidence=conf,
            )
        )
    # Optional, purely additive: refine word *positions* with Tesseract. The
    # Gemini transcription above is already final; this only attaches real
    # per-word boxes where Tesseract agrees, and can never change the text.
    if settings.refine_word_boxes and tesseract_client.is_available(settings):
        report(f"{pg} · refining word positions (Tesseract)…", stage="page")
        _refine_word_boxes(loaded.image, lines, settings)
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _process_page_text_only(
    loaded,
    settings: Settings,
    *,
    report: StepReporter | None = None,
    page_no: int = 0,
    page_total: int = 0,
) -> PageResult:
    """Tesseract-only flow: no Gemini call, real per-word boxes throughout."""
    report = report or _noop
    report(f"Page {page_no}/{page_total} · reading text (Tesseract)…", stage="page")
    tesseract_client.require_available(settings)
    w, h = loaded.sent_width, loaded.sent_height
    lines = tesseract_client.transcribe_page(
        loaded.image, lang=settings.tesseract_language
    )
    plain_text = "\n".join(l.text for l in lines)
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _refine_word_boxes(
    page_image, lines: list[TranscribedLine], settings: Settings
) -> None:
    """In place: attach real Tesseract word boxes to each line where the engine's
    text agrees with Gemini's. Line text is never touched; any per-line failure
    is swallowed so that line simply keeps its proportional word boxes."""
    for line in lines:
        try:
            ocr_words = _tesseract_words_for_line(page_image, line.box, settings)
        except Exception:
            # Tesseract choking on one region must not lose that line's text.
            continue
        if not ocr_words:
            continue  # leave words=None -> hOCR splits the line box proportionally
        line.words = align_words(
            line.text,
            ocr_words,
            line.box,
            matched_conf=settings.word_confidence,
            fallback_conf=settings.interpolated_confidence,
        )


def _tesseract_words_for_line(
    page_image, line_box: PixelBox, settings: Settings
) -> list[OcrWord]:
    """OCR a small crop around ``line_box`` and return Tesseract's per-word boxes
    in page coordinates (PSM 7, since Gemini already isolated one line)."""
    # Pad a few pixels so we don't clip characters at the edges of Gemini's box.
    pad = 4
    x0 = max(0, line_box.x0 - pad)
    y0 = max(0, line_box.y0 - pad)
    x1 = min(page_image.width, line_box.x1 + pad)
    y1 = min(page_image.height, line_box.y1 + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    crop = page_image.crop((x0, y0, x1, y1))
    words: list[OcrWord] = []
    for tl in tesseract_client.transcribe_region(
        crop, lang=settings.tesseract_language, psm=7, offset=(x0, y0)
    ):
        words.extend(tl.words or [])
    return words


# Document output kinds a run can write. An empty/unknown selection means "all of
# them", so the default (and a malformed request) still produces everything. Page
# images are deliberately not here -- they're internal scaffolding (they back the
# box-overlay Preview and are embedded in the searchable PDF), never a download.
OUTPUT_FORMATS = ("txt", "hocr", "alto", "pdf")


def _wanted_outputs(outputs: Collection[str] | None) -> set[str]:
    """Normalize a requested output selection: empty/None -> every format."""
    if not outputs:
        return set(OUTPUT_FORMATS)
    wanted = {o for o in outputs if o in OUTPUT_FORMATS}
    return wanted or set(OUTPUT_FORMATS)


def process_file(
    path: str | Path,
    provider: TranscriptionProvider,
    settings: Settings,
    out_dir: Path,
    *,
    report: StepReporter | None = None,
    should_cancel: CancelCheck | None = None,
    outputs: Collection[str] | None = None,
    skip_text_overlay: bool = False,
) -> FileResult:
    report = report or _noop
    path = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Which document outputs to write (empty/None -> all).
    wanted = _wanted_outputs(outputs)

    # Snapshot the provider's running token total so we can report exactly what
    # *this* file added (the provider is shared across a batch).
    usage_before = _provider_usage(provider)

    report("Loading…", stage="load")
    # Page count from cheap metadata (no rasterization), so a 48-page PDF doesn't
    # block here for a minute before the first cancel check -- pages are rendered
    # lazily, one per loop, with a cancel check before each.
    total_pages = count_content_pages(path)
    report(f"{total_pages} page(s) to transcribe", stage="load")

    page_results: list[PageResult] = []
    image_names: list[str] = []  # page PNGs -- internal scaffolding (preview + PDF)
    pages = iter_pages(
        path,
        preprocess=settings.preprocess,
        max_dimension=settings.max_dimension,
        pdf_dpi=settings.pdf_dpi,
    )
    i = 0
    try:
        while True:
            # Cooperative cancellation, checked BEFORE rendering the next page
            # (the render happens inside next()). A second check inside
            # process_page catches a cancel that lands mid-render, before the
            # expensive transcription. Either abandons this file's partial work.
            if should_cancel and should_cancel():
                raise JobCancelled()
            try:
                loaded = next(pages)
            except StopIteration:
                break
            i += 1
            page_results.append(
                process_page(
                    loaded, provider, settings,
                    report=report, page_no=i, page_total=total_pages,
                    should_cancel=should_cancel,
                )
            )
            # Always write the page PNG: it backs the Preview overlay and the
            # searchable PDF. It is never offered as a download (derivative images
            # are out of scope), but it lives in the per-job temp workspace.
            png_name = f"{loaded.output_stem}.png"
            loaded.image.save(out_dir / png_name)
            image_names.append(png_name)
            # A page counts as "done" (advancing the bar) once it's transcribed;
            # process_batch increments the global page counter on this stage.
            report(f"Page {i}/{total_pages} done", stage="page_done")
    finally:
        # Close the lazy renderer (releases the open PDF) if we bailed early.
        pages.close()

    _labels = {"txt": "text", "hocr": "hOCR", "alto": "ALTO", "pdf": "searchable PDF"}
    writing = [_labels[f] for f in ("txt", "hocr", "alto", "pdf") if f in wanted]
    report(
        f"Writing outputs ({', '.join(writing)})…" if writing else "Finalizing…",
        stage="write",
    )
    stem = path.stem
    txt_name = hocr_name = alto_name = pdf_name = None  # set only for what we write

    if "txt" in wanted:
        txt_name = f"{stem}.txt"
        txt = "\n\n".join(p.plain_text.strip() for p in page_results)
        (out_dir / txt_name).write_text(txt, "utf-8")

    content = settings.content_type or "handwriting"
    if content == "text":
        mode_label = "text"
    else:
        mode_label = f"{content}/{settings.mode}"
        if settings.refine_word_boxes:
            mode_label += "+wordboxes"
    ocr_system = f"CursBreaker ({mode_label})"

    if "hocr" in wanted:
        hocr_name = f"{stem}.hocr"
        (out_dir / hocr_name).write_bytes(
            build_hocr(page_results, ocr_system=ocr_system, language=settings.language)
        )

    # ALTO XML carries the same line/word geometry as hOCR in the Library of
    # Congress's preservation format, for ALTO/METS-based repositories.
    if "alto" in wanted:
        alto_name = f"{stem}.alto.xml"
        (out_dir / alto_name).write_bytes(
            build_alto(page_results, ocr_system=ocr_system, language=settings.language)
        )

    if "pdf" in wanted:
        pdf_name = f"{stem}.pdf"
        pdf_out = out_dir / pdf_name
        if path.suffix.lower() in PDF_EXT:
            if skip_text_overlay and pdf_has_text_layer(path):
                # The source already carries a searchable text layer and the user
                # opted (batch-wide) to keep it rather than add ours: pass the
                # original through unchanged -- no second overlay, no size growth.
                shutil.copyfile(path, pdf_out)
            else:
                # Overlay the text on the *original* PDF -- original image quality,
                # page sizes and orientation preserved, no re-rasterization.
                write_searchable_pdf_over_source(path, page_results, pdf_out)
        else:
            # Embed the original source image(s) untouched (JPEG byte-for-byte;
            # others full-resolution and lossless) -- not the downscaled/enhanced
            # OCR render -- and overlay the text.
            write_searchable_pdf_from_images(
                load_output_images(path, dpi=settings.pdf_dpi), page_results, pdf_out
            )

    n_pages = len(page_results)
    n_lines = sum(len(p.lines) for p in page_results)
    report(f"Done — {n_pages} page(s), {n_lines} line(s)", stage="file_done")
    return FileResult(
        source_name=path.name,
        n_pages=n_pages,
        n_lines=n_lines,
        txt_name=txt_name,
        hocr_name=hocr_name,
        alto_name=alto_name,
        pdf_name=pdf_name,
        image_names=image_names,
        token_usage=_provider_usage(provider) - usage_before,
    )


def _provider_usage(provider: TranscriptionProvider) -> TokenUsage:
    """A copy of a provider's running token total, or an empty one if the
    provider doesn't track usage (keeps callers defensive)."""
    usage = getattr(provider, "usage", None)
    if isinstance(usage, TokenUsage):
        return usage.model_copy()
    return TokenUsage()


def process_batch(
    paths: list[str | Path],
    provider: TranscriptionProvider,
    settings: Settings,
    out_dir: Path,
    report: Reporter | None = None,
    units_total: int = 0,
    unit_counts: list[int] | None = None,
    should_cancel: CancelCheck | None = None,
    on_disk_full: DiskFullHandler | None = None,
    outputs: Collection[str] | None = None,
    skip_text_overlay: bool = False,
) -> list[FileResult]:
    """Transcribe each file, emitting a ``ProgressEvent`` per step to ``report``.

    ``units_total`` is the total page count across the batch (the bar is
    page-driven); the running "pages done" counter advances on each ``page_done``
    step. ``unit_counts`` is the per-file page budget (the up-front estimate whose
    sum is ``units_total``); when given, the page counter is reconciled to that
    budget at each file boundary so a file that errors -- or renders fewer pages
    than its metadata promised -- can't strand the bar below 100% (the symptom
    being a counter frozen at e.g. 558/830 while the log keeps flowing through the
    remaining files). Messages are prefixed with the filename only when the batch
    has more than one file. ``should_cancel`` (checked between files and pages)
    stops the batch cooperatively; files already finished keep their outputs.
    ``on_disk_full`` (if given) is called when a write fails for lack of disk
    space: it blocks until the user resumes (retry the file) or stops, so the
    batch doesn't burn through the rest of the files saving nothing."""
    results: list[FileResult] = []
    total = len(paths)
    state = {"done": 0}  # pages completed across the whole batch
    cancelled = False
    ended = False        # user chose "stop" at a disk-full pause: finish early
    budgeted = 0         # cumulative page budget of every file we've started

    for idx, path in enumerate(paths):
        name = Path(path).name
        prefix = f"{name} · " if total > 1 else ""

        # A per-file line emitter that injects file context + the running page
        # counter. ``page_done`` advances the global counter before the event is
        # built, so ``units_done`` reflects the just-finished page.
        def report_line(message, *, stage="page", _idx=idx, _name=name, _prefix=prefix):
            if stage == "page_done":
                state["done"] += 1
            if report:
                report(ProgressEvent(
                    message=_prefix + message, stage=stage,
                    file_index=_idx, file_total=total, file_name=_name,
                    units_done=state["done"], units_total=units_total,
                ))

        if should_cancel and should_cancel():  # stop before starting a new file
            cancelled = True
            break
        budgeted += unit_counts[idx] if unit_counts else 0

        while True:  # retry this file if a disk-full pause is resolved by "resume"
            try:
                results.append(
                    process_file(
                        path, provider, settings, out_dir,
                        report=report_line, should_cancel=should_cancel,
                        outputs=outputs, skip_text_overlay=skip_text_overlay,
                    )
                )
            except JobCancelled:  # cancelled mid-file: drop the partial file, stop
                cancelled = True
            except Exception as exc:
                # A full disk affects every remaining file. Don't quietly skip and
                # march through the rest of the batch saving nothing (and still
                # spending API tokens): pause, let the user free space and resume,
                # or stop with what's already saved.
                if on_disk_full is not None and _is_disk_full(exc):
                    report_line(
                        "Paused — no space left on the disk. Free up space, then "
                        "Resume, or Stop to keep the files already finished.",
                        stage="paused",
                    )
                    if on_disk_full() == "resume":   # blocks until the user decides
                        report_line(f"Resuming — retrying {name}.", stage="page")
                        continue                     # re-attempt the same file
                    ended = True                     # user chose to stop
                    results.append(
                        FileResult(source_name=name, error="Not saved — the disk was full.")
                    )
                else:
                    # A non-disk error is one bad file: record it and move on. It
                    # emits no "page_done" but was counted in the total, so advance
                    # the counter past its budget (monotonic max) instead of letting
                    # the bar stall on pages that will never report done.
                    results.append(FileResult(source_name=name, error=str(exc)))
                    if unit_counts:
                        state["done"] = max(state["done"], budgeted)
                    report_line(f"Failed: {exc}", stage="error")
            else:
                # Reconcile any drift between the page budget and what actually
                # rendered (metadata can over-count), so the bar tracks real
                # progress across a long batch and still reaches 100% at the end.
                if unit_counts:
                    state["done"] = max(state["done"], budgeted)
            break  # leave the retry loop unless we 'continue'd above

        if cancelled or ended:
            break

    completed = sum(1 for r in results if r.error is None)
    if report and cancelled:
        report(ProgressEvent(
            message="Cancelled.", stage="cancelled",
            file_index=0, file_total=total,
            file_name=Path(paths[0]).name if paths else "",
            units_done=state["done"], units_total=units_total,
        ))
    elif report and ended:
        report(ProgressEvent(
            message=(f"Stopped — out of disk space. {completed} file(s) finished "
                     "and ready to download."),
            stage="stopped",
            file_index=max(0, total - 1), file_total=total,
            file_name=Path(paths[-1]).name if paths else "",
            units_done=state["done"], units_total=units_total,
        ))
    elif report and total > 1:
        report(ProgressEvent(
            message="All files complete.", stage="batch_done",
            file_index=max(0, total - 1), file_total=total,
            file_name=Path(paths[-1]).name if paths else "",
            units_done=state["done"], units_total=units_total,
        ))
    return results


def estimate_usage(
    paths: list[str | Path],
    provider: TranscriptionProvider,
    settings: Settings,
) -> dict:
    """Pre-flight token/cost estimate for a set of files -- before any
    transcription runs, so the user can decide whether to proceed.

    *Input* tokens (dominated by the page image) are measured with the
    provider's free ``count_tokens`` on the first page of each file and scaled by
    the page count -- fast, and exact enough that the user can sanity-check the
    bill. *Output* tokens can't be known until the text is generated, so they use
    a single, surfaced per-call assumption. Two-pass sends the image twice, so it
    counts two calls per page; Printed-only runs make no Gemini call and estimate
    to zero. The returned dict is clearly labelled as an estimate by the UI,
    which also shows the per-million prices used."""
    content = (settings.content_type or "handwriting").lower()
    if content == "text":
        calls_per_page = 0
    else:
        calls_per_page = 1 if settings.mode == "one_pass" else 2

    def _estimate_file(path) -> tuple[int, int]:
        """Return (page_count, input_token_estimate) for one file. Kept fast so a
        batch can run these concurrently."""
        path = Path(path)
        try:
            n_pages = count_content_pages(path)
        except Exception:
            n_pages = 1
        if calls_per_page == 0:
            return n_pages, 0
        try:
            # Render only the first page, and WITHOUT preprocessing: the slow
            # median-filter denoise doesn't change the image's dimensions, and
            # count_tokens depends only on dimensions, so skipping it is free
            # accuracy-wise but much faster.
            first = load_pages(
                path,
                preprocess=False,
                max_dimension=settings.max_dimension,
                pdf_dpi=settings.pdf_dpi,
                max_pages=1,
            )[0]
            # Probe with a blank image of the same size: same (geometric) token
            # count, a few KB to upload instead of a multi-MB scan.
            per_page_input = provider.count_input_tokens(blank_png(first.image.size))
        except Exception:
            per_page_input = 0
        return n_pages, per_page_input * n_pages * calls_per_page

    # Files are independent; overlap their (network-bound) count_tokens calls.
    input_tokens = 0
    total_pages = 0
    if paths:
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as ex:
            for n_pages, file_input in ex.map(_estimate_file, paths):
                total_pages += n_pages
                input_tokens += file_input

    calls = total_pages * calls_per_page
    # Output is a per-page RANGE (sparse vs dense pages); see the constants above.
    mode = "one_pass" if settings.mode == "one_pass" else "two_pass"
    if calls_per_page == 0:
        per_page_low = per_page_high = 0
    else:
        per_page_low = _EST_OUTPUT_PER_PAGE_LOW[mode]
        per_page_high = _EST_OUTPUT_PER_PAGE_HIGH[mode]
    output_low = total_pages * per_page_low
    output_high = total_pages * per_page_high

    # Price automatically from the selected model's published rate (no manual
    # entry). No catalog entry, or no calls -> tokens only, no dollar figure.
    pricing = pricing_for(settings.transcription_model)

    def _cost(output):
        if not (pricing and calls):
            return None
        return cost_for(pricing, TokenUsage(input=input_tokens, output=output, calls=calls))

    cost_low = _cost(output_low)
    cost_high = _cost(output_high)
    if pricing and calls:
        # Tier (effective rate) is set by the per-call prompt size, identical for
        # both ends, so either gives the rates to display.
        in_rate, out_rate = effective_rates(
            pricing, TokenUsage(input=input_tokens, output=output_high, calls=calls)
        )
    else:
        in_rate = out_rate = 0.0
    return {
        "files": len(paths),
        "pages": total_pages,
        "calls": calls,
        "calls_per_page": calls_per_page,
        "input": input_tokens,
        # Output (and thus cost) is a range -- assumed, see per_page_low/high.
        "output_low": output_low,
        "output_high": output_high,
        "total_low": input_tokens + output_low,
        "total_high": input_tokens + output_high,
        "per_page_low": per_page_low,
        "per_page_high": per_page_high,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "model": settings.transcription_model,
        "model_label": pricing.label if pricing else settings.transcription_model,
        "price_input_per_mtok": in_rate,
        "price_output_per_mtok": out_rate,
        "prices_as_of": PRICES_AS_OF,
    }
