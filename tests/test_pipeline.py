import errno
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from lxml import etree

from cursbreaker import tesseract_client
from cursbreaker.config import Settings
from cursbreaker.gemini_client import MockProvider
from cursbreaker.hocr import XHTML_NS
from cursbreaker.models import LineBox
from cursbreaker.pipeline import (
    estimate_usage,
    process_batch,
    process_file,
    process_page,
)

NS = {"x": XHTML_NS}


def _font(size: int = 36) -> "ImageFont.ImageFont":
    try:
        return ImageFont.truetype("src/cursbreaker/fonts/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _printed_page(tmp_path, name="printed.png", size=(900, 240)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = _font(36)
    draw.text((40, 30),  "First printed line",  fill="black", font=font)
    draw.text((40, 110), "Second printed line", fill="black", font=font)
    p = tmp_path / name
    img.save(p)
    return p


@pytest.mark.parametrize("mode", ["two_pass", "one_pass"])
def test_process_image_writes_outputs(png_path, tmp_path, mode):
    out = tmp_path / "out"
    settings = Settings(mode=mode)
    result = process_file(png_path, MockProvider(), settings, out)

    assert result.error is None
    assert result.n_pages == 1
    assert result.n_lines == 4  # the mock returns four lines

    txt = (out / "sample.txt").read_text("utf-8")
    assert "CursBreaker mock transcription" in txt

    hocr = (out / "sample.hocr").read_bytes()
    root = etree.fromstring(hocr)
    assert len(root.xpath("//x:span[@class='ocr_line']", namespaces=NS)) == 4
    # The exported page image is referenced and present on disk.
    assert (out / "sample.png").exists()
    assert 'image "sample.png"' in root.xpath(
        "//x:div[@class='ocr_page']/@title", namespaces=NS
    )[0]
    # A searchable PDF is written alongside the other outputs.
    assert (out / "sample.pdf").exists()
    assert result.pdf_name == "sample.pdf"

    # ALTO XML is written too, parses, and carries one TextLine per line.
    assert result.alto_name == "sample.alto.xml"
    alto_root = etree.fromstring((out / "sample.alto.xml").read_bytes())
    ans = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
    assert len(alto_root.xpath("//a:TextLine", namespaces=ans)) == 4


def test_process_pdf_multipage(pdf_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings(pdf_dpi=120)
    result = process_file(pdf_path, MockProvider(), settings, out)

    assert result.n_pages == 2
    assert (out / "doc.txt").exists()
    assert (out / "doc_page_0001.png").exists()
    assert (out / "doc_page_0002.png").exists()

    root = etree.fromstring((out / "doc.hocr").read_bytes())
    assert len(root.xpath("//x:div[@class='ocr_page']", namespaces=NS)) == 2


def test_batch_isolates_failures(png_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings()
    missing = tmp_path / "does_not_exist.png"
    results = process_batch([png_path, missing], MockProvider(), settings, out)
    assert results[0].error is None
    assert results[1].error is not None  # bad file captured, batch survives


def test_content_type_text_uses_tesseract_only(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="text")  # no API key, no mock — no Gemini
    result = process_file(page, MockProvider(), settings, out)

    assert result.error is None
    assert result.n_pages == 1
    # Tesseract finds at least the two lines we drew.
    assert result.n_lines >= 2

    txt = (out / page.stem + ".txt").read_text("utf-8").lower() if False else (
        (out / f"{page.stem}.txt").read_text("utf-8").lower()
    )
    assert "first" in txt and "second" in txt

    # hOCR carries REAL Tesseract per-word x_wconf (varies), not the nominal 95.
    hocr = (out / f"{page.stem}.hocr").read_text("utf-8")
    assert "ocr-system" in hocr
    assert "CursBreaker (text)" in hocr  # mode label flips to "text"


def test_content_type_text_errors_clearly_when_tesseract_missing(monkeypatch, tmp_path):
    # Force status -> not installed so we hit the require_available guard
    # regardless of what's actually installed in the test env.
    monkeypatch.setattr(
        tesseract_client,
        "status",
        lambda settings=None: tesseract_client.TesseractStatus(
            wrapper_present=True,
            binary_found=False,
            error="Tesseract OCR engine not found. Install it ...",
        ),
    )
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="text")
    # process_batch is the layer that catches per-file errors; it must surface
    # the missing-Tesseract failure with a useful message instead of crashing.
    results = process_batch([page], MockProvider(), settings, out)
    assert results[0].error and "tesseract" in results[0].error.lower()


class _MatchingProvider(MockProvider):
    """A mock whose Gemini transcription matches the printed test page, so the
    refine step has something Tesseract can agree with."""

    _TEXT = "First printed line\nSecond printed line"

    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str:
        return self._TEXT

    def detect_lines(self, image_png: bytes, mime: str = "image/png"):
        return [
            LineBox(text="First printed line", box_2d=[110, 40, 300, 960]),
            LineBox(text="Second printed line", box_2d=[440, 40, 640, 960]),
        ]


def _load_one(page, **overrides):
    from cursbreaker.images import load_pages

    return load_pages(
        page,
        preprocess=overrides.get("preprocess", True),
        max_dimension=overrides.get("max_dimension", 0),
        pdf_dpi=overrides.get("pdf_dpi", 300),
    )[0]


def test_refine_word_boxes_adopts_real_tesseract_boxes_keeping_gemini_text(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    page = _printed_page(tmp_path)
    settings = Settings(
        content_type="handwriting", mode="two_pass", refine_word_boxes=True
    )
    loaded = _load_one(page)
    result = process_page(loaded, _MatchingProvider(), settings)

    line = next(l for l in result.lines if l.text == "First printed line")
    # Refinement attached real per-word data; the *text* is still Gemini's words.
    assert line.words is not None
    assert [w.text for w in line.words] == ["First", "printed", "line"]
    # Every word matched a Tesseract word, so all carry the matched confidence
    # (synthesized fallback boxes would use interpolated_confidence instead).
    assert all(w.confidence == settings.word_confidence for w in line.words)
    # Real boxes advance left-to-right and sit within the page.
    xs = [w.box.x0 for w in line.words]
    assert xs == sorted(xs)
    assert all(0 <= w.box.x0 < w.box.x1 <= loaded.sent_width for w in line.words)


def test_refine_off_leaves_word_boxes_unsynthesized(tmp_path):
    # With the toggle off, the handwriting flow is unchanged: no per-word data,
    # so hOCR falls back to proportional splitting (words=None).
    page = _printed_page(tmp_path)
    settings = Settings(content_type="handwriting", refine_word_boxes=False)
    loaded = _load_one(page)
    result = process_page(loaded, _MatchingProvider(), settings)
    assert all(l.words is None for l in result.lines)


def test_refine_never_changes_transcription_text(tmp_path):
    # The whole point: the plain-text transcription must be byte-for-byte the
    # Gemini output regardless of what Tesseract reads.
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    page = _printed_page(tmp_path)
    loaded = _load_one(page)
    on = process_page(
        loaded,
        _MatchingProvider(),
        Settings(content_type="handwriting", refine_word_boxes=True),
    )
    off = process_page(
        loaded,
        _MatchingProvider(),
        Settings(content_type="handwriting", refine_word_boxes=False),
    )
    assert on.plain_text == off.plain_text == _MatchingProvider._TEXT


def test_refine_word_boxes_label_in_hocr(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(
        content_type="handwriting", mode="two_pass", refine_word_boxes=True
    )
    process_file(page, _MatchingProvider(), settings, out)
    hocr = (out / f"{page.stem}.hocr").read_text("utf-8")
    assert "handwriting/two_pass+wordboxes" in hocr


# --- token usage + cost estimate ----------------------------------------- #

class _BillingProvider(MockProvider):
    """A mock that 'bills' a fixed amount per Gemini call so token deltas are
    non-zero, and reports a fixed input-token count for estimates."""

    def transcribe_text(self, image_png, mime="image/png"):
        self.usage.add_response({"prompt_token_count": 100, "candidates_token_count": 20})
        return super().transcribe_text(image_png, mime)

    def detect_lines(self, image_png, mime="image/png"):
        self.usage.add_response({"prompt_token_count": 100, "candidates_token_count": 5})
        return super().detect_lines(image_png, mime)

    def transcribe_with_boxes(self, image_png, mime="image/png"):
        self.usage.add_response({"prompt_token_count": 100, "candidates_token_count": 25})
        return super().transcribe_with_boxes(image_png, mime)

    def count_input_tokens(self, image_png, mime="image/png"):
        return 500


def test_process_file_records_per_file_token_usage(png_path, tmp_path):
    out = tmp_path / "out"
    prov = _BillingProvider()
    settings = Settings(content_type="handwriting", mode="two_pass")
    result = process_file(png_path, prov, settings, out)
    # Two-pass on one page = transcribe + detect = 2 calls.
    assert result.token_usage.calls == 2
    assert result.token_usage.input == 200
    assert result.token_usage.output == 25


def test_per_file_usage_is_a_delta_not_the_running_total(png_path, tmp_path):
    out = tmp_path / "out"
    prov = _BillingProvider()  # shared across both files
    settings = Settings(content_type="handwriting", mode="one_pass")
    first = process_file(png_path, prov, settings, out)
    second = process_file(png_path, prov, settings, out)
    # Each file reports only its own one call, even though the provider's
    # running total has grown to two.
    assert first.token_usage.calls == 1
    assert second.token_usage.calls == 1
    assert prov.usage.calls == 2


def test_estimate_text_mode_is_free(png_path):
    d = estimate_usage([png_path], _BillingProvider(), Settings(content_type="text"))
    assert d["calls"] == 0
    assert d["input"] == 0
    assert d["output_low"] == 0 and d["output_high"] == 0
    assert d["cost_low"] is None and d["cost_high"] is None


def test_estimate_counts_two_calls_per_page_in_two_pass(png_path):
    settings = Settings(content_type="handwriting", mode="two_pass")
    d = estimate_usage([png_path], _BillingProvider(), settings)
    assert d["pages"] == 1
    assert d["calls"] == 2                 # transcribe + detect
    assert d["input"] == 500 * 1 * 2       # per-page input * pages * calls
    # two-pass output is a per-page range: 3000 (sparse) .. 9000 (dense)
    assert d["output_low"] == 3000 and d["output_high"] == 9000
    assert d["per_page_low"] == 3000 and d["per_page_high"] == 9000


def test_estimate_two_pass_output_exceeds_one_pass(png_path):
    # Two-pass generates the page's text twice (plain + structured boxes), so its
    # output estimate must exceed one-pass at both ends of the range.
    base = dict(content_type="handwriting")
    two = estimate_usage([png_path], _BillingProvider(), Settings(mode="two_pass", **base))
    one = estimate_usage([png_path], _BillingProvider(), Settings(mode="one_pass", **base))
    assert (one["output_low"], one["output_high"]) == (1800, 5400)
    assert (two["output_low"], two["output_high"]) == (3000, 9000)
    assert two["output_high"] > one["output_high"]


def test_estimate_scales_input_by_page_count(pdf_path):
    settings = Settings(content_type="handwriting", mode="one_pass", pdf_dpi=120)
    d = estimate_usage([pdf_path], _BillingProvider(), settings)
    assert d["pages"] == 2
    assert d["calls"] == 2                  # 2 pages * 1 call
    assert d["input"] == 500 * 2 * 1        # first-page count scaled by 2 pages


def test_estimate_sums_across_multiple_files(png_path, tmp_path):
    # Files are estimated concurrently; the totals must still sum correctly.
    f2 = tmp_path / "b.png"; f2.write_bytes(png_path.read_bytes())
    f3 = tmp_path / "c.png"; f3.write_bytes(png_path.read_bytes())
    settings = Settings(content_type="handwriting", mode="one_pass")
    d = estimate_usage([png_path, f2, f3], _BillingProvider(), settings)
    assert d["files"] == 3
    assert d["pages"] == 3
    assert d["input"] == 500 * 3            # 500/page * 3 single-page files
    assert d["output_low"] == 1800 * 3 and d["output_high"] == 5400 * 3


def test_estimate_prices_automatically_from_selected_model(png_path):
    # Flat-priced model: cost comes straight from the catalog, no manual entry.
    settings = Settings(
        content_type="handwriting",
        mode="one_pass",
        transcription_model="gemini-3.5-flash",  # $1.50 in / $9.00 out
    )
    d = estimate_usage([png_path], _BillingProvider(), settings)
    # input = 500 tokens; one-pass output range = 1800..5400 tokens/page.
    expected_low = 500 / 1_000_000 * 1.50 + 1800 / 1_000_000 * 9.00
    expected_high = 500 / 1_000_000 * 1.50 + 5400 / 1_000_000 * 9.00
    assert d["cost_low"] == pytest.approx(expected_low)
    assert d["cost_high"] == pytest.approx(expected_high)
    assert d["cost_high"] > d["cost_low"]
    assert d["output_low"] == 1800 and d["output_high"] == 5400
    # The model + the rates it used are echoed back for the UI to display.
    assert d["model"] == "gemini-3.5-flash"
    assert d["price_input_per_mtok"] == 1.50
    assert d["price_output_per_mtok"] == 9.00
    assert d["prices_as_of"]


def test_estimate_no_cost_for_uncatalogued_model(png_path):
    settings = Settings(
        content_type="handwriting", mode="one_pass", transcription_model="some-old-model"
    )
    d = estimate_usage([png_path], _BillingProvider(), settings)
    assert d["cost_low"] is None and d["cost_high"] is None  # unknown price
    assert d["input"] == 500


# --- progress reporting -------------------------------------------------- #

def _collect_progress(paths, provider, settings, out, units_total=0):
    events = []
    process_batch(paths, provider, settings, out, events.append, units_total=units_total)
    return events


def test_progress_reports_step_sequence_two_pass(pdf_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings(content_type="handwriting", mode="two_pass", pdf_dpi=120)
    events = _collect_progress([pdf_path], MockProvider(), settings, out, units_total=2)
    msgs = [e.message for e in events]
    joined = "\n".join(msgs)
    for needle in [
        "Loading", "2 page(s) to transcribe",
        "Page 1/2 · transcribing (Gemini)", "Page 1/2 · locating lines (Gemini)", "Page 1/2 done",
        "Page 2/2 · transcribing", "Page 2/2 · locating", "Page 2/2 done",
        "Writing outputs", "Done — 2 page(s)",
    ]:
        assert needle in joined, needle
    assert "refining word positions" not in joined  # refine off by default
    # The page counter advances once per finished page and ends at 2/2.
    page_done = [e for e in events if e.stage == "page_done"]
    assert [e.units_done for e in page_done] == [1, 2]
    assert all(e.units_total == 2 for e in events)
    # Single-file job: no filename prefix.
    assert all(not m.startswith(pdf_path.name) for m in msgs)


def test_progress_one_pass_single_call(png_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings(content_type="handwriting", mode="one_pass")
    events = _collect_progress([png_path], MockProvider(), settings, out, units_total=1)
    joined = "\n".join(e.message for e in events)
    assert "Page 1/1 · transcribing + locating (Gemini)" in joined
    assert "locating lines (Gemini)" not in joined  # no separate detect step


def test_progress_text_mode_uses_tesseract_wording(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="text")
    events = _collect_progress([page], MockProvider(), settings, out, units_total=1)
    joined = "\n".join(e.message for e in events)
    assert "reading text (Tesseract)" in joined
    assert "Gemini" not in joined


def test_progress_batch_prefixes_filenames(png_path, tmp_path):
    out = tmp_path / "out"
    second = tmp_path / "second.png"
    second.write_bytes(png_path.read_bytes())
    settings = Settings(content_type="handwriting", mode="one_pass")
    events = _collect_progress([png_path, second], MockProvider(), settings, out, units_total=2)
    page_lines = [e.message for e in events if e.stage == "page"]
    assert any(m.startswith(png_path.name + " · ") for m in page_lines)
    assert any(m.startswith("second.png · ") for m in page_lines)
    assert any(e.stage == "batch_done" for e in events)  # batch cap line


def test_failed_file_still_advances_page_counter(png_path, tmp_path):
    """A file counted in the up-front total but that errors never emits a
    ``page_done``; the counter must still reconcile to ``done == total`` at the
    file boundary so the bar reaches 100% instead of freezing partway (the
    "stuck at 558/830 while the log keeps flowing" bug)."""
    out = tmp_path / "out"
    bad = tmp_path / "unreadable.xyz"   # unsupported -> process_file raises
    bad.write_bytes(b"not an image")
    settings = Settings(content_type="handwriting", mode="one_pass")
    events = []
    # Budget one page each; only the good file can ever report a page_done.
    process_batch(
        [png_path, bad], MockProvider(), settings, out, events.append,
        units_total=2, unit_counts=[1, 1],
    )
    assert any(e.stage == "error" for e in events)   # the bad file is surfaced
    assert [e.units_done for e in events if e.stage == "page_done"] == [1]
    # The final event reconciles the counter past the failed file: 2/2, not 1/2.
    assert events[-1].units_done == 2
    assert events[-1].units_total == 2


def test_page_counter_without_unit_counts_is_unchanged(png_path, tmp_path):
    """Back-compat: with no ``unit_counts`` the counter only advances on real
    ``page_done`` events (a failed file leaves the bar short, as before)."""
    out = tmp_path / "out"
    bad = tmp_path / "unreadable.xyz"
    bad.write_bytes(b"not an image")
    settings = Settings(content_type="handwriting", mode="one_pass")
    events = []
    process_batch([png_path, bad], MockProvider(), settings, out, events.append, units_total=2)
    assert events[-1].units_done == 1  # no reconciliation without the budget


# --- disk-full pause / resume / stop ------------------------------------- #

def test_is_disk_full_detects_enospc_through_exception_chain():
    from cursbreaker.pipeline import _is_disk_full
    assert _is_disk_full(OSError(errno.ENOSPC, "No space left on device"))
    assert not _is_disk_full(OSError(errno.EACCES, "permission denied"))
    assert not _is_disk_full(ValueError("not a disk error"))
    try:
        try:
            raise OSError(errno.ENOSPC, "full")
        except OSError as inner:
            raise RuntimeError("could not write output") from inner
    except RuntimeError as wrapped:
        assert _is_disk_full(wrapped)  # found via the __cause__ chain


def test_disk_full_pauses_then_resume_retries_the_file(png_path, tmp_path, monkeypatch):
    """The first save hits ENOSPC; 'resume' retries the same file, which then
    succeeds — the page counter must account for it, not skip it."""
    real = process_file
    attempts = {"n": 0}

    def flaky(path, *a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, *a, **k)

    monkeypatch.setattr("cursbreaker.pipeline.process_file", flaky)
    paused = {"n": 0}

    def on_disk_full():
        paused["n"] += 1
        return "resume"

    out = tmp_path / "out"
    events = []
    results = process_batch(
        [png_path], MockProvider(), Settings(mode="one_pass"), out, events.append,
        units_total=1, unit_counts=[1], on_disk_full=on_disk_full,
    )
    assert paused["n"] == 1                       # paused exactly once
    assert attempts["n"] == 2                      # the file was retried
    assert results and results[0].error is None    # the retry saved it
    assert any(e.stage == "paused" for e in events)
    assert events[-1].units_done == 1              # counter reached the total


def test_disk_full_stop_keeps_completed_and_halts_remaining(png_path, tmp_path, monkeypatch):
    """'end' at the pause stops the batch: the file that ran out of space is
    flagged, earlier files are kept, later files are never started (no wasted
    API calls)."""
    real = process_file
    started = []

    def flaky(path, *a, **k):
        started.append(Path(path).name)
        if Path(path).name == "b.png":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, *a, **k)

    monkeypatch.setattr("cursbreaker.pipeline.process_file", flaky)
    a = png_path
    b = tmp_path / "b.png"; b.write_bytes(png_path.read_bytes())
    c = tmp_path / "c.png"; c.write_bytes(png_path.read_bytes())
    out = tmp_path / "out"
    events = []
    results = process_batch(
        [a, b, c], MockProvider(), Settings(mode="one_pass"), out, events.append,
        units_total=3, unit_counts=[1, 1, 1], on_disk_full=lambda: "end",
    )
    assert "c.png" not in started                                   # 3rd file never started
    assert results[0].error is None                                 # 1st file kept
    assert any(r.error and "disk" in r.error.lower() for r in results)  # 2nd flagged
    assert any(e.stage == "stopped" for e in events)                # clear terminal event


def test_disk_full_without_handler_keeps_old_skip_behavior(png_path, tmp_path, monkeypatch):
    """Back-compat: with no on_disk_full handler (e.g. CLI/tests), an ENOSPC is
    just a failed file and the batch continues."""
    real = process_file

    def flaky(path, *a, **k):
        if Path(path).name == "b.png":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, *a, **k)

    monkeypatch.setattr("cursbreaker.pipeline.process_file", flaky)
    a = png_path
    b = tmp_path / "b.png"; b.write_bytes(png_path.read_bytes())
    out = tmp_path / "out"
    results = process_batch(
        [a, b], MockProvider(), Settings(mode="one_pass"), tmp_path / "out", None,
        units_total=2, unit_counts=[1, 1],
    )
    assert len(results) == 2 and results[1].error is not None  # bad file recorded, batch went on


# --- output-format selection --------------------------------------------- #

def test_outputs_only_hocr_writes_just_the_hocr_document(png_path, tmp_path):
    out = tmp_path / "out"
    r = process_file(png_path, MockProvider(), Settings(mode="one_pass"), out, outputs={"hocr"})
    assert r.hocr_name and (out / r.hocr_name).exists()
    assert r.txt_name is None and r.alto_name is None and r.pdf_name is None
    # Page PNGs are always written (they back Preview), but aren't a document.
    assert r.image_names and all((out / n).exists() for n in r.image_names)
    docs = sorted(p.name for p in out.iterdir() if not p.name.endswith(".png"))
    assert docs == [r.hocr_name]  # hOCR is the only document on disk


def test_outputs_pdf_builds_from_retained_page_pngs(png_path, tmp_path):
    out = tmp_path / "out"
    r = process_file(png_path, MockProvider(), Settings(mode="one_pass"), out, outputs={"pdf"})
    assert r.pdf_name and (out / r.pdf_name).exists()
    assert r.txt_name is None and r.hocr_name is None and r.alto_name is None
    # PNGs are kept (not deleted) so Preview still works after a PDF-only run.
    assert r.image_names and all((out / n).exists() for n in r.image_names)


def test_page_images_written_even_for_minimal_text_output(png_path, tmp_path):
    out = tmp_path / "out"
    r = process_file(png_path, MockProvider(), Settings(mode="one_pass"), out, outputs={"txt"})
    assert r.txt_name and (out / r.txt_name).exists()
    assert r.image_names and all((out / n).exists() for n in r.image_names)


def test_images_is_not_a_selectable_output(png_path, tmp_path):
    from cursbreaker.pipeline import OUTPUT_FORMATS
    assert "images" not in OUTPUT_FORMATS
    # A stray "images" request is unrecognized, so it falls through to everything.
    out = tmp_path / "out"
    r = process_file(png_path, MockProvider(), Settings(mode="one_pass"), out, outputs=["images"])
    assert r.txt_name and r.hocr_name and r.alto_name and r.pdf_name


def test_outputs_empty_means_every_document(png_path, tmp_path):
    out = tmp_path / "out"
    r = process_file(png_path, MockProvider(), Settings(mode="one_pass"), out, outputs=[])
    assert r.txt_name and r.hocr_name and r.alto_name and r.pdf_name and r.image_names
    for n in (r.txt_name, r.hocr_name, r.alto_name, r.pdf_name, *r.image_names):
        assert (out / n).exists()


# --- searchable PDF preserves original quality --------------------------- #

def test_pdf_input_searchable_pdf_overlays_the_original(pdf_path, tmp_path):
    """A PDF input gets the text overlaid on a copy of the *original* PDF: the
    output keeps the source page size (no 300-dpi re-rasterization) and the
    original page content survives alongside the searchable layer."""
    import fitz
    out = tmp_path / "out"
    r = process_file(pdf_path, MockProvider(), Settings(mode="one_pass"), out, outputs={"pdf"})
    assert r.pdf_name
    with fitz.open(out / r.pdf_name) as doc:
        assert doc.page_count == 2
        # 612x792 source page preserved -> overlay, not a pixel-sized re-raster.
        assert round(doc[0].rect.width) == 612 and round(doc[0].rect.height) == 792
        assert "Page 1 text" in doc[0].get_text()  # original content survives


def test_image_input_searchable_pdf_embeds_full_resolution(png_path, tmp_path):
    """An image input embeds the full-resolution original, even when the OCR
    render was downscaled -- so the user never gets a down-scaled image back."""
    import fitz
    from PIL import Image
    ow, oh = Image.open(png_path).size  # 800x600
    out = tmp_path / "out"
    r = process_file(
        png_path, MockProvider(), Settings(mode="one_pass", max_dimension=200), out,
        outputs={"pdf"},
    )
    assert r.pdf_name
    with fitz.open(out / r.pdf_name) as doc:
        assert round(doc[0].rect.width) == ow and round(doc[0].rect.height) == oh


# --- skip our overlay for PDFs that already have a searchable text layer --- #

def _text_layer_pdf(path, n=2):
    """A PDF that already carries a real, substantial text layer (above the
    detection threshold), like a born-digital or already-OCR'd document."""
    import fitz
    doc = fitz.open()
    for _ in range(n):
        doc.new_page(width=612, height=792).insert_text(
            (72, 100), "This page already carries a real searchable text layer. " * 6
        )
    doc.save(path)
    doc.close()


def _image_only_pdf(path, n=2):
    """A scanned PDF with no text layer -- each page is just an image."""
    import fitz
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
    pix.clear_with(220)
    for _ in range(n):
        doc.new_page(width=612, height=792).insert_image(
            fitz.Rect(0, 0, 612, 792), pixmap=pix
        )
    doc.save(path)
    doc.close()


def test_skip_overlay_passes_already_searchable_pdf_through_unchanged(tmp_path):
    src = tmp_path / "searchable.pdf"
    _text_layer_pdf(src)
    original = src.read_bytes()
    out = tmp_path / "out"
    r = process_file(
        src, MockProvider(), Settings(mode="one_pass"), out,
        outputs={"pdf"}, skip_text_overlay=True,
    )
    assert r.pdf_name
    # Already searchable -> passed through byte-for-byte: no second overlay, no
    # size growth.
    assert (out / r.pdf_name).read_bytes() == original


def test_skip_overlay_still_overlays_a_scan_without_text(tmp_path):
    import fitz
    src = tmp_path / "scan.pdf"
    _image_only_pdf(src)
    original = src.read_bytes()
    out = tmp_path / "out"
    r = process_file(
        src, MockProvider(), Settings(mode="one_pass"), out,
        outputs={"pdf"}, skip_text_overlay=True,
    )
    # No existing layer -> the batch-wide skip doesn't apply to this file; our
    # searchable overlay is still added.
    assert (out / r.pdf_name).read_bytes() != original
    with fitz.open(out / r.pdf_name) as doc:
        assert doc[0].get_text().strip()  # now searchable


def test_without_skip_flag_overlays_even_an_already_searchable_pdf(tmp_path):
    src = tmp_path / "searchable.pdf"
    _text_layer_pdf(src)
    original = src.read_bytes()
    out = tmp_path / "out"
    r = process_file(
        src, MockProvider(), Settings(mode="one_pass"), out, outputs={"pdf"},
    )  # skip_text_overlay defaults to False -> behaviour unchanged
    assert (out / r.pdf_name).read_bytes() != original  # our overlay is applied


def test_progress_default_report_is_optional(png_path, tmp_path):
    # process_file/process_page still work with no reporter passed (back-compat).
    out = tmp_path / "out"
    result = process_file(png_path, MockProvider(), Settings(), out)
    assert result.error is None and result.n_pages == 1


# --- cooperative cancellation -------------------------------------------- #

class _CountingProvider(MockProvider):
    """Counts how many pages were actually transcribed (to prove cancellation
    stops the expensive work, not just the loop)."""

    def __init__(self):
        super().__init__()
        self.transcribe_calls = 0

    def transcribe_text(self, *a, **k):
        self.transcribe_calls += 1
        return super().transcribe_text(*a, **k)

    def transcribe_with_boxes(self, *a, **k):
        self.transcribe_calls += 1
        return super().transcribe_with_boxes(*a, **k)

    def detect_lines(self, *a, **k):
        return super().detect_lines(*a, **k)


def test_cancel_stops_before_next_file(png_path, tmp_path):
    out = tmp_path / "out"
    f2 = tmp_path / "b.png"; f2.write_bytes(png_path.read_bytes())
    f3 = tmp_path / "c.png"; f3.write_bytes(png_path.read_bytes())
    events = []
    settings = Settings(content_type="handwriting", mode="one_pass")
    # Cancel once the first file has fully finished (its .txt is written), so it
    # completes and the batch stops before the second file.
    done_first = out / "sample.txt"
    results = process_batch(
        [png_path, f2, f3], MockProvider(), settings, out, events.append,
        units_total=3, should_cancel=lambda: done_first.exists(),
    )
    assert len(results) == 1                       # only the first file finished
    assert results[0].error is None
    assert any(e.stage == "cancelled" for e in events)


def test_cancel_mid_file_drops_partial_and_skips_remaining_pages(pdf_path, tmp_path):
    out = tmp_path / "out"
    events = []
    prov = _CountingProvider()
    settings = Settings(content_type="handwriting", mode="one_pass", pdf_dpi=120)
    # 2-page file; cancel once page 1's PNG is saved -> page 2 is never rendered
    # or transcribed, and the whole partial file is abandoned.
    after_p1 = out / "doc_page_0001.png"
    results = process_batch(
        [pdf_path], prov, settings, out, events.append,
        units_total=2, should_cancel=lambda: after_p1.exists(),
    )
    assert results == []
    assert prov.transcribe_calls == 1              # page 2 never processed
    assert any(e.stage == "cancelled" for e in events)


def test_cancel_immediately_does_no_work(pdf_path, tmp_path):
    # A cancel that's already set when the job starts renders/transcribes nothing
    # (the lazy loader is never advanced -- no minute-long blocking rasterize).
    out = tmp_path / "out"
    prov = _CountingProvider()
    results = process_batch(
        [pdf_path], prov, Settings(), out, should_cancel=lambda: True
    )
    assert results == []
    assert prov.transcribe_calls == 0


def test_cancel_during_first_pass_skips_second_call(png_path, tmp_path):
    # Two-pass: a cancel that lands during the first Gemini call (transcribe)
    # must prevent the second (locate) call -- so the user isn't billed for a
    # request they no longer want, and cancellation is prompt.
    out = tmp_path / "out"
    state = {"cancel": False, "detect_calls": 0}

    class _P(MockProvider):
        def transcribe_text(self, *a, **k):
            state["cancel"] = True            # user cancels while pass 1 runs
            return super().transcribe_text(*a, **k)

        def detect_lines(self, *a, **k):
            state["detect_calls"] += 1
            return super().detect_lines(*a, **k)

    settings = Settings(content_type="handwriting", mode="two_pass")
    results = process_batch(
        [png_path], _P(), settings, out, should_cancel=lambda: state["cancel"],
    )
    assert results == []
    assert state["detect_calls"] == 0          # the second, billed call never ran


def test_no_cancel_completes_normally(png_path, tmp_path):
    out = tmp_path / "out"
    results = process_batch(
        [png_path], MockProvider(), Settings(), out, should_cancel=lambda: False
    )
    assert len(results) == 1 and results[0].error is None
