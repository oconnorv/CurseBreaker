"""Build a searchable PDF: page image + invisible OCR text overlay.

Each PDF page shows the rendered page image and carries an *invisible*
text layer placed at the bounding-box positions. Users see the original
handwriting, and any PDF viewer can select/search the transcribed text
on top of it.

The text uses PyMuPDF's ``render_mode=3``, which writes the characters
into the PDF content stream without filling or stroking them: invisible
on screen, but harvested by every PDF text extractor and search index.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import fitz

from .hocr import split_line_into_words
from .models import PageResult

_FONT_NAME = "cb-uni"
_FONT_PATH = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"
_TEXTWRITER_FONT: fitz.Font | None = None


def _overlay_font() -> fitz.Font:
    """The bundled Unicode font as a reusable ``fitz.Font`` for ``TextWriter``,
    parsed once and shared across pages/documents. Falls back to PyMuPDF's
    built-in Helvetica if the file is somehow missing."""
    global _TEXTWRITER_FONT
    if _TEXTWRITER_FONT is None:
        try:
            _TEXTWRITER_FONT = fitz.Font(fontfile=str(_FONT_PATH))
        except Exception:
            _TEXTWRITER_FONT = fitz.Font("helv")
    return _TEXTWRITER_FONT


def _register_overlay_font(page) -> str:
    """Make the bundled Unicode font available on this page (for the per-word
    ``insert_text`` fallback), falling back to Helvetica if the file is missing."""
    if _FONT_PATH.is_file():
        try:
            page.insert_font(fontname=_FONT_NAME, fontfile=str(_FONT_PATH))
            return _FONT_NAME
        except Exception:
            pass
    return "helv"


def _iter_overlay_words(page_result: PageResult):
    """Yield ``(word, box)`` for every non-empty word box on the page. Prefer real
    per-word data (e.g. Tesseract) when present, so selection/search land on the
    actual word boxes; otherwise split each line box proportionally (the only
    option for Gemini-sourced lines)."""
    for line in page_result.lines:
        if line.words:
            items = [(w.text, w.box) for w in line.words]
        else:
            items = split_line_into_words(line.text, line.box)
        for word, wbox in items:
            if wbox.x1 > wbox.x0 and wbox.y1 > wbox.y0:
                yield word, wbox


def _overlay_text(pdf_page, page_result: PageResult) -> None:
    """Lay the invisible (``render_mode=3``) OCR text onto an open PDF page.

    Box coordinates are in the OCR image's pixel space (the upright render the
    model saw); we scale them to the page's displayed size. On an unrotated page
    all words are written with a single ``TextWriter`` -- one content stream for
    the whole page. The previous approach called ``insert_text`` once per word,
    which emitted a separate PDF text object per word (hundreds on a dense page)
    and bloated a text-heavy document by ~50x.

    Rotated pages (and the rare case where ``TextWriter`` rejects some input)
    fall back to per-word ``insert_text``, mapping each point from displayed ->
    unrotated coordinates via the derotation matrix and passing the page rotation
    -- placement verified for 90/180/270. A trailing space after each word keeps
    tightly packed boxes from merging ("the cat" -> "thecat") on extraction."""
    rect = pdf_page.rect  # displayed size, with any rotation already applied
    if not page_result.width or not page_result.height:
        return
    sx = rect.width / page_result.width
    sy = rect.height / page_result.height
    rot = pdf_page.rotation

    if rot == 0:
        try:
            tw = fitz.TextWriter(rect)
            font = _overlay_font()
            wrote = False
            for word, wbox in _iter_overlay_words(page_result):
                fontsize = max(1.0, (wbox.y1 - wbox.y0) * sy * 0.7)
                # Baseline at the box's bottom-left (top-left origin).
                tw.append(
                    fitz.Point(wbox.x0 * sx, wbox.y1 * sy), word + " ",
                    font=font, fontsize=fontsize,
                )
                wrote = True
            if wrote:
                tw.write_text(pdf_page, render_mode=3)
            return
        except Exception:
            # TextWriter can reject unusual input; nothing is committed to the
            # page until write_text(), so fall through and lay it down per-word.
            pass

    derot = pdf_page.derotation_matrix
    font = _register_overlay_font(pdf_page)
    for word, wbox in _iter_overlay_words(page_result):
        fontsize = max(1.0, (wbox.y1 - wbox.y0) * sy * 0.7)
        point = fitz.Point(wbox.x0 * sx, wbox.y1 * sy) * derot
        pdf_page.insert_text(
            point, word + " ",
            fontsize=fontsize, fontname=font, render_mode=3, rotate=rot,
        )


def write_searchable_pdf_over_source(
    src_pdf_path: str | Path, pages: list[PageResult], out_path: str | Path
) -> None:
    """Searchable PDF for a *PDF* input: overlay the invisible OCR text onto a
    copy of the original PDF, preserving its images/vectors, page sizes and
    orientation exactly -- no re-rasterization, so the user keeps their original
    image quality. ``pages[i]`` corresponds to source page ``i``.

    The text layer is *appended* to a byte-for-byte copy of the source with an
    incremental save: PyMuPDF writes only the new overlay objects (a few KB per
    page) to the end of the file, never rewriting or serializing the whole
    document. Memory stays flat no matter how large the PDF is -- the earlier
    approach built the entire PDF in RAM (``doc.tobytes()``) and could raise
    ``MemoryError`` on big files -- and the user's original bytes are left
    untouched (the output is the original with the searchable layer appended).
    If a particular PDF can't be saved incrementally (e.g. PyMuPDF had to repair
    it on open), we fall back to a full save, still streamed to a file rather
    than held in memory."""
    src_pdf_path = Path(src_pdf_path)
    out_path = Path(out_path)
    # Validate against the source before creating any output file.
    with fitz.open(str(src_pdf_path)) as src:
        if len(pages) > src.page_count:
            raise ValueError("more OCR pages than the source PDF has")
    # Copy first, then open the copy: an incremental save can only append to a
    # file that already holds the original, and this keeps the source read-only.
    shutil.copyfile(src_pdf_path, out_path)
    doc = fitz.open(str(out_path))
    tmp = None
    try:
        for i, page_result in enumerate(pages):
            _overlay_text(doc[i], page_result)
        try:
            # Append only our new objects to the copied original; deflate
            # compresses the added text streams. Existing page images are
            # untouched (not re-encoded).
            doc.save(
                str(out_path),
                incremental=True,
                encryption=fitz.PDF_ENCRYPT_KEEP,
                deflate=True,
            )
        except Exception:
            # Some inputs can't be saved incrementally (e.g. a PDF PyMuPDF had to
            # repair on open). Full-save to a sibling temp file and swap it in
            # after closing -- a full save streams to the file, so still no
            # whole-document blob in memory, and closing first lets the replace
            # succeed on Windows, where an open file can't be replaced.
            tmp = out_path.with_name(out_path.name + ".tmp")
            doc.save(str(tmp), garbage=3, deflate=True)
    finally:
        doc.close()
    if tmp is not None:
        os.replace(tmp, out_path)


def write_searchable_pdf_from_images(
    images, pages: list[PageResult], out_path: str | Path
) -> None:
    """Searchable PDF for *image* inputs: one page per full-resolution source
    image, with the invisible OCR text overlaid. ``images`` are ``OutputImage``
    specs: a passthrough one embeds the user's original bytes untouched (a JPEG
    stays byte-for-byte identical), a decoded one embeds full-resolution pixels
    losslessly -- never the downscaled/enhanced OCR render."""
    if len(images) != len(pages):
        raise ValueError("images and pages must have the same length")
    doc = fitz.open()
    try:
        for spec, page_result in zip(images, pages):
            w, h = spec.width, spec.height
            page = doc.new_page(width=w, height=h)
            rect = fitz.Rect(0, 0, w, h)
            if spec.data is not None:
                # Embed the original encoded bytes as-is (JPEG -> DCTDecode, exact);
                # rotate reproduces the source's EXIF orientation.
                page.insert_image(rect, stream=spec.data, rotate=spec.rotate)
            else:
                buf = io.BytesIO()
                spec.image.save(buf, format="PNG")  # lossless: original pixels kept
                page.insert_image(rect, stream=buf.getvalue())
            _overlay_text(page, page_result)
        doc.save(str(out_path), deflate=True)
    finally:
        doc.close()
