"""Load and normalize input images.

Supports TIFF, JPEG, PNG and GIF (raster, including multi-frame) plus PDF
(rasterized per page with PyMuPDF). Every page is returned as an RGB PIL image
alongside the bytes we will actually send to Gemini, so bounding boxes can be
mapped back to the correct dimensions.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

RASTER_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = RASTER_EXT | PDF_EXT


@dataclass
class LoadedPage:
    """A single page ready for transcription.

    ``image`` is the (possibly preprocessed/resized) RGB image that the
    bounding boxes will refer to. ``orig_width``/``orig_height`` are the
    dimensions of the user's source page; if we resized before sending to the
    API, the pipeline scales boxes back to these so the hOCR pairs with the
    original file.
    """

    image: Image.Image
    orig_width: int
    orig_height: int
    source_path: Path
    page_index: int  # 0 for single images; page number within a PDF/multiframe
    output_stem: str  # base name for the .txt/.hocr/.png outputs

    @property
    def sent_width(self) -> int:
        return self.image.width

    @property
    def sent_height(self) -> int:
        return self.image.height

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return buf.getvalue()


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT


def blank_png(size: tuple[int, int]) -> bytes:
    """A tiny solid-white PNG of the given pixel size. Gemini counts image tokens
    geometrically (tiling by dimensions, independent of content), so counting on
    this yields the same token count as the real page with a ~KB upload instead
    of multiple MB -- used to make the pre-flight estimate fast."""
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def _is_thumbnail_frame(im: Image.Image) -> bool:
    """Return True for a TIFF frame flagged as a reduced-resolution thumbnail
    or transparency mask via the ``NewSubfileType`` tag (254).

    Many scanners embed a thumbnail as a second internal frame, which Pillow's
    ``n_frames`` would otherwise count as a page.
    """
    try:
        nst = im.tag_v2.get(254, 0)
    except (AttributeError, KeyError, TypeError):
        return False
    # Bit 0 (= 1): reduced-resolution version of another image (thumbnail).
    # Bit 2 (= 4): transparency mask for another image in this file.
    return bool(nst & 0b101)


def count_content_pages(path: str | Path) -> int:
    """Return the number of *content* pages, skipping embedded thumbnails."""
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext in PDF_EXT:
            with fitz.open(path) as doc:
                return doc.page_count
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            if n <= 1:
                return n
            keep = 0
            for i in range(n):
                im.seek(i)
                if not _is_thumbnail_frame(im):
                    keep += 1
            # If every frame is somehow flagged, fall back to the raw count
            # rather than reporting zero pages.
            return keep or n
    except Exception:
        return 1


def pdf_has_text_layer(
    path: str | Path,
    *,
    max_pages_scanned: int = 50,
    min_chars: int = 100,
    min_ratio: float = 0.5,
) -> bool:
    """True if a PDF already carries a real, extractable text layer -- i.e. most
    of its (sampled) pages hold a substantial amount of selectable text, the
    hallmark of a document that was born digital or already OCR'd. Image inputs
    and non-PDFs never have one.

    Only page *text* is read (no rasterization), and at most ``max_pages_scanned``
    pages are inspected, so this stays fast even on a huge PDF. Biased against
    false positives -- a running header/footer alone won't clear ``min_chars`` --
    so we never offer to skip our overlay for a document that isn't really
    searchable."""
    path = Path(path)
    if path.suffix.lower() not in PDF_EXT:
        return False
    try:
        with fitz.open(path) as doc:
            scanned = min(doc.page_count, max_pages_scanned)
            if scanned <= 0:
                return False
            with_text = sum(
                1 for i in range(scanned)
                if len(doc[i].get_text().strip()) >= min_chars
            )
            return with_text / scanned >= min_ratio
    except Exception:
        return False


def _preprocess(img: Image.Image, *, enabled: bool, max_dimension: int) -> Image.Image:
    """Conservative preprocessing. Aggressive filters can hurt handwriting, so
    we only normalize orientation/color and apply gentle enhancement."""
    img = ImageOps.exif_transpose(img)  # honor camera/scanner rotation
    if img.mode != "RGB":
        img = img.convert("RGB")
    if enabled:
        img = ImageEnhance.Brightness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.05)
        img = img.filter(ImageFilter.MedianFilter(size=3))  # mild denoise
    if max_dimension and max(img.size) > max_dimension:
        scale = max_dimension / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _raster_frames(path: Path, *, dpi: int, max_frames: int = 0) -> list[Image.Image]:
    """Return every content frame of a raster file (1 for normal images, N for
    multi-frame TIFF/animated GIF). Embedded thumbnails / reduced-resolution
    preview frames (TIFF ``NewSubfileType`` flag) are skipped. ``max_frames``
    (>0) stops early once that many content frames are collected -- used by the
    cost estimate, which only needs the first page.

    Tries Pillow first. Falls back to PyMuPDF (already a dependency) for
    files Pillow can't decode -- most commonly TIFFs whose compression
    isn't supported by Pillow's bundled libtiff (old fax/JBIG/JPEG-in-
    TIFF variants and similar).
    """
    try:
        frames: list[Image.Image] = []
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            for i in range(n):
                im.seek(i)
                if n > 1 and _is_thumbnail_frame(im):
                    continue
                frames.append(im.copy())
                if max_frames and len(frames) >= max_frames:
                    break
        if not frames:
            # Every frame was flagged; keep the first so we never return
            # nothing for an otherwise readable file.
            with Image.open(path) as im:
                im.seek(0)
                frames = [im.copy()]
        return frames
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return _fitz_frames(path, zoom=dpi / 72.0, max_frames=max_frames)


def _iter_fitz_frames(path: Path, *, zoom: float, max_frames: int = 0) -> Iterator[Image.Image]:
    """Rasterize a PDF (or Pillow-undecodable file) one page at a time with
    PyMuPDF, yielding lazily so a caller can stop early -- only the current page
    is held in memory, and cancellation isn't blocked behind a whole-document
    render."""
    matrix = fitz.Matrix(zoom, zoom)
    yielded = 0
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yielded += 1
            if max_frames and yielded >= max_frames:
                return


def _fitz_frames(path: Path, *, zoom: float, max_frames: int = 0) -> list[Image.Image]:
    return list(_iter_fitz_frames(path, zoom=zoom, max_frames=max_frames))


def iter_pages(
    path: str | Path,
    *,
    preprocess: bool = True,
    max_dimension: int = 0,
    pdf_dpi: int = 300,
    max_pages: int = 0,
) -> Iterator[LoadedPage]:
    """Yield one ``LoadedPage`` at a time. PDFs are rasterized lazily (one page
    per ``next()``), so a large document doesn't block the caller in a single
    long render and only one page sits in memory at once. Raster files (usually
    1 frame; multi-frame TIFF/GIF are small) load eagerly to keep their
    decode-fallback robust. ``max_pages`` (>0) stops after that many pages."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type: {ext}")
    # Whether to suffix output stems with a page number -- from a cheap metadata
    # read, so it doesn't depend on having rendered everything first.
    multi = count_content_pages(path) > 1

    if ext in PDF_EXT:
        frames: Iterator[Image.Image] = _iter_fitz_frames(
            path, zoom=pdf_dpi / 72.0, max_frames=max_pages
        )
    else:
        frames = iter(_raster_frames(path, dpi=pdf_dpi, max_frames=max_pages))

    for i, frame in enumerate(frames):
        orig_w, orig_h = frame.size
        processed = _preprocess(
            frame, enabled=preprocess, max_dimension=max_dimension
        )
        stem = f"{path.stem}_page_{i + 1:04d}" if multi else path.stem
        yield LoadedPage(
            image=processed,
            orig_width=orig_w,
            orig_height=orig_h,
            source_path=path,
            page_index=i,
            output_stem=stem,
        )


def load_pages(
    path: str | Path,
    *,
    preprocess: bool = True,
    max_dimension: int = 0,
    pdf_dpi: int = 300,
    max_pages: int = 0,
) -> list[LoadedPage]:
    """Eager list of pages -- a back-compat wrapper around ``iter_pages``.
    ``max_pages`` (>0) loads only the first N content pages (the cost estimate
    renders just the first page of a large file)."""
    return list(
        iter_pages(
            path,
            preprocess=preprocess,
            max_dimension=max_dimension,
            pdf_dpi=pdf_dpi,
            max_pages=max_pages,
        )
    )


# EXIF orientation tag -> the ``rotate`` PyMuPDF's insert_image needs to display
# the *stored* image upright. Verified empirically: 6 and 8 are the opposite of
# the naive "rotate clockwise" reading. Flips (2/4/5/7) aren't pure rotations and
# fall through to a decoded, EXIF-corrected embed instead.
_EXIF_INSERT_ROTATE = {1: 0, 3: 180, 6: 270, 8: 90}


@dataclass
class OutputImage:
    """One image to embed in the output PDF, in upright (displayed) orientation
    that matches the OCR coordinate space.

    Either ``data`` -- the *original encoded bytes*, embedded as-is so a JPEG
    stays byte-for-byte the user's file (PDF stores it as DCTDecode; PNG stays
    lossless) -- displayed with ``rotate``; or ``image`` -- a decoded PIL image
    embedded losslessly (the fallback for multi-frame files, EXIF flips, or
    formats PDF can't carry natively, where pixels are still preserved exactly)."""

    width: int
    height: int
    data: bytes | None = None
    rotate: int = 0
    image: Image.Image | None = None


def load_output_images(path: str | Path, *, dpi: int = 300) -> list[OutputImage]:
    """Per content-frame images for the output searchable PDF, preserving the
    user's original as closely as the PDF format allows.

    A single-frame JPEG or PNG is passed through byte-for-byte (only its EXIF
    rotation is reproduced via the page rotation), so the embedded image is the
    user's untouched original. Anything else -- multi-frame TIFF/GIF, an EXIF
    flip, a thumbnail-bearing file, or a Pillow-undecodable file -- is decoded,
    oriented upright and embedded losslessly (pixels identical, never downscaled
    or enhanced). Frame N matches OCR page N."""
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                orientation = im.getexif().get(274, 1) or 1
                if getattr(im, "n_frames", 1) == 1 and orientation in _EXIF_INSERT_ROTATE:
                    rotate = _EXIF_INSERT_ROTATE[orientation]
                    w, h = im.size
                    if rotate in (90, 270):  # upright dimensions after rotation
                        w, h = h, w
                    return [OutputImage(width=w, height=h, data=data, rotate=rotate)]
        except Exception:
            pass  # fall through to the decode path
    out = []
    for frame in _raster_frames(path, dpi=dpi):
        img = ImageOps.exif_transpose(frame).convert("RGB")
        out.append(OutputImage(width=img.width, height=img.height, image=img))
    return out
