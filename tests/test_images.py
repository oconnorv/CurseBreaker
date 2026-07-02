from PIL import Image, TiffImagePlugin, UnidentifiedImageError

import fitz

from cursbreaker.images import (
    count_content_pages,
    is_supported,
    load_output_images,
    load_pages,
    pdf_has_text_layer,
)


def _tiff_with_thumbnail(path):
    """A two-frame TIFF: a 'real' page plus a thumbnail flagged via the
    NewSubfileType tag (bit 0 = reduced-resolution version of another image)."""
    main = Image.new("RGB", (800, 600), "white")
    thumb = Image.new("RGB", (100, 75), "gray")
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[254] = 1
    thumb.encoderinfo = {"tiffinfo": ifd}
    main.save(path, format="TIFF", save_all=True, append_images=[thumb])
    return path


def test_supported_extensions():
    assert is_supported("a.tif") and is_supported("b.JPEG") and is_supported("c.pdf")
    assert not is_supported("d.docx")


def test_load_single_image_keeps_dimensions(png_path):
    pages = load_pages(png_path)
    assert len(pages) == 1
    page = pages[0]
    assert (page.orig_width, page.orig_height) == (800, 600)
    # No resize by default => the image sent to the API matches the original.
    assert (page.sent_width, page.sent_height) == (800, 600)
    assert page.output_stem == "sample"
    assert page.to_png_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_resize_scales_sent_image_only(png_path):
    pages = load_pages(png_path, max_dimension=400)
    page = pages[0]
    assert max(page.sent_width, page.sent_height) == 400
    assert (page.orig_width, page.orig_height) == (800, 600)


def test_load_normal_tiff(tmp_path):
    p = tmp_path / "scan.tif"
    Image.new("RGB", (640, 480), "white").save(p)
    pages = load_pages(p)
    assert len(pages) == 1
    assert (pages[0].orig_width, pages[0].orig_height) == (640, 480)


def test_tiff_falls_back_to_fitz_when_pillow_cannot_decode(tmp_path, monkeypatch):
    # Save a real TIFF that fitz can open; force the Pillow path to fail.
    p = tmp_path / "tricky.tif"
    Image.new("RGB", (400, 300), "white").save(p)

    real_open = Image.open

    def reject_tiff(path, *args, **kwargs):
        if str(path).lower().endswith((".tif", ".tiff")):
            raise UnidentifiedImageError("simulated Pillow decoder failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("cursbreaker.images.Image.open", reject_tiff)

    pages = load_pages(p, pdf_dpi=72)  # zoom=1.0 in the fitz fallback
    assert len(pages) == 1
    # Falling back through fitz should still give us a usable page image.
    assert pages[0].sent_width > 0 and pages[0].sent_height > 0


def test_count_content_pages_skips_tiff_thumbnail(tmp_path):
    p = _tiff_with_thumbnail(tmp_path / "scan_with_thumb.tif")
    # Pillow reports 2 frames; only one is real content.
    with Image.open(p) as im:
        assert getattr(im, "n_frames", 1) == 2
    assert count_content_pages(p) == 1


def test_load_pages_skips_tiff_thumbnail_frame(tmp_path):
    p = _tiff_with_thumbnail(tmp_path / "scan.tif")
    pages = load_pages(p)
    assert len(pages) == 1
    # The content frame is the larger one (800x600), not the 100x75 thumbnail.
    assert (pages[0].orig_width, pages[0].orig_height) == (800, 600)


def test_pdf_rasterizes_each_page(pdf_path):
    pages = load_pages(pdf_path, pdf_dpi=150)
    assert len(pages) == 2
    # 612pt * 150/72 ~= 1275 px wide.
    assert abs(pages[0].sent_width - 1275) <= 2
    assert pages[0].output_stem == "doc_page_0001"
    assert pages[1].output_stem == "doc_page_0002"


def test_iter_pages_renders_pdf_lazily(pdf_path, monkeypatch):
    # The whole point of the lazy loader: rendering one page does NOT rasterize
    # the rest of the document (so a 48-page PDF can be cancelled promptly and
    # only one page sits in memory at a time).
    from cursbreaker import images

    calls = {"n": 0}
    real = images.Image.frombytes

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(images.Image, "frombytes", counting)
    it = images.iter_pages(pdf_path, pdf_dpi=72)
    next(it)                       # render only the first page
    assert calls["n"] == 1         # page 2 of the 2-page PDF is NOT rendered yet
    next(it)
    assert calls["n"] == 2         # advancing renders the next page
    it.close()


# --- load_output_images: preserve the user's original bytes -------------- #

def test_output_image_jpeg_is_byte_for_byte_passthrough(tmp_path):
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (640, 480), (200, 100, 50)).save(p, format="JPEG", quality=90)
    original = p.read_bytes()
    specs = load_output_images(p)
    assert len(specs) == 1
    spec = specs[0]
    # The user's original JPEG bytes are carried through untouched (no re-encode,
    # no decode), so the embedded image is exactly the file they uploaded.
    assert spec.data == original
    assert spec.image is None
    assert (spec.width, spec.height) == (640, 480)
    assert spec.rotate == 0


def test_output_image_png_is_byte_for_byte_passthrough(tmp_path):
    p = tmp_path / "scan.png"
    Image.new("RGB", (300, 200), (1, 2, 3)).save(p, format="PNG")
    original = p.read_bytes()
    spec = load_output_images(p)[0]
    assert spec.data == original and spec.image is None
    assert (spec.width, spec.height) == (300, 200)


def test_output_image_jpeg_exif_rotation_reproduced_without_re_encoding(tmp_path):
    # A JPEG whose pixels are stored sideways with EXIF orientation 6 (rotate 90
    # CW for display). We must keep the original bytes AND record the rotation so
    # the page shows it upright -- matching the EXIF-corrected OCR coordinates.
    p = tmp_path / "rot.jpg"
    im = Image.new("RGB", (640, 480), (10, 20, 30))
    exif = im.getexif()
    exif[274] = 6
    im.save(p, format="JPEG", quality=90, exif=exif)
    original = p.read_bytes()
    spec = load_output_images(p)[0]
    assert spec.data == original          # bytes untouched
    assert spec.rotate == 270             # PyMuPDF rotate that uprights orientation 6
    # Upright (displayed) dimensions are the stored 640x480 swapped.
    assert (spec.width, spec.height) == (480, 640)


def test_output_image_multiframe_tiff_falls_back_to_decoded_frames(tmp_path):
    # TIFF can't be embedded byte-for-byte, so each content frame is decoded and
    # carried as a (lossless) PIL image -- still the full-resolution original,
    # with the embedded thumbnail skipped.
    p = _tiff_with_thumbnail(tmp_path / "multi.tif")
    specs = load_output_images(p)
    assert len(specs) == 1                # thumbnail frame skipped
    spec = specs[0]
    assert spec.data is None and spec.image is not None
    assert (spec.width, spec.height) == (800, 600)


# --- pdf_has_text_layer: detect an already-searchable PDF ---------------- #

def _pdf(path, texts):
    """A PDF whose page i shows ``texts[i]`` (empty string -> a bare image page,
    i.e. a scan with no text layer)."""
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
    pix.clear_with(220)
    for t in texts:
        page = doc.new_page(width=612, height=792)
        if t:
            page.insert_text((72, 100), t)
        else:
            page.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def test_pdf_has_text_layer_true_for_real_text(tmp_path):
    p = _pdf(tmp_path / "text.pdf", ["A page of real searchable text. " * 6] * 3)
    assert pdf_has_text_layer(p) is True


def test_pdf_has_text_layer_false_for_image_only_scan(tmp_path):
    p = _pdf(tmp_path / "scan.pdf", ["", "", ""])
    assert pdf_has_text_layer(p) is False


def test_pdf_has_text_layer_false_for_sparse_text_and_non_pdfs(tmp_path, png_path):
    # A running header / page number alone isn't a real text layer.
    sparse = _pdf(tmp_path / "sparse.pdf", ["p. 1", "p. 2", "p. 3"])
    assert pdf_has_text_layer(sparse) is False
    # Images never have a text layer.
    assert pdf_has_text_layer(png_path) is False


def test_pdf_has_text_layer_true_when_most_pages_have_text(tmp_path):
    # Majority-rule: a mostly-text document with a couple of blank/image pages
    # still counts as already searchable.
    p = _pdf(tmp_path / "mostly.pdf", ["Real searchable text here. " * 6] * 3 + ["", ""])
    assert pdf_has_text_layer(p) is True
