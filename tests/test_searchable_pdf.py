import io

import fitz
from PIL import Image

from cursbreaker.images import OutputImage
from cursbreaker.models import PageResult, PixelBox, TranscribedLine
from cursbreaker.searchable_pdf import (
    write_searchable_pdf_from_images,
    write_searchable_pdf_over_source,
)


def _img(size=(400, 200), color="white"):
    return Image.new("RGB", size, color)


def _decoded(img):
    """An OutputImage that embeds a decoded PIL image (the lossless fallback)."""
    return OutputImage(width=img.width, height=img.height, image=img)


def _page(width, height, lines, plain=""):
    return PageResult(
        image_name="p.png", width=width, height=height, lines=lines, plain_text=plain
    )


# --- image inputs: embed the full-resolution original + invisible text ---- #

def test_image_pdf_has_invisible_text_and_image(tmp_path):
    page = _page(400, 200, [
        TranscribedLine(text="hello world", box=PixelBox(x0=10, y0=20, x1=300, y1=60)),
        TranscribedLine(text="second line here", box=PixelBox(x0=10, y0=80, x1=380, y1=120)),
    ], "hello world\nsecond line here")
    out = tmp_path / "out.pdf"
    write_searchable_pdf_from_images([_decoded(_img((400, 200)))], [page], out)
    with fitz.open(out) as doc:
        assert doc.page_count == 1
        assert round(doc[0].rect.width) == 400 and round(doc[0].rect.height) == 200
        text = doc[0].get_text()
        for word in ("hello", "world", "second", "line", "here"):
            assert word in text, f"missing {word!r}"
        assert len(doc[0].get_images()) >= 1


def test_image_pdf_embeds_full_res_even_when_ocr_downscaled(tmp_path):
    # OCR ran on a 200x100 downscale; the output must embed the 800x400 original.
    page = _page(200, 100, [
        TranscribedLine(text="big page", box=PixelBox(x0=10, y0=10, x1=180, y1=40)),
    ], "big page")
    out = tmp_path / "full.pdf"
    write_searchable_pdf_from_images([_decoded(_img((800, 400)))], [page], out)
    with fitz.open(out) as doc:
        assert round(doc[0].rect.width) == 800 and round(doc[0].rect.height) == 400
        assert "big" in doc[0].get_text()


def test_unicode_round_trips(tmp_path):
    page = _page(700, 240, [
        TranscribedLine(text="Café naïveté résumé", box=PixelBox(x0=10, y0=20, x1=600, y1=60)),
        TranscribedLine(text="Привет мир", box=PixelBox(x0=10, y0=90, x1=500, y1=130)),
        TranscribedLine(text="Καλημέρα κόσμε", box=PixelBox(x0=10, y0=160, x1=600, y1=200)),
    ])
    out = tmp_path / "u.pdf"
    write_searchable_pdf_from_images([_decoded(_img((700, 240)))], [page], out)
    with fitz.open(out) as doc:
        text = doc[0].get_text()
    for word in ("Café", "naïveté", "résumé", "Привет", "мир", "Καλημέρα", "κόσμε"):
        assert word in text, f"missing {word!r}"


def test_image_length_mismatch_raises(tmp_path):
    page = _page(400, 200, [])
    try:
        write_searchable_pdf_from_images([_decoded(_img())], [page, page], tmp_path / "x.pdf")
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched lengths")


def test_passthrough_embeds_original_bytes_unaltered(tmp_path):
    # A passthrough OutputImage carries the user's original encoded bytes; the
    # PDF must store them byte-for-byte (JPEG -> DCTDecode), so the user's image
    # comes back completely unaltered except for the added searchable layer.
    buf = io.BytesIO()
    _img((500, 300), "white").save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()
    spec = OutputImage(width=500, height=300, data=jpeg_bytes)
    page = _page(500, 300, [
        TranscribedLine(text="overlay text", box=PixelBox(x0=10, y0=10, x1=300, y1=60)),
    ], "overlay text")
    out = tmp_path / "passthrough.pdf"
    write_searchable_pdf_from_images([spec], [page], out)
    with fitz.open(out) as doc:
        assert round(doc[0].rect.width) == 500 and round(doc[0].rect.height) == 300
        xref = doc[0].get_images()[0][0]
        info = doc.extract_image(xref)
        assert info["image"] == jpeg_bytes  # exact original JPEG, no re-encode
        assert "overlay" in doc[0].get_text()  # still searchable


# --- PDF inputs: overlay text on the original PDF (no re-rasterization) ---- #

def _source_pdf(path, n=2, width=612, height=792, rotation=0):
    doc = fitz.open()
    for i in range(n):
        p = doc.new_page(width=width, height=height)
        p.insert_text((72, 100), f"ORIGINAL {i + 1}")
        if rotation:
            p.set_rotation(rotation)
    doc.save(path)
    doc.close()


def test_pdf_overlay_preserves_original_pages(tmp_path):
    src = tmp_path / "src.pdf"
    _source_pdf(src, n=2)
    # OCR results came from a 300-dpi render (612*4.17 x 792*4.17 ~= 2550x3300),
    # but we overlay on the original, so the output keeps the 612x792 page size.
    pages = [
        _page(2550, 3300, [
            TranscribedLine(text=f"ocr page {i + 1}", box=PixelBox(x0=100, y0=100, x1=900, y1=220)),
        ], f"ocr page {i + 1}")
        for i in range(2)
    ]
    out = tmp_path / "over.pdf"
    write_searchable_pdf_over_source(src, pages, out)
    with fitz.open(out) as doc:
        assert doc.page_count == 2
        for i in range(2):
            # Original page size kept (not the pixel-sized page a re-raster makes).
            assert round(doc[i].rect.width) == 612 and round(doc[i].rect.height) == 792
            text = doc[i].get_text()
            assert f"ORIGINAL {i + 1}" in text   # original content survives
            assert "ocr" in text and "page" in text  # plus the searchable overlay


def test_pdf_overlay_keeps_rotation_and_stays_searchable(tmp_path):
    src = tmp_path / "rot.pdf"
    _source_pdf(src, n=1, rotation=90)
    # A 90-rotated 612x792 page displays landscape (792x612); the OCR render is
    # that displayed orientation at 300 dpi (~3300x2550).
    page = _page(3300, 2550, [
        TranscribedLine(text="sideways words", box=PixelBox(x0=100, y0=100, x1=1200, y1=260)),
    ], "sideways words")
    out = tmp_path / "rot_out.pdf"
    write_searchable_pdf_over_source(src, [page], out)
    with fitz.open(out) as doc:
        assert doc[0].rotation == 90              # original orientation preserved
        text = doc[0].get_text()
        assert "ORIGINAL" in text                 # original content survives
        assert "sideways" in text and "words" in text  # overlay is searchable


def test_pdf_overlay_more_pages_than_source_raises(tmp_path):
    src = tmp_path / "one.pdf"
    _source_pdf(src, n=1)
    pages = [_page(612, 792, []), _page(612, 792, [])]
    out = tmp_path / "x.pdf"
    try:
        write_searchable_pdf_over_source(src, pages, out)
    except ValueError:
        # Validated before any output file is created.
        assert not out.exists()
        return
    raise AssertionError("expected ValueError when OCR pages exceed source pages")


def test_pdf_overlay_appends_to_the_unaltered_original(tmp_path):
    # The text layer is appended incrementally: the output begins with the
    # source's exact bytes (original untouched) with only the searchable layer
    # added on the end. That append-don't-rebuild approach is also what keeps
    # memory flat on huge PDFs -- we never serialize the whole document at once.
    src = tmp_path / "src.pdf"
    _source_pdf(src, n=3)
    original = src.read_bytes()
    pages = [
        _page(612, 792, [
            TranscribedLine(text=f"layer {i + 1}", box=PixelBox(x0=72, y0=120, x1=400, y1=160)),
        ], f"layer {i + 1}")
        for i in range(3)
    ]
    out = tmp_path / "appended.pdf"
    write_searchable_pdf_over_source(src, pages, out)
    result = out.read_bytes()
    assert result.startswith(original)   # original bytes preserved verbatim
    assert len(result) > len(original)   # plus the appended text layer
    with fitz.open(out) as doc:
        assert doc.page_count == 3
        assert "ORIGINAL 1" in doc[0].get_text() and "layer" in doc[0].get_text()


def test_pdf_overlay_writes_one_content_stream_per_page(tmp_path):
    # Regression: the invisible text layer is written as a single content stream
    # per page (via TextWriter), not one PDF text object per word. The old
    # per-word insert_text appended a separate content stream for every word
    # (~hundreds on a dense page), which bloated text-heavy PDFs by ~50x. Here
    # each page carries 1,200 words but must stay at one original + one overlay
    # stream.
    src = tmp_path / "src.pdf"
    _source_pdf(src, n=2)  # each source page starts with a single content stream
    many = " ".join(f"word{i}" for i in range(60))
    pages = [
        _page(612, 792, [
            TranscribedLine(
                text=many,
                box=PixelBox(x0=40, y0=40 + r * 20, x1=560, y1=54 + r * 20),
            )
            for r in range(20)  # 20 lines x 60 words = 1,200 words/page
        ], many)
        for _ in range(2)
    ]
    out = tmp_path / "dense.pdf"
    write_searchable_pdf_over_source(src, pages, out)
    with fitz.open(out) as doc:
        for i in range(2):
            streams = len(doc[i].get_contents())
            assert streams <= 2, (
                f"page {i} has {streams} content streams; the overlay should add "
                "just one (per-word insert_text would add ~1,200)"
            )
            assert "word0" in doc[i].get_text() and "word59" in doc[i].get_text()


def test_pdf_overlay_falls_back_to_full_save(tmp_path, monkeypatch):
    # When a PDF can't be saved incrementally (PyMuPDF refuses, e.g. it repaired
    # the file on open), we fall back to a full save to a temp file and swap it
    # in -- still producing a correct, searchable PDF and cleaning up the temp.
    src = tmp_path / "src.pdf"
    _source_pdf(src, n=1)
    page = _page(612, 792, [
        TranscribedLine(text="fallback works", box=PixelBox(x0=72, y0=120, x1=400, y1=160)),
    ], "fallback works")
    out = tmp_path / "fb.pdf"

    real_save = fitz.Document.save

    def reject_incremental(self, *args, **kwargs):
        if kwargs.get("incremental"):
            raise RuntimeError("simulated: incremental save not possible")
        return real_save(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Document, "save", reject_incremental)
    write_searchable_pdf_over_source(src, [page], out)

    with fitz.open(out) as doc:
        text = doc[0].get_text()
        assert "fallback" in text and "ORIGINAL 1" in text
    assert not out.with_name(out.name + ".tmp").exists()  # temp swapped in & gone
