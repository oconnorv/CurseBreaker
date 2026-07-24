import io
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cursbreaker.server import app

client = TestClient(app)


def _wait_done(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def _wait_pages(file_ids, timeout=10.0):
    """Poll /api/staged-pages until the background worker has counted every id."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pages = client.get("/api/staged-pages").json()["pages"]
        if all(pages.get(i) is not None for i in file_ids):
            return pages
        time.sleep(0.05)
    raise AssertionError("page counts not computed in time")


@pytest.fixture
def run_with_mock(monkeypatch):
    """Run real jobs with the deterministic MockProvider -- no network or live
    key. Demo mode no longer exists, so tests opt into the mock explicitly by
    stubbing the provider factory and storing a dummy key so the 'no key' guard
    passes (the dummy is never validated, since make_provider is replaced)."""
    from cursbreaker import server
    from cursbreaker.gemini_client import MockProvider

    monkeypatch.setattr(server, "make_provider", lambda s: MockProvider())
    client.post("/api/settings", json={"api_key": "AIza_test_dummy_key_ABCDEF"})
    return MockProvider


@pytest.fixture
def run_with_slow_mock(monkeypatch):
    """Like run_with_mock but each Gemini call sleeps briefly, so a job stays
    'running' long enough to be cancelled deterministically."""
    import time as _time
    from cursbreaker import server
    from cursbreaker.gemini_client import MockProvider

    class _Slow(MockProvider):
        def transcribe_text(self, *a, **k):
            _time.sleep(0.2); return super().transcribe_text(*a, **k)

        def detect_lines(self, *a, **k):
            _time.sleep(0.2); return super().detect_lines(*a, **k)

        def transcribe_with_boxes(self, *a, **k):
            _time.sleep(0.2); return super().transcribe_with_boxes(*a, **k)

    monkeypatch.setattr(server, "make_provider", lambda s: _Slow())
    client.post("/api/settings", json={"api_key": "AIza_test_dummy_key_ABCDEF"})
    return _Slow


def test_settings_hides_key_and_roundtrips():
    r = client.get("/api/settings").json()
    assert r["api_key_set"] is False
    assert "api_key" not in r
    assert r["api_key_hint"] == ""
    assert r["api_key_source"] is None

    r2 = client.post("/api/settings", json={"mode": "one_pass"}).json()
    assert r2["mode"] == "one_pass"


def test_settings_exposes_key_hint_without_revealing_value():
    # Save a key
    r = client.post("/api/settings", json={"api_key": "AIzaSyABCDEFGHIJ_pretend_key_XYZ34"}).json()
    assert r["api_key_set"] is True
    assert "api_key" not in r          # raw key never leaves the server
    assert r["api_key_hint"].startswith("••••")
    assert r["api_key_hint"].endswith("XYZ34"[-4:])   # last 4 of the stored key
    assert r["api_key_source"] == "config"


def test_clear_api_key_endpoint():
    client.post("/api/settings", json={"api_key": "AIzaSy_keytoremove_ABCD"})
    assert client.get("/api/settings").json()["api_key_set"] is True
    r = client.delete("/api/settings/api_key").json()
    assert r["api_key_set"] is False
    assert r["api_key_hint"] == ""


def test_env_var_overrides_and_is_reported_as_source(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-set-key-ENDS")
    r = client.get("/api/settings").json()
    assert r["api_key_set"] is True
    assert r["api_key_source"] == "env"
    assert r["api_key_hint"].endswith("ENDS")


def test_full_flow(run_with_mock, png_path):
    client.post("/api/settings", json={"mode": "two_pass"})

    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    # Page counts are computed off the request: null in the response, then filled
    # in by the background worker (Transcribe doesn't wait on them).
    assert up["files"][0]["pages"] is None
    assert _wait_pages([file_id])[file_id] == 1

    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    status = _wait_done(started["job_id"])
    assert status["status"] == "done"

    result = status["results"][0]
    assert result["error"] is None
    assert result["n_lines"] == 4

    txt = client.get(result["txt"])
    assert txt.status_code == 200
    assert b"mock transcription" in txt.content

    hocr = client.get(result["hocr"])
    assert hocr.status_code == 200
    assert b"ocr_line" in hocr.content

    preview = client.get(result["images"][0]["preview"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"

    # Searchable PDF download
    assert result["pdf"], "searchable PDF URL missing from job result"
    pdf = client.get(result["pdf"])
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"

    # ALTO XML download
    assert result["alto"], "ALTO URL missing from job result"
    alto = client.get(result["alto"])
    assert alto.status_code == 200
    assert b"<alto" in alto.content and b"ns-v4" in alto.content

    zipped = client.get(f"/api/download/{started['job_id']}.zip")
    assert zipped.status_code == 200
    assert zipped.content[:2] == b"PK"
    names = zipfile.ZipFile(io.BytesIO(zipped.content)).namelist()
    assert "sample.alto.xml" in names and "sample.hocr" in names
    # Page images are internal scaffolding -- never swept into a download.
    assert not any(n.endswith(".png") for n in names)

    # Type-filtered zip: just the hOCR (the common "I only want hOCR" case).
    only_hocr = client.get(f"/api/download/{started['job_id']}.zip?types=hocr")
    assert only_hocr.status_code == 200
    h = zipfile.ZipFile(io.BytesIO(only_hocr.content)).namelist()
    assert h and all(n.endswith(".hocr") for n in h)

    # Multiple types combine into one archive; unrequested types stay out.
    two = client.get(f"/api/download/{started['job_id']}.zip?types=hocr,txt")
    t = set(zipfile.ZipFile(io.BytesIO(two.content)).namelist())
    assert any(n.endswith(".hocr") for n in t) and any(n.endswith(".txt") for n in t)
    assert not any(n.endswith((".pdf", ".alto.xml", ".png")) for n in t)

    # An unrecognized type is a 400 — never a silent full download.
    assert client.get(f"/api/download/{started['job_id']}.zip?types=bogus").status_code == 400


def test_upload_streams_multiple_files_in_one_batch(png_path, pdf_path):
    # The browser uploads in batches; one request can carry several files. Each
    # is streamed to disk (not read whole into memory) and staged immediately
    # with a deferred (null) page count.
    with open(png_path, "rb") as p, open(pdf_path, "rb") as d:
        up = client.post(
            "/api/upload",
            files=[
                ("files", ("a.png", p, "image/png")),
                ("files", ("b.pdf", d, "application/pdf")),
            ],
        ).json()
    by_name = {f["name"]: f for f in up["files"]}
    assert set(by_name) == {"a.png", "b.pdf"}
    assert by_name["a.png"]["pages"] is None and by_name["b.pdf"]["pages"] is None
    # The background worker then fills in the real counts.
    pages = _wait_pages([by_name["a.png"]["id"], by_name["b.pdf"]["id"]])
    assert pages[by_name["a.png"]["id"]] == 1
    assert pages[by_name["b.pdf"]["id"]] == 2  # the 2-page PDF fixture


def test_staged_pages_are_deferred_then_filled_in(png_path):
    # Staging never blocks on page counting: the upload returns null, and the
    # count appears on /api/staged-pages once the background worker computes it.
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    fid = up["files"][0]["id"]
    assert up["files"][0]["pages"] is None
    assert _wait_pages([fid])[fid] == 1


def _searchable_pdf_bytes():
    import fitz
    doc = fitz.open()
    for _ in range(2):
        doc.new_page(width=612, height=792).insert_text(
            (72, 100), "Lots of real, already-searchable text on this page. " * 6
        )
    data = doc.tobytes()
    doc.close()
    return data


def test_staged_pages_report_existing_text_layer(png_path):
    # The background scan also flags whether each staged file already carries a
    # searchable text layer, so the UI can offer the batch-wide skip.
    pdf = _searchable_pdf_bytes()
    up = client.post(
        "/api/upload",
        files=[
            ("files", ("textdoc.pdf", pdf, "application/pdf")),
            ("files", ("sample.png", png_path.read_bytes(), "image/png")),
        ],
    ).json()
    ids = {f["name"]: f["id"] for f in up["files"]}
    _wait_pages(list(ids.values()))
    layers = client.get("/api/staged-pages").json()["text_layers"]
    assert layers[ids["textdoc.pdf"]] is True    # already searchable
    assert layers[ids["sample.png"]] is False    # an image never is


def test_process_skip_existing_text_overlay_passes_pdf_through(run_with_mock):
    # End-to-end: a batch with skip enabled hands back the already-searchable PDF
    # unchanged (the produced "searchable PDF" is byte-for-byte the original).
    pdf = _searchable_pdf_bytes()
    up = client.post(
        "/api/upload", files={"files": ("textdoc.pdf", pdf, "application/pdf")}
    ).json()
    fid = up["files"][0]["id"]
    started = client.post(
        "/api/process",
        json={"file_ids": [fid], "outputs": ["pdf"], "skip_existing_text_overlay": True},
    ).json()
    status = _wait_done(started["job_id"])
    assert status["status"] == "done"
    result = status["results"][0]
    assert result["pdf"]
    assert client.get(result["pdf"]).content == pdf  # original passed through


def test_uploaded_bytes_survive_streaming_roundtrip(png_path):
    # Streaming the body to disk must reproduce the source exactly -- a corrupt
    # copy would surface as a wrong page count or a failed decode downstream.
    from cursbreaker import server

    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    staged_path = server.STAGED[up["files"][0]["id"]]
    assert staged_path.read_bytes() == png_path.read_bytes()


def test_upload_rejects_unsupported_types(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_bytes(b"nope")
    with open(bad, "rb") as fh:
        r = client.post(
            "/api/upload", files={"files": ("notes.docx", fh, "application/octet-stream")}
        )
    assert r.status_code == 400


def test_process_requires_key(png_path):
    # Isolated config has no stored key and no ambient env key -> 400.
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/process", json={"file_ids": [file_id]})
    assert r.status_code == 400


def test_download_unknown_job_is_404():
    r = client.get("/api/download/does-not-exist/whatever.txt")
    assert r.status_code == 404


def test_index_credits_mark_humphries_and_authorship():
    html = client.get("/").text
    assert "Mark Humphries" in html
    assert "Generative History" in html
    assert "generativehistory.substack.com" in html
    assert "John O'Connor" in html
    assert "Charlotte Mecklenburg Library" in html


def test_index_explains_how_to_get_an_api_key():
    # Non-technical users (GLAM staff) need an in-app pointer to creating a key,
    # not just an empty field. The collapsed help links to AI Studio, names the
    # create step, and reassures that there's a free tier.
    html = client.get("/").text
    assert "aistudio.google.com" in html
    assert "Create API key" in html
    assert "free" in html.lower()


def test_index_has_global_live_region_announcer():
    # A single always-present polite live region carries transient status
    # (key saved/cleared, estimate ready, job done) to screen readers reliably.
    html = client.get("/").text
    assert 'id="a11y-status" class="sr-only" role="status" aria-live="polite"' in html


def test_app_js_routes_status_through_announcer():
    # Guard the wiring so the live region isn't an unused empty element: the
    # announce() helper exists, targets #a11y-status, and a rejected key flags
    # the field for assistive tech.
    js = client.get("/static/app.js").text
    assert "function announce(" in js
    assert "a11y-status" in js
    assert 'setAttribute("aria-invalid"' in js


def test_status_badges_have_a_non_color_icon_cue():
    # Red-green colour blindness collapses the green "ok" and red "warn" badge
    # tints to nearly identical tones, so each state must also carry a shape/icon
    # cue rather than rely on hue alone (belt-and-braces over the differing text).
    css = client.get("/static/styles.css").text
    assert ".badge.ok::before" in css and ".badge.warn::before" in css
    assert r'content: "\2713"' in css   # check mark for the ok state
    assert 'content: "!"' in css        # a distinct glyph for the warn state


def test_staged_list_keeps_list_semantics():
    # list-style:none strips the implicit list role in Safari/VoiceOver, so the
    # staged-files list carries an explicit role="list" to stay announced.
    html = client.get("/").text
    assert '<ul id="staged" class="staged" role="list"' in html


def test_favicon_route_never_500s():
    # 200 when a favicon file is present; 204 when it isn't — never a 404/500.
    r = client.get("/favicon.ico")
    assert r.status_code in (200, 204)


def test_tesseract_status_endpoint_reports_availability():
    r = client.get("/api/tesseract")
    assert r.status_code == 200
    body = r.json()
    assert "available" in body and isinstance(body["available"], bool)
    assert "languages" in body and isinstance(body["languages"], list)
    # Richer diagnostics so the UI can explain *which* piece is missing.
    for key in ("wrapper_present", "binary_found", "cmd_path", "version",
                "error", "install_hint"):
        assert key in body
    assert isinstance(body["wrapper_present"], bool)
    assert isinstance(body["binary_found"], bool)


def test_content_type_round_trips_through_settings_api():
    r = client.post(
        "/api/settings", json={"content_type": "handwriting", "tesseract_language": "eng"}
    ).json()
    assert r["content_type"] == "handwriting"
    assert r["tesseract_language"] == "eng"


def test_refine_word_boxes_round_trips_through_settings_api():
    r = client.post("/api/settings", json={"refine_word_boxes": True}).json()
    assert r["refine_word_boxes"] is True


def test_legacy_mixed_content_type_migrates_to_handwriting_plus_refine():
    # "mixed" was retired; posting it (e.g. from an old client) must migrate to
    # the handwriting flow with word-box refinement on, not persist "mixed".
    r = client.post("/api/settings", json={"content_type": "mixed"}).json()
    assert r["content_type"] == "handwriting"
    assert r["refine_word_boxes"] is True


def test_index_has_content_type_selector_and_tesseract_status():
    html = client.get("/").text
    # The two content-type radios are present; "mixed" was retired in favor of
    # the refine-word-positions toggle.
    assert 'name="content_type"' in html
    for v in ("handwriting", "text"):
        assert f'value="{v}"' in html
    assert 'value="mixed"' not in html
    assert 'id="refine_word_boxes"' in html
    # A visible status block + a place to pick a Tesseract language.
    assert 'id="tesseract-info"' in html
    assert 'id="tesseract_language"' in html


def test_demo_mode_is_gone():
    # The user-facing demo/mock mode was removed entirely.
    html = client.get("/").text
    assert 'id="use_mock"' not in html
    assert "Demo mode" not in html
    # ...and it isn't a setting the API exposes or accepts.
    settings = client.get("/api/settings").json()
    assert "use_mock" not in settings
    client.post("/api/settings", json={"use_mock": True})
    assert "use_mock" not in client.get("/api/settings").json()


def test_heartbeat_endpoint_updates_timestamp():
    from cursbreaker import server
    server._LAST_PING_AT = None
    r = client.post("/api/heartbeat")
    assert r.status_code == 200
    assert server._LAST_PING_AT is not None


def test_heartbeat_bye_pulls_timestamp_back():
    from cursbreaker import server
    import time

    server._LAST_PING_AT = None
    before = time.time()
    client.post("/api/heartbeat?bye=true")
    # The bye signal moves the last-seen time into the past so the watchdog
    # fires soon, while still leaving a few seconds for a refresh.
    assert server._LAST_PING_AT is not None
    assert server._LAST_PING_AT < before + 0.5


def test_should_shutdown_predicate():
    from cursbreaker.server import _should_shutdown
    # No ping yet -> never shut down
    assert _should_shutdown(None, 10, now=100) is False
    # Recent ping -> stay up
    assert _should_shutdown(95, 10, now=100) is False
    # Stale ping -> shut down
    assert _should_shutdown(50, 10, now=100) is True
    # Job in flight pins the server alive even past the grace period
    assert _should_shutdown(50, 10, now=100, jobs_running=True) is False


def test_autoshutdown_waits_for_first_ping_before_arming(monkeypatch):
    """A failed browser-open means no tab ever connects; the watchdog must not
    quit the server on its own before then, or the user can never reach the
    printed URL. start_autoshutdown leaves the last-seen time unset until a real
    ping lands (None -> _should_shutdown stays False)."""
    from cursbreaker import server

    class _StubThread:  # don't spawn a real watchdog in the test
        def __init__(self, *a, **k): pass
        def start(self): pass

    monkeypatch.setattr(server, "_AUTOSHUTDOWN_STARTED", False)
    monkeypatch.setattr(server, "_LAST_PING_AT", 1234.0)  # stale value to be cleared
    monkeypatch.setattr(server.threading, "Thread", _StubThread)
    server.start_autoshutdown(grace_seconds=1, poll_seconds=1)
    assert server._LAST_PING_AT is None
    assert server._should_shutdown(server._LAST_PING_AT, 1, now=1_000_000) is False


def _make_access_record(status: int) -> "logging.LogRecord":
    import logging
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname="", lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:12345", "GET", "/x", "1.1", status),
        exc_info=None,
    )


def test_pretty_access_formatter_marks_status_classes():
    from cursbreaker.server import PrettyAccessFormatter
    fmt = PrettyAccessFormatter(use_colors=False)
    assert " ok " in fmt.format(_make_access_record(200))
    assert " ok " in fmt.format(_make_access_record(304))
    assert "warn" in fmt.format(_make_access_record(404))
    assert " err" in fmt.format(_make_access_record(500))
    # The leading "INFO" prefix from the default formatter is gone.
    assert "INFO" not in fmt.format(_make_access_record(200))


def test_pretty_access_formatter_emits_ansi_when_colors_on():
    from cursbreaker.server import PrettyAccessFormatter
    out = PrettyAccessFormatter(use_colors=True).format(_make_access_record(200))
    assert "\x1b[32m" in out and "\x1b[0m" in out  # green + reset

    out = PrettyAccessFormatter(use_colors=True).format(_make_access_record(500))
    assert "\x1b[31m" in out and "\x1b[0m" in out  # red + reset

    # use_colors=False should leave the line free of escape codes.
    out = PrettyAccessFormatter(use_colors=False).format(_make_access_record(200))
    assert "\x1b[" not in out


def test_pretty_access_formatter_falls_back_for_non_access_records():
    import logging
    from cursbreaker.server import PrettyAccessFormatter
    record = logging.LogRecord(
        name="uvicorn.error", level=logging.INFO, pathname="", lineno=0,
        msg="some lifecycle message", args=None, exc_info=None,
    )
    out = PrettyAccessFormatter(use_colors=False).format(record)
    assert "some lifecycle message" in out


def test_access_log_filter_drops_heartbeat_keeps_others():
    import logging

    from cursbreaker.server import install_access_log_filter

    install_access_log_filter()
    logger = logging.getLogger("uvicorn.access")
    our = next(
        f for f in logger.filters
        if getattr(f, "_cursbreaker_heartbeat", False)
    )

    fmt = '%s - "%s %s HTTP/%s" %d'
    hb = logger.makeRecord(
        "uvicorn.access", logging.INFO, "", 0, fmt,
        ("127.0.0.1:12345", "POST", "/api/heartbeat", "1.1", 200),
        None,
    )
    hb_bye = logger.makeRecord(
        "uvicorn.access", logging.INFO, "", 0, fmt,
        ("127.0.0.1:12345", "POST", "/api/heartbeat?bye=true", "1.1", 200),
        None,
    )
    upload = logger.makeRecord(
        "uvicorn.access", logging.INFO, "", 0, fmt,
        ("127.0.0.1:12345", "POST", "/api/upload", "1.1", 200),
        None,
    )

    assert our.filter(hb) is False
    assert our.filter(hb_bye) is False
    assert our.filter(upload) is True

    # Idempotent: calling twice doesn't stack duplicate filters.
    before = sum(
        1 for f in logger.filters
        if getattr(f, "_cursbreaker_heartbeat", False)
    )
    install_access_log_filter()
    after = sum(
        1 for f in logger.filters
        if getattr(f, "_cursbreaker_heartbeat", False)
    )
    assert before == after == 1


def test_key_status_no_key_by_default():
    # Fresh isolated config + cleared env -> nothing stored.
    assert client.get("/api/key-status").json()["state"] == "no_key"


def test_key_status_reports_invalid_revoked_key(monkeypatch):
    from cursbreaker import gemini_client

    monkeypatch.setenv("GEMINI_API_KEY", "revoked-key")

    def boom(key):
        e = Exception("400 API key not valid. Please pass a valid API key.")
        e.code = 400
        raise e

    monkeypatch.setattr(gemini_client, "_probe_models", boom)
    r = client.get("/api/key-status").json()
    assert r["state"] == "invalid"
    assert r["message"]


# --- token usage + cost estimate ----------------------------------------- #

def test_job_status_includes_token_fields(run_with_mock, png_path):
    client.post("/api/settings", json={"mode": "two_pass"})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    status = _wait_done(started["job_id"])

    assert "tokens" in status
    for k in ("input", "output", "thinking", "total", "calls", "cost"):
        assert k in status["tokens"]
    # MockProvider makes no real call, so nothing is billed.
    assert status["tokens"]["calls"] == 0
    assert status["results"][0]["tokens"]["total"] == 0
    # The internal provider handle must never be serialized to the client.
    assert "_provider" not in status


def test_append_log_caps_storage_but_counts_total():
    # Keeps only the most recent `cap` lines in memory, but tracks log_total so
    # the browser can append past the cap instead of freezing when the stored
    # window plateaus (the file-83-of-160 stall).
    from cursbreaker.server import _append_log
    job = {"log": [], "log_total": 0}
    for i in range(10):
        _append_log(job, f"line {i}", cap=4)
    assert job["log"] == ["line 6", "line 7", "line 8", "line 9"]  # newest 4 kept
    assert job["log_total"] == 10                                  # all 10 counted


def test_process_outputs_only_creates_selected_formats(run_with_mock, png_path):
    with open(png_path, "rb") as fh:
        up = client.post("/api/upload", files={"files": ("p.png", fh, "image/png")}).json()
    fid = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [fid], "outputs": ["hocr"]}).json()
    status = _wait_done(started["job_id"])

    assert status["status"] == "done"
    res = status["results"][0]
    assert res["hocr"]                                   # the one requested document
    assert res["txt"] is None and res["alto"] is None and res["pdf"] is None
    # Page images are still produced (they back Preview) but are preview-only:
    # present in the result, each with a preview URL and no download URL.
    assert res["images"] and all("download" not in im and im.get("preview") for im in res["images"])


def _run_default_job(png_path):
    with open(png_path, "rb") as fh:
        up = client.post("/api/upload", files={"files": ("p.png", fh, "image/png")}).json()
    started = client.post("/api/process", json={"file_ids": [up["files"][0]["id"]]}).json()
    job_id = started["job_id"]
    _wait_done(job_id)
    return job_id


def test_download_zip_reports_disk_full_clearly(run_with_mock, png_path, monkeypatch):
    """A full disk can't build the temp zip. Instead of failing silently (or a
    500 mid-write that reads as a crash), the endpoint returns 507 with a clear
    message -- and the probe reports it the same way, before any download starts."""
    import collections
    from cursbreaker import server

    job_id = _run_default_job(png_path)
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(server.shutil, "disk_usage", lambda p: Usage(10**9, 10**9, 0))
    r = client.get(f"/api/download/{job_id}.zip")
    assert r.status_code == 507
    assert "disk space" in r.json()["detail"].lower()
    assert client.get(f"/api/download/{job_id}.zip?probe=1").status_code == 507


def test_download_probe_ok_then_real_download_works(run_with_mock, png_path):
    job_id = _run_default_job(png_path)
    # Plenty of space in the test env -> probe is a cheap green light...
    probe = client.get(f"/api/download/{job_id}.zip?probe=1")
    assert probe.status_code == 200 and probe.json() == {"ok": True}
    # ...and the real download still streams a zip.
    z = client.get(f"/api/download/{job_id}.zip")
    assert z.status_code == 200 and z.content[:2] == b"PK"


def test_resume_and_end_release_a_paused_job():
    """/resume and /end record the action and wake the blocked worker; both are
    no-ops unless the job is actually paused, and 404 on an unknown job."""
    import threading
    from cursbreaker.server import JOBS

    def _paused():
        return {
            "status": "running", "paused": True, "pause_reason": "full",
            "log": [], "log_total": 0,
            "_resume": threading.Event(), "_resume_action": None,
        }

    JOBS["jr"] = _paused()
    assert client.post("/api/jobs/jr/resume").status_code == 200
    assert JOBS["jr"]["_resume_action"] == "resume" and JOBS["jr"]["_resume"].is_set()

    JOBS["je"] = _paused()
    assert client.post("/api/jobs/je/end").status_code == 200
    assert JOBS["je"]["_resume_action"] == "end" and JOBS["je"]["_resume"].is_set()

    # Not paused -> no-op: action stays None, worker isn't signalled.
    JOBS["jn"] = {**_paused(), "paused": False}
    client.post("/api/jobs/jn/resume")
    assert JOBS["jn"]["_resume_action"] is None and not JOBS["jn"]["_resume"].is_set()

    assert client.post("/api/jobs/nope/resume").status_code == 404
    assert client.post("/api/jobs/nope/end").status_code == 404

    for k in ("jr", "je", "jn"):
        JOBS.pop(k, None)


def test_job_status_exposes_activity_log_and_unit_counters(run_with_mock, pdf_path):
    client.post("/api/settings", json={"mode": "two_pass"})
    with open(pdf_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("doc.pdf", fh, "application/pdf")}
        ).json()
    file_id = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    status = _wait_done(started["job_id"])

    assert status["status"] == "done"
    # Page-driven bar units: the 2-page fixture -> 2 of 2.
    assert status["total_units"] == 2
    assert status["done_units"] == 2
    # Verbose activity log is present and captures the real steps.
    log = status["log"]
    assert isinstance(log, list) and log
    joined = "\n".join(log)
    assert "2 page(s) to transcribe" in joined
    assert "Page 1/2" in joined and "Page 2/2" in joined
    assert "Writing outputs" in joined
    assert isinstance(status["current"], str) and status["current"]
    assert "stage" in status
    # The private cancel flag is never serialized to the client.
    assert "_cancel" not in status


def test_cancel_running_job(run_with_slow_mock, pdf_path):
    client.post("/api/settings", json={"mode": "two_pass"})  # 2 calls/page -> slow
    with open(pdf_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("doc.pdf", fh, "application/pdf")}
        ).json()
    file_id = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    jid = started["job_id"]
    r = client.post(f"/api/jobs/{jid}/cancel").json()
    assert r["cancelling"] is True
    # The cancellation notice is in the activity log immediately -- before the
    # worker reaches a cancel boundary (it's still mid-call).
    interim = client.get(f"/api/jobs/{jid}").json()
    assert any("Cancellation requested" in line for line in interim["log"])
    status = _wait_done(jid)
    assert status["status"] == "cancelled"
    # ...and the final "Cancelled." line is there once it actually stops.
    assert any(line == "Cancelled." for line in status["log"])
    # The app is still alive and serving after a cancel.
    assert client.get("/").status_code == 200


def test_cancel_unknown_job_is_404():
    assert client.post("/api/jobs/does-not-exist/cancel").status_code == 404


def test_upload_out_of_space_is_clean_507_not_500(monkeypatch, png_path):
    # Disk-full mid-write used to bubble up as a 500 + traceback; it should be a
    # clean 507 with a helpful message, and the partial file must be rolled back.
    import errno

    from cursbreaker import server

    def boom(src, dst, length=0):  # simulate the disk filling during the copy
        raise OSError(errno.ENOSPC, "No space left on device")

    before = len(list(server.STAGE_DIR.iterdir()))
    monkeypatch.setattr(server.shutil, "copyfileobj", boom)
    with open(png_path, "rb") as fh:
        r = client.post("/api/upload", files={"files": ("sample.png", fh, "image/png")})
    assert r.status_code == 507
    assert "disk space" in r.json()["detail"].lower()
    assert len(list(server.STAGE_DIR.iterdir())) == before  # no half-written dir left


def test_sweep_removes_stale_workspaces_but_keeps_current_and_others(tmp_path, monkeypatch):
    from cursbreaker import server

    current = tmp_path / "cursbreaker_current"
    current.mkdir()
    stale = tmp_path / "cursbreaker_old"
    (stale / "stage").mkdir(parents=True)
    (stale / "stage" / "big.bin").write_bytes(b"x" * 1024)
    unrelated = tmp_path / "someone_elses_dir"
    unrelated.mkdir()

    monkeypatch.setattr(server, "_BASE", current)
    server.sweep_stale_workspaces()

    assert current.exists()        # the live session is kept
    assert not stale.exists()      # a stale cursbreaker_* workspace is reclaimed
    assert unrelated.exists()      # unrelated temp dirs are never touched


def test_cleanup_workspace_removes_the_session_dir(tmp_path, monkeypatch):
    from cursbreaker import server

    base = tmp_path / "cursbreaker_session"
    (base / "jobs").mkdir(parents=True)
    (base / "jobs" / "out.txt").write_text("data")
    monkeypatch.setattr(server, "_BASE", base)
    server._cleanup_workspace()
    assert not base.exists()


def test_stage_path_reads_files_in_place_without_copying(tmp_path, png_path):
    # Pointing at a folder stages the originals BY PATH -- no copy into the
    # server's workspace -- and skips unsupported files.
    from cursbreaker import server

    book = tmp_path / "book"
    book.mkdir()
    data = open(png_path, "rb").read()
    (book / "page1.png").write_bytes(data)
    (book / "page2.png").write_bytes(data)
    (book / "notes.txt").write_text("not an image")

    r = client.post("/api/stage-path", json={"path": str(book)})
    assert r.status_code == 200
    body = r.json()
    assert {f["name"] for f in body["files"]} == {"page1.png", "page2.png"}
    assert body["skipped"] == 1
    for f in body["files"]:
        staged = server.STAGED[f["id"]]
        assert staged.parent == book                   # the original location
        assert server.STAGE_DIR not in staged.parents  # not a copy in our temp


def test_stage_path_accepts_a_single_file(png_path):
    r = client.post("/api/stage-path", json={"path": str(png_path)})
    assert r.status_code == 200
    assert len(r.json()["files"]) == 1


def test_stage_path_strips_windows_copy_as_path_quotes(png_path):
    # Windows "Copy as path" wraps the path in double quotes.
    r = client.post("/api/stage-path", json={"path": f'"{png_path}"'})
    assert r.status_code == 200
    assert len(r.json()["files"]) == 1


def test_stage_path_missing_is_404(tmp_path):
    r = client.post("/api/stage-path", json={"path": str(tmp_path / "nope")})
    assert r.status_code == 404


def test_stage_path_empty_is_400():
    assert client.post("/api/stage-path", json={"path": "   "}).status_code == 400


def test_stage_path_folder_without_supported_files_is_400(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("x")
    assert client.post("/api/stage-path", json={"path": str(d)}).status_code == 400


def test_estimate_not_billable_for_printed_only(png_path):
    # Printed-only runs locally (Tesseract), so there's no Gemini token cost.
    client.post("/api/settings", json={"content_type": "text"})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/estimate", json={"file_ids": [file_id]}).json()
    assert r["billable"] is False
    assert "Printed-only" in r["reason"]
    assert r["total_low"] == 0 and r["total_high"] == 0
    assert r["cost_low"] is None and r["cost_high"] is None


def test_estimate_requires_key_when_billable(png_path):
    client.post("/api/settings", json={"content_type": "handwriting"})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/estimate", json={"file_ids": [file_id]})
    assert r.status_code == 400  # no key -> can't estimate Gemini cost


def test_estimate_billable_with_fake_provider(monkeypatch, png_path):
    from cursbreaker import server
    from cursbreaker.gemini_client import MockProvider

    class _P(MockProvider):
        def count_input_tokens(self, image_png, mime="image/png"):
            return 1000

    client.post(
        "/api/settings",
        json={
            "content_type": "handwriting",
            "mode": "one_pass",
            "transcription_model": "gemini-3.1-flash-lite",  # $0.25 in / $1.50 out
            "api_key": "AIza_estimate_test_key_WXYZ",
        },
    )
    monkeypatch.setattr(server, "make_provider", lambda s: _P())
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/estimate", json={"file_ids": [file_id]}).json()
    assert r["billable"] is True
    assert r["input"] == 1000          # 1 page * 1 call * 1000 input tokens
    # Cost is a range derived from the model's published price; one-pass output
    # range is 1800..5400 tokens/page.
    expected_low = 1000 / 1_000_000 * 0.25 + 1800 / 1_000_000 * 1.50
    expected_high = 1000 / 1_000_000 * 0.25 + 5400 / 1_000_000 * 1.50
    assert r["cost_low"] == pytest.approx(expected_low)
    assert r["cost_high"] == pytest.approx(expected_high)
    assert r["model"] == "gemini-3.1-flash-lite"
    assert r["price_input_per_mtok"] == 0.25


def test_estimate_no_staged_files_is_400():
    r = client.post("/api/estimate", json={"file_ids": ["nope"]})
    assert r.status_code == 400


def test_models_endpoint_returns_priced_catalog():
    body = client.get("/api/models").json()
    ids = [m["id"] for m in body["models"]]
    # Pro is first (the dropdown's default position + the saved default model).
    assert ids[0] == "gemini-3.1-pro-preview"
    assert "gemini-3.5-flash" in ids
    assert "gemini-3.1-flash-lite" in ids
    assert body["prices_as_of"]            # shown in the UI for transparency
    flash = next(m for m in body["models"] if m["id"] == "gemini-3.5-flash")
    assert flash["input_per_mtok"] == 1.50 and flash["output_per_mtok"] == 9.00
    pro = next(m for m in body["models"] if m["id"] == "gemini-3.1-pro-preview")
    assert pro["tier_threshold"] == 200_000   # tiered pricing is exposed


def test_model_choice_round_trips_through_settings_api():
    r = client.post(
        "/api/settings", json={"transcription_model": "gemini-3.5-flash"}
    ).json()
    assert r["transcription_model"] == "gemini-3.5-flash"


def test_detection_model_follows_transcription_model():
    # The single picker is enforced server-side: posting only the transcription
    # model keeps detection (two-pass) on the same model, so the priced/reported
    # model can't drift from the one detection actually uses.
    r = client.post(
        "/api/settings", json={"transcription_model": "gemini-3.1-flash-lite"}
    ).json()
    assert r["transcription_model"] == "gemini-3.1-flash-lite"
    assert r["detection_model"] == "gemini-3.1-flash-lite"


def test_index_has_cost_controls():
    html = client.get("/").text
    assert 'id="estimate"' in html
    assert 'id="model"' in html             # curated dropdown replaces free text
    assert 'id="token-text"' in html
    # The manual price inputs are gone; pricing is automatic now.
    assert 'id="price_input_per_mtok"' not in html
    assert 'id="price_output_per_mtok"' not in html


def test_clear_batch_removes_staged_uploads_and_job_outputs(run_with_mock, png_path):
    from cursbreaker import server

    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    fid = up["files"][0]["id"]
    upload_dir = Path(server.STAGED[fid]).parent  # STAGE_DIR/<id>/
    assert upload_dir.exists()

    job_id = client.post("/api/process", json={"file_ids": [fid]}).json()["job_id"]
    _wait_done(job_id)
    out_dir = Path(server.JOBS[job_id]["out_dir"])
    assert out_dir.exists() and any(out_dir.iterdir())  # outputs were written

    r = client.post("/api/clear").json()
    assert r["files_cleared"] >= 1 and r["jobs_cleared"] >= 1
    assert r["freed_bytes"] > 0
    # Both the in-memory state and the on-disk files are gone.
    assert server.STAGED == {} and server.JOBS == {}
    assert not upload_dir.exists()
    assert not out_dir.exists()


def test_clear_batch_keeps_path_staged_originals(png_path):
    from cursbreaker import server

    # Staging by path reads the user's original in place; clearing must only
    # unstage it, never delete it.
    client.post("/api/stage-path", json={"path": str(png_path)})
    assert str(png_path) in {str(p) for p in server.STAGED.values()}

    client.post("/api/clear")
    assert str(png_path) not in {str(p) for p in server.STAGED.values()}
    assert png_path.exists()  # the original is untouched on disk


def test_clear_batch_refused_while_job_running(run_with_slow_mock, png_path):
    from cursbreaker import server

    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("slow.png", fh, "image/png")}
        ).json()
    fid = up["files"][0]["id"]
    job_id = client.post("/api/process", json={"file_ids": [fid]}).json()["job_id"]

    # The slow mock keeps the job 'running' long enough to hit the guard.
    resp = client.post("/api/clear")
    assert resp.status_code == 409
    assert job_id in server.JOBS  # nothing was cleared

    # Don't leave a running job behind for later tests.
    client.post(f"/api/jobs/{job_id}/cancel")
    _wait_done(job_id)
