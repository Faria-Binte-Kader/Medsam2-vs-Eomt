import hashlib
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import download_data as D


def make_server(payload, honor_range=True, fail_after=None):
    """Local HTTP server. fail_after=N: first request sends N bytes then drops the connection."""
    state = {"calls": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["calls"] += 1
            start, code = 0, 200
            rng = self.headers.get("Range")
            if rng and honor_range:
                start, code = int(rng.split("=")[1].split("-")[0]), 206
            body = payload[start:]
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if fail_after is not None and state["calls"] == 1:
                self.wfile.write(body[:fail_after])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/f", state


def info(payload, url, md5=None):
    return {"size": len(payload), "md5": md5 or hashlib.md5(payload).hexdigest(), "url": url}


PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


def test_download_ok_and_idempotent(tmp_path):
    srv, url, state = make_server(PAYLOAD)
    p = D.download_file("a.bin", info(PAYLOAD, url), tmp_path)
    assert p.read_bytes() == PAYLOAD and not (tmp_path / "a.bin.part").exists()
    D.download_file("a.bin", info(PAYLOAD, url), tmp_path)
    assert state["calls"] == 1
    srv.shutdown()


def test_resume_from_partial_file(tmp_path):
    srv, url, _ = make_server(PAYLOAD)
    (tmp_path / "a.bin.part").write_bytes(PAYLOAD[:300_000])
    D.download_file("a.bin", info(PAYLOAD, url), tmp_path)
    assert (tmp_path / "a.bin").read_bytes() == PAYLOAD
    srv.shutdown()


def test_retry_after_dropped_connection(tmp_path):
    srv, url, state = make_server(PAYLOAD, fail_after=100_000)
    D.download_file("a.bin", info(PAYLOAD, url), tmp_path, retries=3, backoff=0)
    assert (tmp_path / "a.bin").read_bytes() == PAYLOAD and state["calls"] >= 2
    srv.shutdown()


def test_server_ignoring_range_restarts_cleanly(tmp_path):
    srv, url, _ = make_server(PAYLOAD, honor_range=False)
    (tmp_path / "a.bin.part").write_bytes(PAYLOAD[:1000])
    D.download_file("a.bin", info(PAYLOAD, url), tmp_path)
    assert (tmp_path / "a.bin").read_bytes() == PAYLOAD
    srv.shutdown()


def test_md5_mismatch_deletes_part_and_never_finalizes(tmp_path):
    srv, url, _ = make_server(PAYLOAD)
    with pytest.raises(D.DownloadError, match="MD5"):
        D.download_file("a.bin", info(PAYLOAD, url, md5="0" * 32), tmp_path)
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()
    srv.shutdown()


def test_unreachable_server_keeps_part_and_reports(tmp_path):
    (tmp_path / "a.bin.part").write_bytes(PAYLOAD[:10])
    with pytest.raises(D.DownloadError, match="re-run"):
        D.download_file("a.bin", info(PAYLOAD, "http://127.0.0.1:1/f"), tmp_path, retries=2, backoff=0)
    assert (tmp_path / "a.bin.part").exists()


def test_disk_check_refuses(tmp_path):
    with pytest.raises(D.DownloadError, match="not enough disk"):
        D.check_disk(tmp_path, 10**18)


def test_existing_corrupt_file_not_overwritten(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"junk")
    with pytest.raises(D.DownloadError, match="do not match"):
        D.download_file("a.bin", info(PAYLOAD, "http://x/f"), tmp_path)
    assert (tmp_path / "a.bin").read_bytes() == b"junk"


def test_safe_extract_selective_and_blocks_traversal(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("p1/a.txt", "a")
        zf.writestr("p2/b.txt", "b")
    out = tmp_path / "out"
    got = D.safe_extract(z, out, "p1/*")
    assert [g.name for g in got] == ["a.txt"] and not (out / "p2").exists()

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(D.DownloadError, match="unsafe"):
        D.safe_extract(evil, tmp_path / "out2")
    assert not (tmp_path / "escape.txt").exists()
    assert [m[0] for m in D.list_members(z)] == ["p1/a.txt", "p2/b.txt"]


def test_list_missing_archive_gives_clear_error(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(D.C, "RAW_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--list", "nope.zip"])
    with pytest.raises(D.DownloadError, match="not found"):
        D.main()
