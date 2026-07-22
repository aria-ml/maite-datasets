import hashlib
import os
from pathlib import Path

import pytest
import requests
from huggingface_hub.utils.tqdm import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from requests import RequestException, Response

from maite_datasets._fileio import (
    HF_ATTEMPTS,
    HF_DOWNLOAD_FLAGS,
    HF_MAX_WORKERS_ANONYMOUS,
    HF_MAX_WORKERS_AUTHENTICATED,
    HFResource,
    ResourcePart,
    URLResource,
    _download_dataset,
    _download_part,
    _ensure_exists,
    _extract_archive,
    _extract_tar_archive,
    _extract_zip_archive,
    _hf_download_settings,
    _hf_extract,
    _is_kaggle,
    _part_filename,
    _remove_folder_nest,
    _session_setup,
    _validate_file,
    hf_constants,
)

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class MockHTTPError(Response):
    def __init__(self):
        super().__init__()
        self.reason = "MockError"
        self.status_code = 404


class MockRequestException(Response):
    def __init__(self):
        self.reason = "MockError"
        self.status_code = 404

    def raise_for_status(self):
        raise RequestException


class MockKaggleResponse:
    def __init__(self, status_code=200):
        self.headers = {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@pytest.mark.optional
class TestDownloadPart:
    """Mirror fallback: a part is fetched from the first of its mirrors that works."""

    def _part(self, *filenames):
        return ResourcePart(
            "example", tuple(URLResource(f"https://example.invalid/{f}", f, False, "sum") for f in filenames)
        )

    def test_first_working_mirror_wins_and_later_ones_are_untried(self, monkeypatch, tmp_path):
        tried = []

        def fake_ensure(resource, directory, root, download, verbose):
            tried.append(resource.filename)

        monkeypatch.setattr("maite_datasets._fileio._ensure_exists", fake_ensure)
        _download_part(self._part("a.zip", "b.zip"), tmp_path, tmp_path)
        assert tried == ["a.zip"]

    def test_falls_through_to_next_mirror(self, capsys, monkeypatch, tmp_path):
        tried = []

        def fake_ensure(resource, directory, root, download, verbose):
            tried.append(resource.filename)
            if resource.filename == "a.zip":
                raise Exception("checksum mismatch")

        monkeypatch.setattr("maite_datasets._fileio._ensure_exists", fake_ensure)
        _download_part(self._part("a.zip", "b.zip"), tmp_path, tmp_path, verbose=True)
        assert tried == ["a.zip", "b.zip"]
        assert "Could not retrieve example, trying the next mirror." in capsys.readouterr().out

    def test_last_mirror_failure_propagates(self, monkeypatch, tmp_path):
        def fake_ensure(resource, directory, root, download, verbose):
            raise RuntimeError(f"{resource.filename} is gone")

        monkeypatch.setattr("maite_datasets._fileio._ensure_exists", fake_ensure)
        with pytest.raises(RuntimeError, match="b.zip is gone"):
            _download_part(self._part("a.zip", "b.zip"), tmp_path, tmp_path)

    def test_no_fallback_when_downloads_disabled(self, monkeypatch, tmp_path):
        """Without downloads every mirror fails identically, so retrying only hides the cause."""
        tried = []

        def fake_ensure(resource, directory, root, download, verbose):
            tried.append(resource.filename)
            raise FileNotFoundError("download parameter is set to False")

        monkeypatch.setattr("maite_datasets._fileio._ensure_exists", fake_ensure)
        with pytest.raises(FileNotFoundError):
            _download_part(self._part("a.zip", "b.zip"), tmp_path, tmp_path, download=False)
        assert tried == ["a.zip"]


@pytest.mark.optional
class TestHFDownloadSettings:
    """Hub preferences apply only for the duration of a download, then get put back."""

    @pytest.mark.parametrize("flag", HF_DOWNLOAD_FLAGS)
    def test_sets_the_constant_the_hub_actually_reads(self, flag):
        # The hub reads the env var into a constant as it imports, so at runtime the
        # constant -- not os.environ -- is the switch that has any effect.
        assert getattr(hf_constants, flag) is False
        with _hf_download_settings():
            assert getattr(hf_constants, flag) is True
            assert os.environ[flag] == "1"
        assert getattr(hf_constants, flag) is False

    @pytest.mark.parametrize("flag", HF_DOWNLOAD_FLAGS)
    def test_restores_an_absent_env_var_by_removing_it(self, monkeypatch, flag):
        monkeypatch.delenv(flag, raising=False)
        with _hf_download_settings():
            pass
        assert flag not in os.environ

    @pytest.mark.parametrize("flag", HF_DOWNLOAD_FLAGS)
    @pytest.mark.parametrize("preset", ["0", "1"])
    def test_restores_a_callers_own_setting(self, monkeypatch, flag, preset):
        monkeypatch.setenv(flag, preset)
        monkeypatch.setattr(hf_constants, flag, preset == "1")
        with _hf_download_settings():
            assert getattr(hf_constants, flag) is True
        assert os.environ[flag] == preset
        assert getattr(hf_constants, flag) is (preset == "1")

    def test_silences_progress_bars_and_restores_them(self):
        """Progress bars need the public toggle: the hub binds that constant by value."""
        assert are_progress_bars_disabled() is False
        with _hf_download_settings():
            assert are_progress_bars_disabled() is True
        assert are_progress_bars_disabled() is False

    def test_leaves_progress_bars_off_if_the_caller_had_them_off(self):
        disable_progress_bars()
        try:
            with _hf_download_settings():
                pass
            assert are_progress_bars_disabled() is True
        finally:
            enable_progress_bars()

    def test_restores_when_the_download_raises(self, monkeypatch):
        for flag in HF_DOWNLOAD_FLAGS:
            monkeypatch.delenv(flag, raising=False)
        with pytest.raises(ConnectionError), _hf_download_settings():
            raise ConnectionError("429")
        for flag in HF_DOWNLOAD_FLAGS:
            assert getattr(hf_constants, flag) is False
            assert flag not in os.environ
        assert are_progress_bars_disabled() is False

    def test_is_a_no_op_without_huggingface_hub(self, monkeypatch):
        monkeypatch.setattr("maite_datasets._fileio.hf_constants", None)
        with _hf_download_settings():
            pass


@pytest.mark.optional
class TestHFResource:
    def test_ensure_exists_dispatches_to_the_hub(self, monkeypatch, tmp_path):
        called = {}
        monkeypatch.setattr(
            "maite_datasets._fileio._hf_extract",
            lambda repo_id, repo_type, local_dir, allow_patterns, verbose: called.update(
                repo_id=repo_id, repo_type=repo_type, patterns=allow_patterns
            ),
        )
        _ensure_exists(HFResource("owner/name", "dataset", ["train/*"]), tmp_path, tmp_path, True, False)
        assert called == {"repo_id": "owner/name", "repo_type": "dataset", "patterns": ["train/*"]}

    def test_download_false_raises_without_touching_the_hub(self, monkeypatch, tmp_path):
        def fail(*args, **kwargs):
            raise AssertionError("should not reach the hub")

        monkeypatch.setattr("maite_datasets._fileio._hf_extract", fail)
        with pytest.raises(FileNotFoundError, match="owner/name has not been downloaded"):
            _ensure_exists(HFResource("owner/name"), tmp_path, tmp_path, False, False)

    def test_part_filename_rejects_hf_mirrors(self):
        """HF unpacks a file tree, so there is no archive filename to hand back."""
        with pytest.raises(TypeError, match="fetched from huggingface"):
            _part_filename(ResourcePart("example", (HFResource("owner/name"),)))


@pytest.mark.optional
class TestHelperFunctionsBaseDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_no_zip(self, capsys, dataset_no_zip, verbose):
        resource = URLResource("fakeurl", "stuff.txt", True, TEMP_MD5)
        _ensure_exists(resource, dataset_no_zip.parent, dataset_no_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert captured.out == "stuff.txt already exists, skipping download.\n"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_single_zip(self, capsys, dataset_single_zip, verbose):
        checksum = get_tmp_hash(dataset_single_zip)
        resource = URLResource("fakeurl", "testing.zip", True, checksum)
        _ensure_exists(
            resource,
            dataset_single_zip.parent,
            dataset_single_zip.parent,
            True,
            verbose,
        )
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting testing.zip..." in captured.out

    def test_ensure_exists_file_exists_bad_checksum(self, dataset_no_zip):
        resource = URLResource("fakeurl", "stuff.txt", True, TEMP_SHA256)
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_ensure_exists_download_non_zip(self, capsys, mnist_folder, monkeypatch, tmp_path):
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"fake-mnist-content")
        checksum = hashlib.sha256(payload.read_bytes()).hexdigest()

        def fake_download(url, file_path, *args, **kwargs):
            file_path.write_bytes(payload.read_bytes())

        monkeypatch.setattr("maite_datasets._fileio._download_dataset", fake_download)

        url = "https://example.invalid/mnist.npz"
        resource = URLResource(url, "mnist.npz", False, checksum)
        _ensure_exists(resource, mnist_folder, mnist_folder.parent, True, True)
        captured = capsys.readouterr()
        assert f"Downloading mnist.npz from {url}" in captured.out

    def test_ensure_exists_download_bad_checksum(self, mnist_folder, monkeypatch):
        def fake_download(url, file_path, *args, **kwargs):
            file_path.write_bytes(b"anything")

        monkeypatch.setattr("maite_datasets._fileio._download_dataset", fake_download)

        resource = URLResource("https://example.invalid/mnist.npz", "mnist.npz", False, "abc")
        err_msg = "File checksum mismatch on downloaded mnist.npz. Retry the download."
        with pytest.raises(Exception) as e:
            _ensure_exists(resource, mnist_folder, mnist_folder.parent, True, False)
        assert err_msg in str(e.value)
        # The corrupt download is discarded, so a retry downloads again rather than
        # tripping the "already exists" branch and failing identically forever.
        assert not (mnist_folder / "mnist.npz").exists()

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_download_zip(self, capsys, mnist_folder, verbose, monkeypatch, tmp_path):
        from zipfile import ZipFile

        # Build a real zip fixture so extraction has something valid to operate on.
        inner = tmp_path / "payload.txt"
        inner.write_text("hello")
        zip_fixture = tmp_path / "fixture.zip"
        with ZipFile(zip_fixture, "w") as zf:
            zf.write(inner, arcname="payload.txt")
        checksum = get_tmp_hash(zip_fixture)

        def fake_download(url, file_path, *args, **kwargs):
            file_path.write_bytes(zip_fixture.read_bytes())

        monkeypatch.setattr("maite_datasets._fileio._download_dataset", fake_download)

        resource = URLResource("https://example.invalid/fake.zip", "2021.zip", True, checksum)
        _ensure_exists(resource, mnist_folder, mnist_folder.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert f"Extracting {resource.filename}..." in captured.out

    def test_ensure_exists_error(self, dataset_no_zip):
        resource = URLResource("fakeurl", "something.zip", True, "")
        err_msg = "Data could not be loaded with the provided root directory,"
        with pytest.raises(FileNotFoundError) as e:
            _ensure_exists(resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_download_dataset_http_error(self, monkeypatch):
        def mock_get(*args, **kwargs):
            return MockHTTPError()

        monkeypatch.setattr(requests.Session, "get", mock_get)
        with pytest.raises(RuntimeError):
            _download_dataset(url="http://mock/", file_path=Path("fake/path"))

    def test_download_dataset_request_error(self, monkeypatch):
        def mock_get(*args, **kwargs):
            return MockRequestException()

        monkeypatch.setattr(requests.Session, "get", mock_get)
        with pytest.raises(ValueError):
            _download_dataset(url="http://mock/", file_path=Path("fake/path"))

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://www.kaggle.com/api/v1/datasets/download/owner/name?datasetVersionNumber=1", True),
            ("https://kaggle.com/api/v1/datasets/download/owner/name", True),
            ("https://huggingface.co/datasets/owner/name/resolve/main/train.zip", False),
            # Host must *end* in kaggle.com -- a look-alike domain must not opt in.
            ("https://www.kaggle.com.example.invalid/archive.zip", False),
            ("not-a-url", False),
        ],
    )
    def test_is_kaggle(self, url, expected):
        assert _is_kaggle(url) is expected

    def test_non_kaggle_sets_referer_and_makes_no_request(self, monkeypatch):
        called = False

        def mock_get(*args, **kwargs):
            nonlocal called
            called = True
            return MockKaggleResponse()

        monkeypatch.setattr(requests.Session, "get", mock_get)
        session = _session_setup("https://example.invalid/archive.zip", timeout=30)
        assert called is False
        assert session.headers["Referer"] == "https://google.com/"

    def test_kaggle_warms_cookies_from_landing_page(self, monkeypatch):
        requested = []

        def mock_get(self, url, **kwargs):
            requested.append(url)
            # Stand in for the Set-Cookie handling requests does on a real response.
            self.cookies.set("ka_sessionid", "abc123")
            return MockKaggleResponse()

        monkeypatch.setattr(requests.Session, "get", mock_get)
        session = _session_setup("https://www.kaggle.com/api/v1/datasets/download/owner/name", timeout=30)
        assert requested == ["https://www.kaggle.com"]
        assert session.cookies.get("ka_sessionid") == "abc123"
        # No Referer on the Kaggle path; the warm-up cookies are what authorize the download.
        assert "Referer" not in session.headers

    def test_kaggle_http_error_on_warmup_propagates(self, monkeypatch):
        def mock_get(self, url, **kwargs):
            return MockKaggleResponse(status_code=403)

        monkeypatch.setattr(requests.Session, "get", mock_get)
        with pytest.raises(requests.exceptions.HTTPError):
            _session_setup("https://www.kaggle.com/api/v1/datasets/download/owner/name", timeout=30)

    @pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
    def test_validate_file(self, dataset_no_zip, use_md5, hash_value):
        assert _validate_file(dataset_no_zip, hash_value, use_md5)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_zip_extraction_nested_zip(self, capsys, dataset_nested_zip, verbose):
        _extract_archive(
            dataset_nested_zip.suffix,
            dataset_nested_zip,
            dataset_nested_zip.parent,
            False,
            verbose,
        )
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting nested zip" in captured.out

    def test_extract_archive_bad_zip(self, dataset_no_zip):
        err_msg = f"{dataset_no_zip.name} is not a valid zip file, skipping extraction."
        with pytest.raises(FileNotFoundError) as e:
            _extract_zip_archive(dataset_no_zip, dataset_no_zip.parent)
        assert err_msg in str(e.value)

    def test_tarfile_error(self, dataset_single_zip):
        err_msg = f"{dataset_single_zip.name} is not a valid tar file"
        with pytest.raises(FileNotFoundError) as e:
            _extract_tar_archive(file_path=dataset_single_zip, extract_to=dataset_single_zip.parent)
        assert err_msg in str(e.value)


class MockResponse:
    """Minimal stand-in for the streamed responses `_download_dataset` consumes."""

    def __init__(self, content=b"", content_type="application/octet-stream", text=""):
        self._content = content
        self.text = text
        self.headers = {"content-type": content_type, "content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def iter_content(self, block_size):
        for start in range(0, len(self._content), block_size):
            yield self._content[start : start + block_size]


GDRIVE_WARNING_PAGE = (
    '<form id="download-form" action="https://drive.usercontent.google.com/download">'
    '<input name="id" value="abc123"><input name="confirm" value="t"></form>'
)


@pytest.mark.optional
class TestDownloadDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_download_writes_content(self, monkeypatch, tmp_path, verbose):
        monkeypatch.setattr(requests.Session, "get", lambda *args, **kwargs: MockResponse(b"payload-bytes"))
        file_path = tmp_path / "out.bin"
        _download_dataset(url="http://mock/", file_path=file_path, verbose=verbose)
        assert file_path.read_bytes() == b"payload-bytes"

    def test_download_follows_google_drive_confirmation(self, monkeypatch, tmp_path):
        """The first response is Google's virus-scan warning page; the form is resubmitted."""
        responses = [
            MockResponse(content_type="text/html", text=GDRIVE_WARNING_PAGE),
            MockResponse(b"real-bytes"),
        ]
        requested = []

        def mock_get(self, url, **kwargs):
            requested.append((url, kwargs.get("params")))
            return responses.pop(0)

        monkeypatch.setattr(requests.Session, "get", mock_get)
        file_path = tmp_path / "out.bin"
        _download_dataset(url="http://mock/", file_path=file_path)
        assert file_path.read_bytes() == b"real-bytes"
        assert requested[1] == (
            "https://drive.usercontent.google.com/download",
            {"id": "abc123", "confirm": "t"},
        )

    def test_download_html_without_form_raises(self, monkeypatch, tmp_path):
        """An HTML page that isn't Drive's confirm form is an error/login page, not a file."""
        monkeypatch.setattr(
            requests.Session,
            "get",
            lambda *args, **kwargs: MockResponse(b"<html>nope</html>", content_type="text/html", text="<html>nope"),
        )
        file_path = tmp_path / "out.bin"
        with pytest.raises(RuntimeError, match="expected a file, received a web page"):
            _download_dataset(url="http://mock/", file_path=file_path)
        assert not file_path.exists()


@pytest.mark.optional
class TestRemoveFolderNest:
    def test_moves_contents_up_and_removes_folder(self, capsys, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "a.txt").write_text("a")
        (nested / "sub").mkdir()

        _remove_folder_nest(nested, verbose=True)

        assert not nested.exists()
        assert (tmp_path / "a.txt").read_text() == "a"
        assert (tmp_path / "sub").is_dir()
        assert "All contents moved up one level successfully!" in capsys.readouterr().out

    def test_keeps_conflicting_files(self, capsys, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "a.txt").write_text("new")
        (tmp_path / "a.txt").write_text("existing")

        _remove_folder_nest(nested, verbose=True)

        assert (tmp_path / "a.txt").read_text() == "existing"
        assert (nested / "a.txt").exists()
        assert "The following files were not moved:" in capsys.readouterr().out

    def test_overwrite_replaces_existing(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "a.txt").write_text("new")
        (tmp_path / "a.txt").write_text("existing")

        _remove_folder_nest(nested, overwrite=True)

        assert (tmp_path / "a.txt").read_text() == "new"
        assert not nested.exists()


@pytest.mark.optional
class TestHFExtract:
    @pytest.fixture
    def hf_mock(self, monkeypatch):
        """Patch in a fake hub, returning the record of snapshot_download calls."""
        calls = []

        def fake_snapshot(repo_id, repo_type, local_dir, allow_patterns, max_workers):
            calls.append(
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "local_dir": local_dir,
                    "patterns": allow_patterns,
                    "workers": max_workers,
                }
            )

        def make_api(filelist):
            class FakeApi:
                def list_repo_files(self, repo_id, repo_type):
                    return filelist

            monkeypatch.setattr("maite_datasets._fileio.HfApi", FakeApi)
            monkeypatch.setattr("maite_datasets._fileio.snapshot_download", fake_snapshot)
            monkeypatch.setattr("maite_datasets._fileio.get_token", lambda: None)
            return calls

        return make_api

    @pytest.mark.parametrize(
        "allow_patterns, expected_count",
        [
            (None, 3),
            ("train/*", 2),
            (["val/*", "train/a.png"], 2),
        ],
    )
    def test_hf_extract_reports_the_count_the_hub_will_fetch(
        self, capsys, hf_mock, tmp_path, allow_patterns, expected_count
    ):
        """The count comes from the hub's own matcher, so it cannot drift from the fetch."""
        hf_mock(["train/a.png", "train/b.png", "val/c.png"])
        _hf_extract("repo", "dataset", tmp_path, allow_patterns=allow_patterns, verbose=True)
        assert f"Downloading {expected_count} files ..." in capsys.readouterr().out

    def test_hf_extract_delegates_the_transfer_to_snapshot_download(self, hf_mock, tmp_path):
        """One parallel call, not one request per file."""
        calls = hf_mock(["train/a.png", "train/b.png", "val/c.png"])
        _hf_extract("repo", "dataset", tmp_path, allow_patterns="train/*")
        assert calls == [
            {
                "repo_id": "repo",
                "repo_type": "dataset",
                "local_dir": tmp_path,
                "patterns": "train/*",
                "workers": HF_MAX_WORKERS_ANONYMOUS,
            }
        ]

    @pytest.mark.parametrize(
        "token, expected_workers",
        [(None, HF_MAX_WORKERS_ANONYMOUS), ("hf_abc123", HF_MAX_WORKERS_AUTHENTICATED)],
    )
    def test_hf_extract_widens_only_when_authenticated(self, hf_mock, monkeypatch, tmp_path, token, expected_workers):
        """Anonymous callers are rate-limited, and a 429 outlasts any backoff worth waiting."""
        calls = hf_mock(["a.png"])
        monkeypatch.setattr("maite_datasets._fileio.get_token", lambda: token)
        _hf_extract("repo", "dataset", tmp_path)
        assert calls[0]["workers"] == expected_workers

    def test_hf_extract_points_anonymous_users_at_hf_token(self, capsys, hf_mock, tmp_path):
        hf_mock([f"{i}.png" for i in range(501)])
        _hf_extract("repo", "dataset", tmp_path, verbose=True)
        assert "Set HF_TOKEN" in capsys.readouterr().out

    def test_hf_extract_resumes_after_a_rate_limit(self, capsys, monkeypatch, tmp_path):
        """A 429 surfaces as a bare ConnectionError; each retry resumes from disk."""
        monkeypatch.setattr("maite_datasets._fileio.time.sleep", lambda _: None)
        attempts = []

        def flaky_snapshot(**kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("Network error: HTTP status client error (429 Too Many Requests)")

        class FakeApi:
            def list_repo_files(self, repo_id, repo_type):
                return ["a.png"]

        monkeypatch.setattr("maite_datasets._fileio.HfApi", FakeApi)
        monkeypatch.setattr("maite_datasets._fileio.snapshot_download", flaky_snapshot)
        monkeypatch.setattr("maite_datasets._fileio.get_token", lambda: None)
        _hf_extract("repo", "dataset", tmp_path, verbose=True)
        assert len(attempts) == 3
        assert "429 Too Many Requests" in capsys.readouterr().out

    def test_hf_extract_gives_up_after_the_last_attempt(self, monkeypatch, tmp_path):
        monkeypatch.setattr("maite_datasets._fileio.time.sleep", lambda _: None)
        attempts = []

        def always_fails(**kwargs):
            attempts.append(1)
            raise ConnectionError("429 Too Many Requests")

        class FakeApi:
            def list_repo_files(self, repo_id, repo_type):
                return ["a.png"]

        monkeypatch.setattr("maite_datasets._fileio.HfApi", FakeApi)
        monkeypatch.setattr("maite_datasets._fileio.snapshot_download", always_fails)
        monkeypatch.setattr("maite_datasets._fileio.get_token", lambda: None)
        with pytest.raises(RuntimeError, match="re-running resumes") as e:
            _hf_extract("repo", "dataset", tmp_path)
        assert len(attempts) == HF_ATTEMPTS
        # The hub's own error is preserved as the cause rather than swallowed.
        assert isinstance(e.value.__cause__, ConnectionError)
        assert "HF_TOKEN" in str(e.value)

    def test_hf_extract_warns_on_large_download(self, capsys, hf_mock, tmp_path):
        hf_mock([f"{i}.png" for i in range(501)])
        _hf_extract("repo", "dataset", tmp_path, verbose=True)
        assert "Downloading 501 files. This may take a while ..." in capsys.readouterr().out

    def test_hf_extract_without_huggingface_hub(self, monkeypatch, tmp_path):
        monkeypatch.setattr("maite_datasets._fileio.HfApi", None)
        with pytest.raises(ImportError, match="huggingface-hub is a required library"):
            _hf_extract("repo", "dataset", tmp_path)
