import hashlib
from pathlib import Path

import pytest
import requests
from requests import RequestException, Response

from maite_datasets._fileio import (
    _download_dataset,
    _ensure_exists,
    _extract_archive,
    _extract_tar_archive,
    _extract_zip_archive,
    _hf_extract,
    _remove_folder_nest,
    _validate_file,
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


@pytest.mark.optional
class TestHelperFunctionsBaseDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_no_zip(self, capsys, dataset_no_zip, verbose):
        resource = ("fakeurl", "stuff.txt", True, TEMP_MD5)
        _ensure_exists(*resource, dataset_no_zip.parent, dataset_no_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert captured.out == "stuff.txt already exists, skipping download.\n"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_single_zip(self, capsys, dataset_single_zip, verbose):
        checksum = get_tmp_hash(dataset_single_zip)
        resource = ("fakeurl", "testing.zip", True, checksum)
        _ensure_exists(
            *resource,
            dataset_single_zip.parent,
            dataset_single_zip.parent,
            True,
            verbose,
        )
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting testing.zip..." in captured.out

    def test_ensure_exists_file_exists_bad_checksum(self, dataset_no_zip):
        resource = ("fakeurl", "stuff.txt", True, TEMP_SHA256)
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(*resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_ensure_exists_download_non_zip(self, capsys, mnist_folder, monkeypatch, tmp_path):
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"fake-mnist-content")
        checksum = hashlib.sha256(payload.read_bytes()).hexdigest()

        def fake_download(url, file_path, *args, **kwargs):
            file_path.write_bytes(payload.read_bytes())

        monkeypatch.setattr("maite_datasets._fileio._download_dataset", fake_download)

        url = "https://example.invalid/mnist.npz"
        resource = (url, "mnist.npz", False, checksum)
        _ensure_exists(*resource, mnist_folder, mnist_folder.parent, True, True)
        captured = capsys.readouterr()
        assert f"Downloading mnist.npz from {url}" in captured.out

    def test_ensure_exists_download_bad_checksum(self, mnist_folder, monkeypatch):
        def fake_download(url, file_path, *args, **kwargs):
            file_path.write_bytes(b"anything")

        monkeypatch.setattr("maite_datasets._fileio._download_dataset", fake_download)

        resource = ("https://example.invalid/mnist.npz", "mnist.npz", False, "abc")
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(*resource, mnist_folder, mnist_folder.parent, True, False)
        assert err_msg in str(e.value)

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

        resource = ("https://example.invalid/fake.zip", "2021.zip", True, checksum)
        _ensure_exists(*resource, mnist_folder, mnist_folder.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert f"Extracting {resource[1]}..." in captured.out

    def test_ensure_exists_error(self, dataset_no_zip):
        resource = ("fakeurl", "something.zip", True, "")
        err_msg = "Data could not be loaded with the provided root directory,"
        with pytest.raises(FileNotFoundError) as e:
            _ensure_exists(*resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
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

    def test_download_html_without_form_is_written_as_is(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            requests.Session,
            "get",
            lambda *args, **kwargs: MockResponse(b"<html>nope</html>", content_type="text/html", text="<html>nope"),
        )
        file_path = tmp_path / "out.html"
        _download_dataset(url="http://mock/", file_path=file_path)
        assert file_path.read_bytes() == b"<html>nope</html>"


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
        """Patch in a fake hub, returning the record of files it was asked to download."""
        downloaded = []

        def fake_download(repo_id, filename, repo_type, local_dir):
            downloaded.append(filename)
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("data")

        def make_api(filelist):
            class FakeApi:
                def list_repo_files(self, repo_id, repo_type):
                    return filelist

            monkeypatch.setattr("maite_datasets._fileio.HfApi", FakeApi)
            monkeypatch.setattr("maite_datasets._fileio.hf_hub_download", fake_download)
            return downloaded

        return make_api

    @pytest.mark.parametrize(
        "allow_patterns, expected",
        [
            (None, ["train/a.png", "train/b.png", "val/c.png"]),
            ("train/*", ["train/a.png", "train/b.png"]),
            (["val/*", "train/a.png"], ["train/a.png", "val/c.png"]),
        ],
    )
    def test_hf_extract_pattern_selection(self, hf_mock, tmp_path, allow_patterns, expected):
        downloaded = hf_mock(["train/a.png", "train/b.png", "val/c.png"])
        _hf_extract("repo", "dataset", tmp_path, allow_patterns=allow_patterns)
        assert downloaded == expected

    def test_hf_extract_skips_existing_files(self, hf_mock, tmp_path):
        downloaded = hf_mock(["a.png", "b.png"])
        (tmp_path / "a.png").write_text("already here")
        _hf_extract("repo", "dataset", tmp_path)
        assert downloaded == ["b.png"]

    def test_hf_extract_warns_on_large_download(self, capsys, hf_mock, tmp_path):
        hf_mock([f"{i}.png" for i in range(501)])
        _hf_extract("repo", "dataset", tmp_path, verbose=True)
        assert "Downloading 501 files. This may take a while ..." in capsys.readouterr().out

    def test_hf_extract_without_huggingface_hub(self, monkeypatch, tmp_path):
        monkeypatch.setattr("maite_datasets._fileio.HfApi", None)
        with pytest.raises(ImportError, match="huggingface-hub is a required library"):
            _hf_extract("repo", "dataset", tmp_path)
