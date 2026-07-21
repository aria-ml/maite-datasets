from __future__ import annotations

__all__ = []

import hashlib
import re
import shutil
import tarfile
import zipfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import requests

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import os

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None

ARCHIVE_ENDINGS = [".zip", ".tar", ".tgz"]
COMPRESS_ENDINGS = [".gz", ".bz2"]


def _print(text: str, verbose: bool) -> None:
    if verbose:
        print(text)


def _validate_file(fpath: Path | str, file_md5: str, md5: bool = False, chunk_size: int = 65535) -> bool:
    hasher = hashlib.md5(usedforsecurity=False) if md5 else hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _session_setup(kaggle: bool, timeout: int) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",  # noqa: E501
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
    )

    if kaggle:
        response = session.get(
            "https://www.kaggle.com",
            timeout=timeout,
        )
        response.raise_for_status()
        # session.cookies is auto-populated by requests' Session machinery --
        # ka_sessionid (and anything else Kaggle sets) is already there.
    else:
        session.headers["Referer"] = "https://google.com/"

    return session


def _download_dataset(
    url: str, file_path: Path, kaggle: bool = False, timeout: int = 60, verbose: bool = False
) -> None:
    """Download a single resource from its URL to the `data_folder`."""
    error_msg = "URL fetch failure on {}: {} -- {}"
    try:
        session = _session_setup(kaggle, timeout)
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Large Google Drive files return an HTML "can't scan for viruses" warning page
        # instead of the file. The page is a <form> whose hidden inputs carry the confirm
        # token; re-request the form action with those inputs (the cookie set on this
        # first response is what lets the resubmission serve the real bytes). Note this
        # only works when the resource URL is the drive.google.com/uc?export=download&id=
        # form -- a pre-confirmed drive.usercontent.google.com/...&confirm=t URL loops on
        # the warning page instead. If Google changes this form layout and this proves
        # fragile, swap to the `gdown` library which tracks these changes.
        if "text/html" in response.headers.get("content-type", ""):
            action = re.search(r'id="download-form" action="([^"]+)"', response.text)
            params = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', response.text))
            if action is not None:
                # Drop the session's "Referer: google.com" header on the resubmit -- with it
                # present, Google re-serves the warning page instead of the file bytes.
                # (requests removes a header whose value is None.)
                response = session.get(
                    action.group(1), params=params, headers={"Referer": ""}, stream=True, timeout=timeout
                )
                response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            raise RuntimeError(f"{error_msg.format(url, e.response.status_code, e.response.reason)}") from e
        raise RuntimeError(f"{error_msg.format(url, 'Unknown error', str(e))}") from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"{error_msg.format(url, 'Unknown error', str(e))}") from e

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB
    progress_bar = None if tqdm is None else tqdm(total=total_size, unit="iB", unit_scale=True, disable=not verbose)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            if progress_bar is not None:
                progress_bar.update(len(chunk))
    if progress_bar is not None:
        progress_bar.close()


def _extract_zip_archive(file_path: Path, extract_to: Path) -> None:
    """Extracts the zip file to the given directory."""
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)  # noqa: S202
            file_path.unlink()
    except zipfile.BadZipFile:
        raise FileNotFoundError(f"{file_path.name} is not a valid zip file, skipping extraction.")


def _extract_tar_archive(file_path: Path, extract_to: Path) -> None:
    """Extracts a tar file (or compressed tar) to the specified directory."""
    try:
        with tarfile.open(file_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)  # noqa: S202
            file_path.unlink()
    except tarfile.TarError:
        raise FileNotFoundError(f"{file_path.name} is not a valid tar file, skipping extraction.")


def _extract_archive(
    file_ext: str,
    file_path: Path,
    directory: Path,
    compression: bool = False,
    verbose: bool = False,
) -> None:
    """
    Single function to extract and then flatten if necessary.
    Recursively extracts nested zip files as well.
    Extracts and flattens all folders to the base directory.
    """
    if file_ext != ".zip" or compression:
        _extract_tar_archive(file_path, directory)
    else:
        _extract_zip_archive(file_path, directory)
    # Look for nested zip files in the extraction directory and extract them recursively.
    # Does NOT extract in place - extracts everything to directory
    for child in directory.iterdir():
        if child.suffix == ".zip":
            _print(f"Extracting nested zip: {child} to {directory}", verbose)
            _extract_zip_archive(child, directory)


def _ensure_exists(
    url: str,
    filename: str,
    md5: bool,
    checksum: str,
    kaggle: bool,
    directory: Path,
    root: Path,
    download: bool = True,
    verbose: bool = False,
) -> None:
    """
    For each resource, download it if it doesn't exist in the dataset_dir.
    If the resource is a zip file, extract it (including recursively extracting nested zips).
    """
    file_path = directory / str(filename)
    alternate_path = root / str(filename)
    _, file_ext = file_path.stem, file_path.suffix
    compression = False
    if file_ext in COMPRESS_ENDINGS:
        file_ext = file_path.suffixes[0]
        compression = True

    check_path = alternate_path if alternate_path.exists() and not file_path.exists() else file_path

    # Download file if it doesn't exist.
    if not check_path.exists() and download:
        _print(f"Downloading {filename} from {url}", verbose)
        _download_dataset(url, check_path, kaggle=kaggle, verbose=verbose)
        if not _validate_file(check_path, checksum, md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")

        # If the file is a zip, tar or tgz extract it into the designated folder.
        if file_ext in ARCHIVE_ENDINGS:
            _print(f"Extracting {filename}...", verbose)
            _extract_archive(file_ext, check_path, directory, compression, verbose)

    elif not check_path.exists() and not download:
        raise FileNotFoundError(
            "Data could not be loaded with the provided root directory, "
            f"the file path to the file {filename} does not exist, "
            "and the download parameter is set to False."
        )
    else:
        if not _validate_file(check_path, checksum, md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")
        _print(f"{filename} already exists, skipping download.", verbose)

        if file_ext in ARCHIVE_ENDINGS:
            _print(f"Extracting {filename}...", verbose)
            _extract_archive(file_ext, check_path, directory, compression, verbose)


def _remove_folder_nest(directory: str | Path, overwrite: bool = False, verbose: bool = False) -> None:
    """
    Moves all files and subfolders up one level in the folder structure.
    Checks for overwrites first and once finished, removes folder if empty.
    """
    source_dir = Path(directory)
    parent_dir = source_dir.parent

    # Moving everything up one level without overwrites
    not_moved = []
    for item in source_dir.iterdir():
        destination = parent_dir / item.name
        if overwrite or not destination.exists():
            shutil.move(str(item), str(destination))
        else:
            not_moved.append(str(item.name))

    if not any(source_dir.iterdir()):
        source_dir.rmdir()

    if not not_moved:
        _print("All contents moved up one level successfully!", verbose)
    else:
        _print(f"The following files were not moved:\n\t{(', '.join(not_moved))}", verbose)


def _hf_extract(
    repo_id: str,
    repo_type: Literal["dataset", "model"],
    local_dir: Path,
    allow_patterns: list[str] | str | None = None,
    verbose: bool = False,
) -> None:
    """Downloads dataset from Huggingface to local_dir.

    ``allow_patterns`` are shell-style globs matched against the repo-relative file
    paths (``fnmatch``, so ``*`` also spans ``/``); None selects every file.
    """
    if HfApi is None or hf_hub_download is None:
        raise ImportError(
            "huggingface-hub is a required library to download from huggingface. "
            "Either download maite-datasets[hf-hub] or pip install huggingface-hub."
        )

    api = HfApi()
    filelist = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    if allow_patterns is None:
        selected_files = list(filelist)
    else:
        patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns)
        selected_files = [f for f in filelist if any(fnmatch(f, pattern) for pattern in patterns)]
    num_files = len(selected_files)
    extra = ". This may take a while ..." if num_files > 500 else " ..."
    _print(f"Downloading {num_files} files{extra}", verbose)

    for filename in selected_files:
        if not (local_dir / filename).exists():
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_dir,
            )
