from __future__ import annotations

__all__ = []

import hashlib
import re
import shutil
import tarfile
import zipfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias
from urllib.parse import urlparse

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


class URLResource(NamedTuple):
    """An archive fetched over HTTP and verified against a pinned checksum."""

    url: str
    filename: str
    md5: bool
    checksum: str


class HFResource(NamedTuple):
    """A huggingface repo, fetched through the hub client rather than as an archive.

    Has no checksum: the hub verifies its own transfers, and the repo is a file tree
    rather than a single archive, so there is nothing to pin.
    """

    repo_id: str
    repo_type: Literal["dataset", "model"] = "dataset"
    allow_patterns: list[str] | str | None = None


Resource: TypeAlias = URLResource | HFResource


class ResourcePart(NamedTuple):
    """One component of a dataset, plus the interchangeable places it can be fetched from.

    ``name`` identifies the part independently of whichever mirror served it. Mirrors
    publish the same content under different archive filenames (``archive.zip`` on
    Kaggle vs ``M3FD_Detection.zip`` on Drive), so a name derived from a filename would
    not survive a fallback to the next mirror.
    """

    name: str
    mirrors: tuple[Resource, ...]


def _print(text: str, verbose: bool) -> None:
    if verbose:
        print(text)


def _validate_file(fpath: Path | str, file_md5: str, md5: bool = False, chunk_size: int = 65535) -> bool:
    hasher = hashlib.md5(usedforsecurity=False) if md5 else hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _is_kaggle(url: str) -> bool:
    """Whether `url` points at Kaggle, and so needs the cookie warm-up below.

    Derived from the host rather than carried on :class:`URLResource` so a resource
    can never claim to be something its URL isn't.
    """
    hostname = urlparse(url).hostname or ""
    return hostname == "kaggle.com" or hostname.endswith(".kaggle.com")


def _session_setup(url: str, timeout: int) -> requests.Session:
    """Build the session used to fetch `url`, warming Kaggle cookies when needed."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",  # noqa: E501
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
    )

    if _is_kaggle(url):
        # Kaggle's download API rejects a cold session; one GET of the landing page
        # populates session.cookies (ka_sessionid and friends) via requests' own
        # Session machinery, and the download then carries them.
        response = session.get("https://www.kaggle.com", timeout=timeout)
        response.raise_for_status()
    else:
        session.headers["Referer"] = "https://google.com/"

    return session


def _download_dataset(url: str, file_path: Path, timeout: int = 60, verbose: bool = False) -> None:
    """Download a single resource from its URL to the `data_folder`."""
    error_msg = "URL fetch failure on {}: {} -- {}"
    try:
        session = _session_setup(url, timeout)
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
            else:
                # Not the Drive confirm form, so it's an error/login page (Kaggle serves
                # one when the cookie warm-up fails). Streaming it into a .zip would
                # surface much later as an opaque checksum mismatch.
                raise RuntimeError(error_msg.format(url, "HTML response", "expected a file, received a web page"))
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
    resource: Resource,
    directory: Path,
    root: Path,
    download: bool = True,
    verbose: bool = False,
) -> None:
    """
    For each resource, download it if it doesn't exist in the dataset_dir.
    If the resource is a zip file, extract it (including recursively extracting nested zips).

    Takes the whole :class:`URLResource` rather than its fields so growing the tuple
    never silently shifts the positional arguments that follow it.
    """
    if isinstance(resource, HFResource):
        if not download:
            raise FileNotFoundError(
                "Data could not be loaded with the provided root directory, "
                f"the huggingface repo {resource.repo_id} has not been downloaded, "
                "and the download parameter is set to False."
            )
        _print(f"Downloading {resource.repo_id} from huggingface", verbose)
        _hf_extract(resource.repo_id, resource.repo_type, directory, resource.allow_patterns, verbose)
        return

    url, filename, md5, checksum = resource
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
        _download_dataset(url, check_path, verbose=verbose)
        if not _validate_file(check_path, checksum, md5):
            # Discard the bad bytes; left in place they take the "already exists" branch
            # on every later run, which then fails the same way forever.
            check_path.unlink(missing_ok=True)
            raise Exception(f"File checksum mismatch on downloaded {filename}. Retry the download.")

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


def _part_filename(part: ResourcePart) -> str:
    """Archive filename of `part`'s primary mirror, for loaders that re-open the file.

    Only meaningful for archive mirrors; a huggingface repo unpacks to a file tree with
    no single archive to name.
    """
    primary = part.mirrors[0]
    if isinstance(primary, HFResource):
        raise TypeError(f"{part.name} is fetched from huggingface and has no archive filename.")
    return primary.filename


def _download_part(
    part: ResourcePart,
    directory: Path,
    root: Path,
    download: bool = True,
    verbose: bool = False,
) -> None:
    """Fetch `part` from the first of its mirrors that succeeds.

    Mirrors hold identical content, so whichever one lands is equivalent. The fallback
    earns its keep because these archives are large: hosts throttle them, and a mirror
    that re-publishes its archive invalidates the checksum pinned against it.
    """
    last = len(part.mirrors) - 1
    for index, mirror in enumerate(part.mirrors):
        try:
            _ensure_exists(mirror, directory, root, download, verbose)
            return
        except Exception:
            # Only the last mirror's failure is worth surfacing. With downloads
            # disabled every mirror fails identically, so there is nothing to retry.
            if index == last or not download:
                raise
            _print(f"Could not retrieve {part.name}, trying the next mirror.", verbose)


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
