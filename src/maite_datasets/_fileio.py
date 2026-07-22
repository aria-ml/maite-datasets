from __future__ import annotations

__all__ = []

import hashlib
import os
import re
import shutil
import tarfile
import time
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias
from urllib.parse import urlparse

import requests

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from huggingface_hub import HfApi, get_token, snapshot_download
    from huggingface_hub import constants as hf_constants

    # Documented public toggles, but imported from the defining submodule because the
    # utils package does not re-export them in ``__all__``.
    from huggingface_hub.utils.tqdm import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )
except ImportError:
    HfApi = None
    snapshot_download = None
    get_token = None
    hf_constants = None
    are_progress_bars_disabled = disable_progress_bars = enable_progress_bars = None

ARCHIVE_ENDINGS = [".zip", ".tar", ".tgz"]
COMPRESS_ENDINGS = [".gz", ".bz2"]

# A huggingface repo of loose image files is one request per image, so width decides
# how long a few thousand images take. What it is safe to ask for depends entirely on
# authentication: with a token the hub's default concurrency is fine, without one even
# four workers earns a 429 partway through, and the throttle that follows outlasts any
# backoff worth waiting for. So stay serial when anonymous -- slower, but it finishes.
HF_MAX_WORKERS_ANONYMOUS = 1
HF_MAX_WORKERS_AUTHENTICATED = 8
HF_ATTEMPTS = 5
HF_RETRY_BACKOFF = 5.0


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


# Hub flags this package overrides while downloading, each read as a live attribute of
# huggingface_hub.constants and so settable at runtime.
#   HF_HUB_DISABLE_XET       -- xet's chunk-level dedup pays off on large files but costs
#                               a token negotiation per file, and these repos are
#                               thousands of small images: it measured roughly five times
#                               slower and multiplied requests enough to draw 429s.
#   HF_HUB_DISABLE_TELEMETRY -- fetching a dataset should not phone home for the caller.
# Progress bars are handled separately: the hub binds that constant by value at import
# (``from ..constants import ...``), so only its public toggle has any effect.
HF_DOWNLOAD_FLAGS = ("HF_HUB_DISABLE_XET", "HF_HUB_DISABLE_TELEMETRY")


@contextmanager
def _hf_download_settings() -> Generator[None, None, None]:
    """Apply this package's huggingface preferences for one download, then undo them.

    Scoped to the call rather than set at import on purpose. Importing this package
    should not quietly change huggingface's behaviour for unrelated code in the same
    process, and the env-var-at-import approach it replaces was a silent no-op whenever
    the caller happened to import ``huggingface_hub`` first -- the hub reads each env var
    into a constant as it is imported, so a later write to ``os.environ`` arrives too
    late. Setting the constants is what actually takes effect; the env vars are set
    alongside them only so subprocesses agree. Everything is restored on the way out.
    """
    if (
        hf_constants is None
        or are_progress_bars_disabled is None
        or disable_progress_bars is None
        or enable_progress_bars is None
    ):
        yield
        return

    previous_flags = {name: getattr(hf_constants, name) for name in HF_DOWNLOAD_FLAGS}
    previous_env = {name: os.environ.get(name) for name in HF_DOWNLOAD_FLAGS}
    previously_quiet = are_progress_bars_disabled()

    for name in HF_DOWNLOAD_FLAGS:
        setattr(hf_constants, name, True)
        os.environ[name] = "1"
    # This module prints its own file count, so the hub's bars are redundant noise.
    disable_progress_bars()
    try:
        yield
    finally:
        for name, flag in previous_flags.items():
            setattr(hf_constants, name, flag)
            prior = previous_env[name]
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior
        if not previously_quiet:
            enable_progress_bars()


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
    paths (so ``*`` also spans ``/``); None selects every file. The listing here exists
    only to report the count, and uses ``fnmatchcase`` to match how the hub filters the
    same patterns, so the number reported is the number fetched.
    """
    if HfApi is None or snapshot_download is None or get_token is None:
        raise ImportError(
            "huggingface-hub is a required library to download from huggingface. "
            "Either download maite-datasets[hf-hub] or pip install huggingface-hub."
        )

    api = HfApi()
    filelist = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    patterns = [allow_patterns] if isinstance(allow_patterns, str) else allow_patterns
    num_files = (
        len(filelist)
        if patterns is None
        else sum(1 for f in filelist if any(fnmatchcase(f, pattern) for pattern in patterns))
    )
    extra = ". This may take a while ..." if num_files > 500 else " ..."
    _print(f"Downloading {num_files} files{extra}", verbose)

    authenticated = get_token() is not None
    if not authenticated and num_files > 500:
        _print(
            "No huggingface token found, so files are fetched one at a time to stay under "
            "the anonymous rate limit. Set HF_TOKEN (or run `huggingface-cli login`) to "
            "download in parallel.",
            verbose,
        )

    # Every attempt resumes from what is already on disk, so a dropped transfer costs
    # only the backoff. The retries catch OSError, which covers both the bare
    # ConnectionError the xet transfer layer raises on a 429 and HfHubHTTPError (a
    # requests.RequestException, and so an OSError too).
    for attempt in range(HF_ATTEMPTS):
        try:
            with _hf_download_settings():
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_dir=local_dir,
                    allow_patterns=allow_patterns,
                    max_workers=HF_MAX_WORKERS_AUTHENTICATED if authenticated else HF_MAX_WORKERS_ANONYMOUS,
                )
            return
        except OSError as e:
            if attempt == HF_ATTEMPTS - 1:
                # The anonymous quota is a request count per window, not a concurrency
                # limit, so a repo of this many files can exhaust it at any width and the
                # window outlasts any backoff worth blocking on. Partial progress is kept,
                # so say so -- re-running picks up from here rather than starting over.
                raise RuntimeError(
                    f"Could not finish downloading {repo_id} from huggingface after "
                    f"{HF_ATTEMPTS} attempts: {e}\n"
                    f"Files already fetched are kept in {local_dir}, so re-running resumes "
                    "from there. If this is a rate limit, set HF_TOKEN (or run "
                    "`huggingface-cli login`) for a higher quota."
                ) from e
            delay = HF_RETRY_BACKOFF * 2**attempt
            _print(f"Download interrupted ({e}); resuming in {delay:.0f}s ...", verbose)
            time.sleep(delay)
