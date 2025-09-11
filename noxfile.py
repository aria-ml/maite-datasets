import os
import re
from sys import version_info

import nox
import nox_uv

PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
PYTHON_RE_PATTERN = re.compile(r"\d\.\d{1,2}")
IS_CI = bool(os.environ.get("CI"))

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "lint"]


def get_python_version(s: nox.Session) -> str:
    matches = PYTHON_RE_PATTERN.search(s.name)
    return matches.group(0) if matches else PYTHON_VERSION


@nox_uv.session(uv_groups=["test"])
def test(s: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`."""
    python_version = get_python_version(s)
    cov_args = ["--cov", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    s.run_install("uv", "sync", "--no-dev", "--all-extras", "--group=test")
    s.run(
        "pytest",
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *s.posargs,
    )
    s.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@nox_uv.session(uv_groups=["type"])
def type(s: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    s.run_install("uv", "sync", "--no-dev", "--all-extras", "--group=type")
    s.run("pyright", "--stats", "src/")
    s.run("pyright", "--ignoreexternal", "--verifytypes", "maite_datasets")


@nox_uv.session(uv_groups=["lint"])
def lint(s: nox.Session) -> None:
    """Perform linting and spellcheck."""
    s.run_install("uv", "sync", "--only-group=lint")
    s.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    s.run("ruff", "format", "--check" if IS_CI else ".")
    s.run("codespell")
