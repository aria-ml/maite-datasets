import os
import re
from sys import version_info

import nox

PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"
IS_CI = bool(os.environ.get("CI"))

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "lint"]

COMMON_ENVS = {"TQDM_DISABLE": "1"}


def prep(session: nox.Session) -> str:
    session.env["UV_PROJECT_ENVIRONMENT"] = session.env["VIRTUAL_ENV"]
    version = session.name
    pattern = re.compile(r".*(3.\d+)$")
    matches = pattern.match(version)
    version = matches.groups()[0] if matches is not None and len(matches.groups()) > 0 else PYTHON_VERSION
    return version


@nox.session
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`."""
    python_version = prep(session)
    cov_args = ["--cov", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    session.run_install("uv", "sync", "--no-dev", "--all-extras", "--group=test")
    session.run(
        "pytest",
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *session.posargs,
        env={**COMMON_ENVS},
    )
    session.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@nox.session
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    prep(session)
    session.run_install("uv", "sync", "--no-dev", "--all-extras", "--group=type")
    session.run("pyright", "--stats", "src/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "maite_datasets")


@nox.session
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    prep(session)
    session.run_install("uv", "sync", "--only-group=lint")
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")
