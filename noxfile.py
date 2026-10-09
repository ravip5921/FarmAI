import nox

# Run all sessions in the activated environment (for example, the farm-ai
# conda environment used locally and in CI).
nox.options.sessions = ["typecheck", "tests", "format"]


@nox.session(venv_backend="none")
def typecheck(session: nox.Session) -> None:
    """Run mypy type checks against the `src` package using the active env.

    NOTE: This session runs `mypy` as an external command so it is executed
    from the PATH of the currently-active environment. Ensure `mypy` is
    installed in your conda env before running `nox`.
    """
    session.run(
        "mypy",
        "src",
        "--config-file",
        "mypy.ini",
        "--no-sqlite-cache",
        external=True,
    )


@nox.session(venv_backend="none")
def tests(session: nox.Session) -> None:
    """Run unit tests under coverage using the active env and emit reports.

    This session invokes `coverage` as an external command so it will use the
    `coverage` installation available in the activated environment.
    """
    session.run(
        "coverage",
        "run",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        external=True,
    )

    session.run(
        "coverage", "report", "--show-missing", "--fail-under=100", external=True
    )

    session.run("coverage", "xml", "-o", "coverage.xml", external=True)


@nox.session(venv_backend="none")
def format(session: nox.Session) -> None:
    """Auto-format code, sort imports, and remove unused imports/variables."""

    paths = ["src", "tests", "noxfile.py"]

    session.run(
        "autoflake",
        "--in-place",
        "--recursive",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        *paths,
        external=True,
    )

    session.run(
        "isort",
        "--profile",
        "black",
        *paths,
        external=True,
    )

    session.run(
        "black",
        "--target-version",
        "py311",
        "--workers",
        "1",
        *paths,
        external=True,
    )
