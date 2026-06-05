import nox

# Prefer running tools in the currently-activated environment (e.g. conda).
# When running `nox` from an activated conda env this file will run the
# external tools from that environment instead of creating new virtualenvs.
nox.options.sessions = ["typecheck", "tests"]
# Try to avoid recreating virtualenvs when possible.
nox.options.reuse_existing_virtualenvs = True


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run mypy type checks against the `src` package using the active env.

    NOTE: This session runs `mypy` as an external command so it is executed
    from the PATH of the currently-active environment. Ensure `mypy` is
    installed in your conda env before running `nox`.
    """
    session.run("mypy", "src", "--config-file", "mypy.ini", external=True)


@nox.session(python="3.11")
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
        "coverage", 
        "report", 
        "--show-missing",  
        "--fail-under=100", 
        external=True
    )
    
    session.run(
        "coverage",
        "xml",
        "-o",
        "coverage.xml",
        external=True
    )
