from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = (ROOT / "deploy" / "deploy-release.sh").read_text(encoding="utf-8")
GIT_BASH = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")


def _function(name: str) -> str:
    start = DEPLOY_SCRIPT.index(f"{name}() {{")
    return DEPLOY_SCRIPT[start : DEPLOY_SCRIPT.index("\n}\n", start) + 3]


@unittest.skipUnless(BASH, "Deployment shell checks require Bash")
class TestDeploySmoke(unittest.TestCase):
    def setUp(self) -> None:
        # Keep Unix socket paths below the platform's length limit, including
        # on macOS where the default temporary directory can be very long.
        directory = TemporaryDirectory(dir="/tmp" if os.name == "posix" else None)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_shell(
        self, body: str, *, real_sleep: bool = False
    ) -> subprocess.CompletedProcess:
        prelude = "set -Eeuo pipefail\nHTTP_ATTEMPTS=3\nHTTP_DELAY_SECONDS=1\n"
        prelude += _function("log") + _function("fail") + _function("wait_for_http")
        if not real_sleep:
            prelude += "sleep() { :; }\n"
        return subprocess.run(
            [BASH, "--noprofile", "--norc"],
            input=prelude + body,
            cwd=self.directory,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def test_smoke_requests_use_private_socket_and_bypass_http_proxies(self) -> None:
        result = self.run_shell("""
curl() { printf '%s\\n' "$@" > args.txt; printf '{"status":"ok"}'; }
SMOKE_PID=$$
wait_for_http http://localhost/api/health health.json 'private socket/api.sock'
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.directory / "args.txt").read_text().splitlines()
        self.assertEqual(
            args[args.index("--unix-socket") + 1], "private socket/api.sock"
        )
        self.assertEqual(args[args.index("--noproxy") + 1], "*")
        self.assertIn("--fail", args)
        self.assertEqual(
            (self.directory / "health.json").read_text(), '{"status":"ok"}'
        )

    def test_production_http_checks_keep_their_tcp_transport(self) -> None:
        result = self.run_shell("""
curl() { printf '%s\\n' "$@" > args.txt; printf 'healthy'; }
wait_for_http http://127.0.0.1:8000/api/health health.json
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.directory / "args.txt").read_text().splitlines()
        self.assertNotIn("--unix-socket", args)
        self.assertEqual(args[-1], "http://127.0.0.1:8000/api/health")

    def test_exited_candidate_fails_before_requesting_another_listener(self) -> None:
        result = self.run_shell("""
curl() { printf 'called' > called.txt; printf '{"status":"ok"}'; }
(exit 0) &
SMOKE_PID=$!
wait "$SMOKE_PID"
if wait_for_http http://localhost/api/health health.json api.sock; then
  exit 99
fi
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Candidate API exited", result.stdout)
        self.assertFalse((self.directory / "called.txt").exists())

    def test_startup_requests_retry_but_http_errors_cannot_pass(self) -> None:
        for succeeds in (True, False):
            with self.subTest(succeeds=succeeds):
                result = self.run_shell(f"SUCCEEDS={int(succeeds)}\n" + """
CALLS=0
SMOKE_PID=$$
curl() {
  CALLS=$((CALLS + 1))
  if [ "$SUCCEEDS" -eq 1 ] && [ "$CALLS" -eq 2 ]; then
    printf 'healthy'
    return 0
  fi
  return 22
}
if wait_for_http http://localhost/api/health health.json api.sock; then
  printf 'success:%s' "$CALLS"
else
  printf 'failure:%s' "$CALLS"
fi
""")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout, "success:2" if succeeds else "failure:3"
                )

    @unittest.skipIf(os.name == "nt", "Uvicorn Unix sockets require a Unix server")
    def test_candidate_smoke_checks_ignore_an_occupied_tcp_port(self) -> None:
        requests = []

        class UnauthorizedHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                requests.append(self.path)
                self.send_response(401)
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), UnauthorizedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)

        candidate = self.directory / "candidate"
        python = candidate / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        package = candidate / "user_interface/backend"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").touch()
        (package / "__init__.py").touch()
        (package / "app.py").write_text(
            """from fastapi import FastAPI
from fastapi.responses import HTMLResponse
app = FastAPI()
@app.get('/api/health')
def health():
    return {'status': 'ok', 'schema_version': 4, 'latest_schema_version': 4}
@app.get('/')
def frontend():
    return HTMLResponse('<div id="root"></div>')
""",
            encoding="utf-8",
        )
        helper = self.directory / "check-api-health.py"
        shutil.copyfile(ROOT / "deploy/check-api-health.py", helper)
        helper.chmod(0o755)
        start = DEPLOY_SCRIPT.index('log "Smoke-testing candidate API and frontend')
        end = DEPLOY_SCRIPT.index('log "Smoke-testing candidate worker', start)
        result = self.run_shell(
            f"SMOKE_PORT={server.server_port}\nFARMAI_SMOKE_PORT={server.server_port}\n"
            + """
CANDIDATE_DIR="$PWD/candidate"
TEMP_DIR="$PWD/tmp"
SHORT_COMMIT=0123456789ab
API_HEALTH_HELPER="$PWD/check-api-health.py"
SMOKE_PID=""
HTTP_ATTEMPTS=10
mkdir "$TEMP_DIR"
trap 'if [ -n "$SMOKE_PID" ]; then kill -TERM "$SMOKE_PID" 2>/dev/null || :; wait "$SMOKE_PID" 2>/dev/null || :; fi' EXIT
"""
            + DEPLOY_SCRIPT[start:end],
            real_sleep=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("api=ok schema=4/4", result.stdout)
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
