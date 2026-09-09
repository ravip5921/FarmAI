#!/usr/bin/env bash

# Build an immutable candidate, smoke-test it in isolation, drain production,
# migrate the shared database, and atomically promote the tested commit.
set -Eeuo pipefail
umask 027

log() {
  printf '[farmai-deploy] %s\n' "$*"
}

fail() {
  printf '[farmai-deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

require_absolute_path() {
  local label="$1"
  local value="$2"
  case "$value" in
    /*) ;;
    *) fail "$label must be an absolute path: $value" ;;
  esac
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$label must be a positive integer"
}

if [ "$#" -ne 1 ]; then
  fail "usage: deploy-release.sh <40-character-git-commit>"
fi

RELEASE_COMMIT="$1"
[[ "$RELEASE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || fail "release commit must be a full lowercase SHA-1"

DEPLOY_CONFIG="${FARMAI_DEPLOY_CONFIG:-/etc/farmai/deploy.env}"
if [ -f "$DEPLOY_CONFIG" ]; then
  # This is a root-owned shell environment file; see deploy/config/.
  # shellcheck disable=SC1090
  source "$DEPLOY_CONFIG"
fi

SOURCE_REPO="${FARMAI_SOURCE_REPO:-/home/ravi/apps/FarmAI}"
DEPLOY_BASE="${FARMAI_DEPLOY_BASE:-/home/ravi/apps/farmai}"
APP_ENV_FILE="${FARMAI_APP_ENV_FILE:-/etc/farmai/farmai.env}"
PYTHON_COMMAND="${FARMAI_PYTHON_COMMAND:-python3.11}"
NVM_DIR="${FARMAI_NVM_DIR:-/home/ravi/.nvm}"
NODE_VERSION="${FARMAI_NODE_VERSION:-22}"
API_SERVICE="${FARMAI_API_SERVICE:-farmai-api.service}"
WORKER_SERVICE="${FARMAI_WORKER_SERVICE:-farmai-worker.service}"
API_URL="${FARMAI_API_URL:-http://127.0.0.1:8000}"
SMOKE_PORT="${FARMAI_SMOKE_PORT:-18080}"
HTTP_ATTEMPTS="${FARMAI_HTTP_ATTEMPTS:-30}"
HTTP_DELAY_SECONDS="${FARMAI_HTTP_DELAY_SECONDS:-2}"
DRAIN_TIMEOUT_SECONDS="${FARMAI_DRAIN_TIMEOUT_SECONDS:-3600}"
DRAIN_STATUS_GRACE_SECONDS="${FARMAI_DRAIN_STATUS_GRACE_SECONDS:-15}"
WORKER_STATUS_MAX_AGE_SECONDS="${FARMAI_WORKER_STATUS_MAX_AGE_SECONDS:-30}"

require_absolute_path "FARMAI_SOURCE_REPO" "$SOURCE_REPO"
require_absolute_path "FARMAI_DEPLOY_BASE" "$DEPLOY_BASE"
require_absolute_path "FARMAI_APP_ENV_FILE" "$APP_ENV_FILE"
require_positive_integer "FARMAI_SMOKE_PORT" "$SMOKE_PORT"
require_positive_integer "FARMAI_NODE_VERSION" "$NODE_VERSION"
require_positive_integer "FARMAI_HTTP_ATTEMPTS" "$HTTP_ATTEMPTS"
require_positive_integer "FARMAI_HTTP_DELAY_SECONDS" "$HTTP_DELAY_SECONDS"
require_positive_integer "FARMAI_DRAIN_TIMEOUT_SECONDS" "$DRAIN_TIMEOUT_SECONDS"
require_positive_integer \
  "FARMAI_DRAIN_STATUS_GRACE_SECONDS" \
  "$DRAIN_STATUS_GRACE_SECONDS"
require_positive_integer \
  "FARMAI_WORKER_STATUS_MAX_AGE_SECONDS" \
  "$WORKER_STATUS_MAX_AGE_SECONDS"

[ -d "$SOURCE_REPO/.git" ] || fail "source repository not found: $SOURCE_REPO"
[ -r "$APP_ENV_FILE" ] || fail "application environment not readable: $APP_ENV_FILE"

# The application environment file is deliberately shell-compatible as well as
# systemd EnvironmentFile-compatible. Candidate and migration commands receive
# the same configuration as the services.
set -a
# shellcheck disable=SC1090
source "$APP_ENV_FILE"
set +a

RELEASES_DIR="$DEPLOY_BASE/releases"
SHARED_DIR="$DEPLOY_BASE/shared"
BACKUPS_DIR="$SHARED_DIR/backups"
TEMP_DIR="$DEPLOY_BASE/tmp"
CURRENT_LINK="$DEPLOY_BASE/current"
RUNTIME_DIR="${FARMAI_UI_RUNTIME_DIR:-$SHARED_DIR/runtime}"
DATABASE_PATH="$RUNTIME_DIR/farmai_ui.sqlite3"
DRAIN_FILE="$RUNTIME_DIR/worker.drain"
WORKER_STATUS_FILE="$RUNTIME_DIR/worker-status.json"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_COMMIT"
SHORT_COMMIT="${RELEASE_COMMIT:0:12}"

require_absolute_path "FARMAI_UI_RUNTIME_DIR" "$RUNTIME_DIR"

install -d -m 0750 "$DEPLOY_BASE" "$RELEASES_DIR" "$SHARED_DIR"
install -d -m 0750 "$BACKUPS_DIR" "$TEMP_DIR" "$RUNTIME_DIR"

command -v flock >/dev/null || fail "flock is required for deployment locking"
exec 9> "$DEPLOY_BASE/deploy.lock"
flock -n 9 || fail "another server-side deployment is already running"

[ -L "$CURRENT_LINK" ] \
  || fail "$CURRENT_LINK must be a symlink to the currently working release"
PREVIOUS_RELEASE="$(readlink -f "$CURRENT_LINK")"
[ -d "$PREVIOUS_RELEASE" ] \
  || fail "current release target does not exist: $PREVIOUS_RELEASE"

STAGING_DIR=""
SMOKE_DIR=""
SMOKE_PID=""
LINK_TEMP=""
ROLLBACK_ARMED=0
SWITCH_COMPLETED=0
API_STOPPED=0
WORKER_STOPPED=0

service_is_active() {
  systemctl is-active --quiet "$1"
}

switch_current_link() {
  local target="$1"
  LINK_TEMP="$DEPLOY_BASE/.current-${BASHPID}"
  if [ -e "$LINK_TEMP" ] || [ -L "$LINK_TEMP" ]; then
    printf '[farmai-deploy] ERROR: temporary current link exists: %s\n' \
      "$LINK_TEMP" >&2
    return 1
  fi
  ln -s "$target" "$LINK_TEMP" || return 1
  mv -Tf "$LINK_TEMP" "$CURRENT_LINK" || return 1
  LINK_TEMP=""
}

safe_remove_tree() {
  local target="$1"
  case "$target" in
    "$TEMP_DIR"/*|"$RELEASE_DIR") rm -rf -- "$target" ;;
    *) log "Refusing to remove unexpected temporary path: $target" ;;
  esac
}

wait_for_http() {
  local url="$1"
  local output="$2"
  local attempt
  for ((attempt = 1; attempt <= HTTP_ATTEMPTS; attempt += 1)); do
    if curl \
      --fail \
      --silent \
      --show-error \
      --connect-timeout 2 \
      --max-time 10 \
      "$url" \
      > "$output"; then
      return 0
    fi
    sleep "$HTTP_DELAY_SECONDS"
  done
  return 1
}

wait_for_worker_state() {
  local helper="$1"
  local updated_after="$2"
  local timeout_seconds="$3"
  shift 3
  local deadline=$((SECONDS + timeout_seconds))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if "$helper" \
      "$WORKER_STATUS_FILE" \
      --states "$@" \
      --updated-after "$updated_after" \
      --max-age "$WORKER_STATUS_MAX_AGE_SECONDS"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

wait_for_legacy_queue() {
  local helper="$1"
  local deadline=$((SECONDS + DRAIN_TIMEOUT_SECONDS))
  local consecutive_idle_checks=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if "$helper" "$DATABASE_PATH"; then
      consecutive_idle_checks=$((consecutive_idle_checks + 1))
      if [ "$consecutive_idle_checks" -ge 2 ]; then
        return 0
      fi
    else
      consecutive_idle_checks=0
    fi
    sleep 2
  done
  return 1
}

rollback_or_resume_previous() {
  local rollback_status=0
  local link_restored=1
  log "Deployment failed; restoring the previous release"

  # The sentinel is still present after every post-switch check. A new worker
  # therefore cannot have claimed production work while rollback is possible.
  if [ "$SWITCH_COMPLETED" -eq 1 ]; then
    if service_is_active "$WORKER_SERVICE"; then
      sudo -n systemctl stop "$WORKER_SERVICE" || rollback_status=1
    fi
    WORKER_STOPPED=1
    if service_is_active "$API_SERVICE"; then
      sudo -n systemctl stop "$API_SERVICE" || rollback_status=1
    fi
    API_STOPPED=1
    if switch_current_link "$PREVIOUS_RELEASE"; then
      SWITCH_COMPLETED=0
    else
      rollback_status=1
      link_restored=0
    fi
  fi

  if [ "$link_restored" -ne 1 ]; then
    if [ -n "$LINK_TEMP" ]; then
      rm -f -- "$LINK_TEMP"
      LINK_TEMP=""
    fi
    log "Could not restore the prior symlink; services remain stopped and drained"
    return 1
  fi

  rm -f -- "$DRAIN_FILE"

  if [ "$API_STOPPED" -eq 1 ]; then
    sudo -n systemctl start "$API_SERVICE" || rollback_status=1
    API_STOPPED=0
  fi
  if [ "$WORKER_STOPPED" -eq 1 ]; then
    sudo -n systemctl start "$WORKER_SERVICE" || rollback_status=1
    WORKER_STOPPED=0
  fi

  if [ "$rollback_status" -eq 0 ]; then
    log "Previous release restored: $PREVIOUS_RELEASE"
  else
    log "Automatic rollback encountered an error; inspect both services immediately"
  fi
}

on_exit() {
  local exit_status="$?"
  trap - EXIT ERR
  set +e

  if [ -n "$SMOKE_PID" ] && kill -0 "$SMOKE_PID" 2>/dev/null; then
    kill -TERM "$SMOKE_PID" 2>/dev/null
    wait "$SMOKE_PID" 2>/dev/null
  fi
  if [ -n "$LINK_TEMP" ]; then
    rm -f -- "$LINK_TEMP"
  fi

  if [ "$exit_status" -ne 0 ] && [ "$ROLLBACK_ARMED" -eq 1 ]; then
    rollback_or_resume_previous
  fi

  if [ -n "$SMOKE_DIR" ] && [ -d "$SMOKE_DIR" ]; then
    safe_remove_tree "$SMOKE_DIR"
  fi
  if [ -n "$STAGING_DIR" ] && [ -d "$STAGING_DIR" ]; then
    safe_remove_tree "$STAGING_DIR"
  fi

  exit "$exit_status"
}
trap on_exit EXIT

log "Fetching immutable release commit $RELEASE_COMMIT"
git -C "$SOURCE_REPO" fetch --no-tags origin interface
RESOLVED_COMMIT="$(git -C "$SOURCE_REPO" rev-parse "$RELEASE_COMMIT^{commit}")"
[ "$RESOLVED_COMMIT" = "$RELEASE_COMMIT" ] \
  || fail "source repository did not resolve the requested commit exactly"
git -C "$SOURCE_REPO" merge-base --is-ancestor \
  "$RELEASE_COMMIT" \
  refs/remotes/origin/interface \
  || fail "requested commit is not on origin/interface"

if [ -f "$RELEASE_DIR/REVISION" ] \
  && [ "$(tr -d '\n' < "$RELEASE_DIR/REVISION")" = "$RELEASE_COMMIT" ] \
  && [ -x "$RELEASE_DIR/.venv/bin/python" ] \
  && [ -f "$RELEASE_DIR/user_interface/frontend/dist/index.html" ] \
  && [ -f "$RELEASE_DIR/.farmai-release-ready" ]; then
  CANDIDATE_DIR="$RELEASE_DIR"
  log "Reusing previously verified immutable release $RELEASE_DIR"
else
  if [ -e "$RELEASE_DIR" ]; then
    [ "$PREVIOUS_RELEASE" != "$RELEASE_DIR" ] \
      || fail "the current release is incomplete or invalid: $RELEASE_DIR"
    log "Removing incomplete deploy-managed release $RELEASE_DIR"
    safe_remove_tree "$RELEASE_DIR"
  fi
  install -d -m 0750 "$RELEASE_DIR"
  # The final release path itself is the staging area. Nothing reads it until
  # the current symlink is switched, and keeping this permanent absolute path
  # prevents virtual-environment scripts from referring to a renamed directory.
  STAGING_DIR="$RELEASE_DIR"
  ARCHIVE_PATH="$TEMP_DIR/source-$RELEASE_COMMIT-$BASHPID.tar"
  git -C "$SOURCE_REPO" archive \
    --format=tar \
    --output="$ARCHIVE_PATH" \
    "$RELEASE_COMMIT"
  tar -xf "$ARCHIVE_PATH" -C "$STAGING_DIR"
  rm -f -- "$ARCHIVE_PATH"
  printf '%s\n' "$RELEASE_COMMIT" > "$STAGING_DIR/REVISION"

  log "Installing candidate Python environment"
  command -v "$PYTHON_COMMAND" >/dev/null \
    || fail "Python command is unavailable: $PYTHON_COMMAND"
  "$PYTHON_COMMAND" -m venv "$STAGING_DIR/.venv"
  "$STAGING_DIR/.venv/bin/python" -m pip install --upgrade pip
  "$STAGING_DIR/.venv/bin/python" -m pip install \
    -r "$STAGING_DIR/requirements.txt"
  "$STAGING_DIR/.venv/bin/python" -m pip install -e "$STAGING_DIR"
  "$STAGING_DIR/.venv/bin/python" -m pip check

  if [ -s "$NVM_DIR/nvm.sh" ]; then
    # nvm is not nounset-clean while loading.
    set +u
    # shellcheck disable=SC1091
    source "$NVM_DIR/nvm.sh"
    nvm use --silent "$NODE_VERSION"
    set -u
  fi
  command -v node >/dev/null || fail "Node.js is unavailable"
  command -v npm >/dev/null || fail "npm is unavailable"
  NODE_MAJOR="$(node --version)"
  NODE_MAJOR="${NODE_MAJOR#v}"
  NODE_MAJOR="${NODE_MAJOR%%.*}"
  [ "$NODE_MAJOR" = "$NODE_VERSION" ] \
    || fail "Node.js $NODE_VERSION is required, found $(node --version)"

  log "Linting and building candidate frontend"
  npm --prefix "$STAGING_DIR/user_interface/frontend" ci
  npm --prefix "$STAGING_DIR/user_interface/frontend" run lint
  npm --prefix "$STAGING_DIR/user_interface/frontend" run build
  CANDIDATE_DIR="$STAGING_DIR"
fi

WORKER_STATUS_HELPER="$CANDIDATE_DIR/deploy/check-worker-status.py"
LEGACY_QUEUE_HELPER="$CANDIDATE_DIR/deploy/check-legacy-queue.py"
BACKUP_HELPER="$CANDIDATE_DIR/deploy/backup-sqlite.py"
API_HEALTH_HELPER="$CANDIDATE_DIR/deploy/check-api-health.py"
[ -x "$WORKER_STATUS_HELPER" ] || fail "worker status helper is not executable"
[ -x "$LEGACY_QUEUE_HELPER" ] || fail "legacy queue helper is not executable"
[ -x "$BACKUP_HELPER" ] || fail "database backup helper is not executable"
[ -x "$API_HEALTH_HELPER" ] || fail "API health helper is not executable"

log "Smoke-testing candidate API and frontend on port $SMOKE_PORT"
SMOKE_DIR="$(mktemp -d "$TEMP_DIR/candidate-smoke-$SHORT_COMMIT.XXXXXX")"
SMOKE_RUNTIME="$SMOKE_DIR/runtime"
install -d -m 0750 "$SMOKE_RUNTIME"
SMOKE_LOG="$SMOKE_DIR/api.log"
FARMAI_UI_RUNTIME_DIR="$SMOKE_RUNTIME" \
  "$CANDIDATE_DIR/.venv/bin/python" \
  -m uvicorn user_interface.backend.app:app \
  --app-dir "$CANDIDATE_DIR" \
  --host 127.0.0.1 \
  --port "$SMOKE_PORT" \
  > "$SMOKE_LOG" 2>&1 &
SMOKE_PID="$!"

if ! wait_for_http \
  "http://127.0.0.1:$SMOKE_PORT/api/health" \
  "$SMOKE_DIR/health.json"; then
  sed -n '1,240p' "$SMOKE_LOG" >&2
  fail "candidate API smoke test failed"
fi
"$API_HEALTH_HELPER" "$SMOKE_DIR/health.json" \
  || fail "candidate API returned an invalid health response"
if ! wait_for_http \
  "http://127.0.0.1:$SMOKE_PORT/" \
  "$SMOKE_DIR/index.html"; then
  fail "candidate frontend smoke test failed"
fi
grep -Eq '<div[^>]+id="root"' "$SMOKE_DIR/index.html" \
  || fail "candidate frontend response is not the built application"
kill -TERM "$SMOKE_PID"
wait "$SMOKE_PID"
SMOKE_PID=""

log "Smoke-testing candidate worker against an empty isolated database"
FARMAI_UI_RUNTIME_DIR="$SMOKE_RUNTIME" \
  "$CANDIDATE_DIR/.venv/bin/python" \
  -m user_interface.backend.worker --once

if [ "$CANDIDATE_DIR" = "$STAGING_DIR" ]; then
  printf '%s\n' "$RELEASE_COMMIT" > "$STAGING_DIR/.farmai-release-ready"
  STAGING_DIR=""
  CANDIDATE_DIR="$RELEASE_DIR"
  WORKER_STATUS_HELPER="$CANDIDATE_DIR/deploy/check-worker-status.py"
  LEGACY_QUEUE_HELPER="$CANDIDATE_DIR/deploy/check-legacy-queue.py"
  BACKUP_HELPER="$CANDIDATE_DIR/deploy/backup-sqlite.py"
  API_HEALTH_HELPER="$CANDIDATE_DIR/deploy/check-api-health.py"
fi

if [ "$PREVIOUS_RELEASE" = "$RELEASE_DIR" ]; then
  log "Release $RELEASE_COMMIT is already current; checking live services"
  service_is_active "$API_SERVICE" || fail "current API service is not active"
  service_is_active "$WORKER_SERVICE" || fail "current worker service is not active"
  [ ! -e "$DRAIN_FILE" ] || fail "current worker is unexpectedly left in drain mode"
  CURRENT_HEALTH="$TEMP_DIR/current-health-$BASHPID.json"
  CURRENT_INDEX="$TEMP_DIR/current-index-$BASHPID.html"
  wait_for_http "$API_URL/api/health" "$CURRENT_HEALTH" \
    || fail "current API is not healthy"
  "$API_HEALTH_HELPER" "$CURRENT_HEALTH" \
    || fail "current API returned an invalid health response"
  wait_for_http "$API_URL/" "$CURRENT_INDEX" \
    || fail "current frontend is not healthy"
  grep -Eq '<div[^>]+id="root"' "$CURRENT_INDEX" \
    || fail "current frontend response is not the built application"
  "$WORKER_STATUS_HELPER" \
    "$WORKER_STATUS_FILE" \
    --states starting idle processing \
    --updated-after 0 \
    --max-age "$WORKER_STATUS_MAX_AGE_SECONDS" \
    || fail "current worker heartbeat is not healthy"
  rm -f -- "$CURRENT_HEALTH" "$CURRENT_INDEX"
  log "Current release verification succeeded"
  exit 0
fi

ROLLBACK_ARMED=1
DRAIN_REQUESTED_AT="$($CANDIDATE_DIR/.venv/bin/python -c 'import time; print(time.time())')"
: > "$DRAIN_FILE"
log "Requested worker drain; in-flight analysis may finish before deployment continues"

if wait_for_worker_state \
  "$WORKER_STATUS_HELPER" \
  "$DRAIN_REQUESTED_AT" \
  "$DRAIN_STATUS_GRACE_SECONDS" \
  draining drained; then
  wait_for_worker_state \
    "$WORKER_STATUS_HELPER" \
    "$DRAIN_REQUESTED_AT" \
    "$DRAIN_TIMEOUT_SECONDS" \
    drained \
    || fail "worker did not reach drained state before timeout"
else
  log "No drain-aware worker responded; entering safe legacy adoption mode"
  if service_is_active "$API_SERVICE"; then
    sudo -n systemctl stop "$API_SERVICE"
    API_STOPPED=1
  fi
  wait_for_legacy_queue "$LEGACY_QUEUE_HELPER" \
    || fail "legacy queue did not become idle before timeout"
fi

if service_is_active "$WORKER_SERVICE"; then
  sudo -n systemctl stop "$WORKER_SERVICE"
fi
WORKER_STOPPED=1
service_is_active "$WORKER_SERVICE" \
  && fail "worker service is still active after stop"

if service_is_active "$API_SERVICE"; then
  sudo -n systemctl stop "$API_SERVICE"
fi
API_STOPPED=1
service_is_active "$API_SERVICE" \
  && fail "API service is still active after stop"

if [ -f "$DATABASE_PATH" ]; then
  BACKUP_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  BACKUP_PATH="$BACKUPS_DIR/farmai-ui-$BACKUP_TIMESTAMP-$SHORT_COMMIT.sqlite3"
  log "Creating consistent pre-migration backup $BACKUP_PATH"
  "$BACKUP_HELPER" "$DATABASE_PATH" "$BACKUP_PATH"
else
  log "No existing database found; migration will create the initial database"
fi

log "Applying ordered database migrations"
"$CANDIDATE_DIR/.venv/bin/python" \
  -m user_interface.backend.migrations \
  upgrade \
  --database "$DATABASE_PATH"

log "Atomically promoting $RELEASE_DIR"
switch_current_link "$RELEASE_DIR"
SWITCH_COMPLETED=1
[ "$(tr -d '\n' < "$CURRENT_LINK/REVISION")" = "$RELEASE_COMMIT" ] \
  || fail "active symlink does not resolve to the tested revision"

sudo -n systemctl start "$API_SERVICE"
API_STOPPED=0
service_is_active "$API_SERVICE" || fail "API service failed to start"

POST_HEALTH="$TEMP_DIR/post-health-$BASHPID.json"
POST_INDEX="$TEMP_DIR/post-index-$BASHPID.html"
if ! wait_for_http "$API_URL/api/health" "$POST_HEALTH"; then
  fail "new API did not become healthy"
fi
"$API_HEALTH_HELPER" "$POST_HEALTH" \
  || fail "new API returned an invalid health response"
if ! wait_for_http "$API_URL/" "$POST_INDEX"; then
  fail "new frontend did not become healthy"
fi
grep -Eq '<div[^>]+id="root"' "$POST_INDEX" \
  || fail "new frontend response is not the built application"
rm -f -- "$POST_HEALTH" "$POST_INDEX"

# Start the candidate worker while the drain sentinel still exists. It must
# prove that it is alive and drained before production work is released.
WORKER_STARTED_AT="$($CANDIDATE_DIR/.venv/bin/python -c 'import time; print(time.time())')"
sudo -n systemctl start "$WORKER_SERVICE"
WORKER_STOPPED=0
service_is_active "$WORKER_SERVICE" || fail "worker service failed to start"
wait_for_worker_state \
  "$WORKER_STATUS_HELPER" \
  "$WORKER_STARTED_AT" \
  "$((HTTP_ATTEMPTS * HTTP_DELAY_SECONDS))" \
  drained \
  || fail "new worker did not report a fresh drained state"

rm -f -- "$DRAIN_FILE"
ROLLBACK_ARMED=0
log "Deployment complete; worker claims are enabled for $RELEASE_COMMIT"
