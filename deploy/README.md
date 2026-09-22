# FarmAI release deployment

The deployment workflow promotes only the exact `interface` commit that passed
the Python test/type gates and the frontend lint/build gates. It never resets
the live checkout or writes build output into the current release.

## Release layout

The checked-in defaults use this layout:

```text
/home/ravi/apps/FarmAI                 source repository; fetch-only
/home/ravi/apps/farmai/current         symlink to the active release
/home/ravi/apps/farmai/releases/<sha>  immutable code, frontend, and venv
/home/ravi/apps/farmai/shared/runtime  SQLite database, documents, job files
/home/ravi/apps/farmai/shared/backups  pre-migration SQLite backups
/etc/farmai/farmai.env                 application configuration and secrets
/etc/farmai/deploy.env                 release-script configuration
```

Each release has its own virtual environment and built frontend, so rolling the
`current` symlink back also rolls back Python packages and static assets. Shared
runtime data is never stored in a release directory.

## One-time server setup

These steps need an administrator and should be performed during a quiet
window. The existing worker does not implement the drain protocol, so first
confirm in the UI/database that it has no queued or running analysis before
stopping it. Pause uploads during this initial adoption.

1. Install Python 3.11 with `venv`, Git, `rsync`, curl, `flock` (normally from
   `util-linux`), Node.js 22, Apache, and the Apache `proxy`/`proxy_http`
   modules. Keep the existing checkout at
   `/home/ravi/apps/FarmAI`; deployments only fetch objects from it.
2. Stop the existing API and worker. Create the release directories owned by
   `ravi`, including `releases`, `shared/runtime`, `shared/backups`, and `tmp`.
3. While both old services are stopped, copy the contents of the old
   `user_interface/runtime/` into `shared/runtime/`. Keep the old copy until the
   new deployment and backup have both been verified.
4. Copy [config/farmai.env.example](config/farmai.env.example) to
   `/etc/farmai/farmai.env` (mode `0640`, readable by `ravi`) and
   [config/deploy.env.example](config/deploy.env.example) to
   `/etc/farmai/deploy.env` (mode `0644`). Update paths, origin, and secrets.
   These files must contain shell-compatible `KEY=value` assignments because
   both systemd and the deployment script read the application file.
5. Initially point `/home/ravi/apps/farmai/current` at the known-working
   `/home/ravi/apps/FarmAI` checkout. This preserves that checkout as the first
   automatic rollback target. Ensure its existing `.venv` and frontend build
   are healthy before continuing.
6. Install the two unit files from [systemd](systemd), adjust `User`, `Group`,
   and paths if needed, then run `systemctl daemon-reload` and enable both
   services. The worker unit intentionally has no forced stop timeout.
7. Validate [sudoers/farmai-deploy](sudoers/farmai-deploy) with `visudo -cf`,
   then install it as `/etc/sudoers.d/farmai-deploy` with mode `0440`. Grant
   only the listed service start/stop commands.
8. Install and enable [apache/farmai.conf](apache/farmai.conf). Add TLS using the
   server's normal certificate process; the checked-in file is the HTTP reverse
   proxy only. Confirm `/api/health` and `/` through Apache.
9. Start the API and worker. Verify that both use the shared runtime directory,
   then push or manually dispatch the workflow for the first managed release.

From the existing checkout, the corresponding setup commands are below. Review
all paths and edit the two environment files before starting the services:

```bash
sudo systemctl stop farmai-worker.service
sudo systemctl stop farmai-api.service

sudo install -d -o ravi -g ravi -m 0750 /home/ravi/apps/farmai
sudo install -d -o ravi -g ravi -m 0750 /home/ravi/apps/farmai/releases
sudo install -d -o ravi -g ravi -m 0750 /home/ravi/apps/farmai/shared/runtime
sudo install -d -o ravi -g ravi -m 0750 /home/ravi/apps/farmai/shared/backups
sudo install -d -o ravi -g ravi -m 0750 /home/ravi/apps/farmai/tmp
sudo -u ravi rsync -a /home/ravi/apps/FarmAI/user_interface/runtime/ /home/ravi/apps/farmai/shared/runtime/
sudo -u ravi ln -s /home/ravi/apps/FarmAI /home/ravi/apps/farmai/current

sudo install -d -o root -g ravi -m 0750 /etc/farmai
sudo install -o root -g ravi -m 0640 deploy/config/farmai.env.example /etc/farmai/farmai.env
sudo install -o root -g root -m 0644 deploy/config/deploy.env.example /etc/farmai/deploy.env

sudo install -o root -g root -m 0644 deploy/systemd/farmai-api.service /etc/systemd/system/farmai-api.service
sudo install -o root -g root -m 0644 deploy/systemd/farmai-worker.service /etc/systemd/system/farmai-worker.service
sudo visudo -cf deploy/sudoers/farmai-deploy
sudo install -o root -g root -m 0440 deploy/sudoers/farmai-deploy /etc/sudoers.d/farmai-deploy

sudo install -o root -g root -m 0644 deploy/apache/farmai.conf /etc/apache2/sites-available/farmai.conf
sudo a2enmod proxy proxy_http
sudo a2ensite farmai.conf
sudo systemctl daemon-reload
sudo systemctl enable farmai-api.service farmai-worker.service
sudo apache2ctl configtest
sudo systemctl reload apache2.service
sudo systemctl start farmai-api.service
sudo systemctl start farmai-worker.service
```

If `current`, either environment file, or a site/unit file already exists, stop
and reconcile it instead of overwriting it blindly. The runtime copy is made
only while both old processes are stopped so the SQLite database and artifacts
represent one consistent point in time.

The first managed deployment detects a legacy worker automatically. It stops
the API to prevent new submissions, waits until the legacy SQLite queue has
been idle on two consecutive checks, and only then stops the worker. Later
deployments use the drain sentinel and remain available while current work
finishes.

## GitHub configuration

The production environment needs these secrets:

- `FARMAI_DEPLOY_HOST`
- `FARMAI_DEPLOY_PORT`
- `FARMAI_DEPLOY_USER`
- `FARMAI_DEPLOY_KEY`
- `FARMAI_DEPLOY_KNOWN_HOSTS` (recommended pinned `known_hosts` line)

If `FARMAI_DEPLOY_KNOWN_HOSTS` is temporarily absent, the workflow falls back
to `ssh-keyscan`; pinning the server key is safer. Protect the GitHub
`production` environment if deployment approval is desired.

## Promotion and rollback behavior

For every release, `deploy-release.sh` performs these operations:

1. Fetch and verify that the requested full SHA is on `origin/interface`.
2. Export that SHA to a new release directory; install isolated dependencies;
   lint/build the frontend again.
3. Start the candidate API on the loopback smoke port with a temporary database,
   verify API/schema/frontend health, and run an empty-queue worker smoke test.
4. Request a production drain. The current worker finishes its active job and
   reports fresh `drained` heartbeats before it is stopped.
5. Stop the API, make a consistent SQLite backup, and invoke:

   ```text
   .venv/bin/python -m user_interface.backend.migrations upgrade --database PATH
   ```

6. Atomically switch `current`, start the API, and validate its API, schema, and
   frontend responses.
7. Start the new worker while the drain sentinel is still present. Only after a
   fresh `drained` heartbeat is observed is the sentinel removed.

Any failure after draining restores the prior symlink and restarts the prior
services. The workflow still fails so the deployment receives attention. The
database is not automatically overwritten from backup: migrations must be
transactional, additive, and compatible with at least the immediately previous
application release. This is what makes code rollback safe without discarding
new uploads. The backup is available for deliberate disaster recovery.

Do not delete the prior release until a newer release has operated successfully.
Release cleanup is intentionally manual so there is always a known rollback
target.
