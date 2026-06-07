"""
tools/mcp_tools.py

Production MCP tool implementations.

These replace the mock_tools.py from the dev phase.
All tools are thin wrappers that call the real downstream systems
(GitHub Actions, Kubernetes, Slack, email) based on env configuration.

Each function returns a dict with at least:
    {"status": "success" | "failed", "detail": str}
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import json
import subprocess
import sys

import requests

from pathlib import Path

load_dotenv()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolExecutionError(Exception):
    """Raised when a tool call fails and cannot be retried."""


# Hardcoded absolute path reference as per your architecture spec
SCRIPT_PATH =  Path(__file__).parent.parent.parent / "fraud_model_server/model_server/scripts/train.py"
LOCKFILE_PATH = Path(__file__).parent.parent.parent / "fraud_model_server/model_server/model/.retrain.lock"
RETRAIN_LOG_DIR = Path(__file__).parent.parent.parent / "data/logs/retrain"


# ---------------------------------------------------------------------------
# Remediation tools
# ---------------------------------------------------------------------------

def trigger_retraining_pipeline(
    model_id:        str,
    environment:     str,
    reason:          str,
    severity:        str          = "unknown",
    prescription:    dict         = None,   
    current_metrics: dict         = None,
    triggered_by:    str          = "remediation_agent",
    active_dataset:  str          = "baseline",
) -> dict[str, Any]:
    """
    Spawns background training subprocess locally or dispatches cloud workflow.
    Aligns paths seamlessly between agentic_ai_project and fraud_model_server environments.
    """
    prescription = prescription or {}
    current_metrics = current_metrics or {}

    if _is_retrain_in_progress():
        logger.info("Retrain already in progress — skipping execution.")
        return {"status": "skipped", "detail": "Retrain execution loop already active."}

    # ── LOCAL SUBPROCESS MODE ────────────────────────────────────────────────
    if os.getenv("LOCAL_MODE", "false").lower() == "true":
        logger.info("[LOCAL MODE] Spawning train.py process context...")

        # TARGET CWD: Set this to the parent folder of scripts (".../model_server")
        # This forces Path("./data/creditcard.csv") inside train.py to resolve to
        # /home/ali/fraud_model_server/model_server/data/creditcard.csv
        target_cwd = str(Path(SCRIPT_PATH).parent.parent)

        local_env = os.environ.copy()
        local_env.update({
            # Disable Python's stdio block-buffering so train.py prints reach
            # the log file in real time and aren't lost on a crash mid-import.
            "PYTHONUNBUFFERED":    "1",
            "MODEL_ID":            str(model_id),
            "ENVIRONMENT":         str(environment),
            "TRIGGERED_BY":        str(triggered_by),
            "FULL_TRAIN":            "true",
            "DRIFT_DATASET":         str(active_dataset),
            "DATA_STRATEGY":         str(prescription.get("data_strategy", "recent_window")),
            "WINDOW_DAYS":           str(prescription.get("window_days", 30)),
            "DRIFT_PERIOD_WEIGHT":   str(prescription.get("drift_period_weight", 1.5)),
            "EXCLUDE_BEFORE":        str(prescription.get("exclude_before", "")),
            "REFIT_PREPROCESSORS":   str(prescription.get("refit_preprocessors", True)).lower(),
            "DRIFTED_FEATURES":      json.dumps(prescription.get("drifted_features", [])),
            "OPTIMIZE_FOR":          str(prescription.get("optimize_for", "recall")),
            "TARGET_RECALL":         str(prescription.get("target_recall", 0.80)),
            "TARGET_ROC_AUC":        str(prescription.get("target_roc_auc", 0.88)),
            "DEPLOYMENT_STRATEGY":   str(prescription.get("deployment_strategy", "canary")),
        })

        # Redirect stdout/stderr into a timestamped, model-tagged log file so
        # the API + dashboard can tail it. Path goes into the lockfile so the
        # retrain-status endpoints can find it without filesystem scanning.
        RETRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Sanitise model_id for the filename — strip non-alphanumeric.
        safe_model = "".join(c if c.isalnum() else "_" for c in str(model_id))[:64]
        log_path = RETRAIN_LOG_DIR / f"{ts}-{safe_model}.log"

        try:
            log_fh = log_path.open("w", buffering=1)  # line-buffered
            # Write a header so the consumer can see what triggered the run.
            log_fh.write(
                f"# retrain log — model_id={model_id} environment={environment}\n"
                f"# started_at={datetime.now(timezone.utc).isoformat()} triggered_by={triggered_by}\n"
                f"# strategy={prescription.get('data_strategy')} window={prescription.get('window_days')}d "
                f"optimize_for={prescription.get('optimize_for')}\n"
                f"# severity={severity} reason={reason!r}\n"
                f"# ─────────────────────────────────────────────────────────────\n"
            )
            log_fh.flush()

            # Launch background process; stderr merges into stdout for a single tail target.
            process = subprocess.Popen(
                [sys.executable, SCRIPT_PATH],
                env=local_env,
                cwd=target_cwd,  # Crucial alignment step
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

            # Drop the local process lockfile tracking asset
            lock_payload = {
                "pid": process.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "model_id": model_id,
                "log_path": str(log_path),
            }
            lock_file = Path(LOCKFILE_PATH)
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file.write_text(json.dumps(lock_payload))

            logger.info(
                "[LOCAL MODE] spawned train.py — pid=%d log=%s",
                process.pid, log_path,
            )
            return {
                "status": "success",
                "detail": (
                    f"LOCAL SUCCESS: Training script spawned in background. "
                    f"PID={process.pid} log={log_path.name}"
                ),
                "local_pid": process.pid,
                "log_path":  str(log_path),
            }
        except Exception as local_exc:
            logger.info("[LOCAL MODE] Failed launching local training subprocess: %s", local_exc)
            return {"status": "failed", "detail": f"Local process spawn error: {local_exc}"}

    # ── 2. CLOUD GITHUB ACTIONS EXECUTION MODE (Original Fallback) ──────────
    try:
        token       = os.environ["GITHUB_TOKEN"]
        owner       = os.environ["GITHUB_OWNER"]
        repo        = os.environ["GITHUB_REPO"]
        workflow_id = os.environ["GITHUB_RETRAIN_WORKFLOW_ID"]
        ref         = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

        url = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/actions/workflows/{workflow_id}/dispatches"
        )

        payload = {
            "ref": ref,
            "inputs": {
                # identity
                "model_id":              model_id,
                "environment":           environment,
                "severity":              severity,
                "reason":                reason,
                "triggered_by":          triggered_by,

                # data prescription
                "data_strategy":         prescription.get("data_strategy", "recent_window"),
                "window_days":           str(prescription.get("window_days", 30)),
                "drift_period_weight":   str(prescription.get("drift_period_weight", 1.5)),
                "exclude_before":        prescription.get("exclude_before", ""),
                "refit_preprocessors":   str(prescription.get("refit_preprocessors", True)).lower(),
                "drifted_features":      json.dumps(prescription.get("drifted_features", [])),

                # model / threshold prescription
                "optimize_for":          prescription.get("optimize_for", "recall"),
                "target_recall":         str(prescription.get("target_recall", 0.80)),
                "target_roc_auc":        str(prescription.get("target_roc_auc", 0.88)),

                # deployment prescription
                "deployment_strategy":   prescription.get("deployment_strategy", "canary"),
                "canary_traffic_pct":    str(prescription.get("canary_traffic_pct", 10)),
                "shadow_period_hours":   str(prescription.get("shadow_period_hours", 2)),

                # current degraded metrics (for the workflow log)
                "current_metrics":       json.dumps(current_metrics),

                "drift_dataset": active_dataset,
            },
        }

        resp = requests.post(
            url,
            json=payload,
            headers={
                "Authorization":       f"Bearer {token}",
                "Accept":              "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )
        resp.raise_for_status()

        logger.info(
            "Retrain dispatched for %s (%s) — strategy=%s window=%sd",
            model_id,
            environment,
            prescription.get("data_strategy"),
            prescription.get("window_days"),
        )

        return {
            "status": "success",
            "detail": (
                f"Retrain workflow dispatched. "
                f"strategy={prescription.get('data_strategy')} "
                f"window={prescription.get('window_days')}d "
                f"optimize_for={prescription.get('optimize_for')}"
            ),
            "workflow_url": f"https://github.com/{owner}/{repo}/actions",
        }

    except Exception as exc:
        logger.error("trigger_retraining_pipeline failed: %s", exc)
        return {"status": "failed", "detail": str(exc)}
    
# Max wall-clock age before a held lock is presumed dead and cleaned up.
# Overridable via env so this can be tuned for long training runs.
_MAX_LOCK_AGE_SECONDS = int(os.getenv("RETRAIN_LOCK_MAX_AGE_SECONDS", str(2 * 3600)))


def _is_zombie(pid: int) -> bool:
    """
    Return True iff `pid` is a Linux zombie (state 'Z' in /proc/<pid>/stat).

    `os.kill(pid, 0)` returns success on zombies because the PID is still in
    the process table — but the process is dead and cannot do anything.
    """
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
        # Format: pid (comm) state ... — comm may contain spaces/parens, so
        # parse from the LAST ')' which terminates the comm field.
        after_comm = stat_line.rsplit(") ", 1)[-1]
        state = after_comm.split(" ", 1)[0]
        return state == "Z"
    except (FileNotFoundError, PermissionError, OSError):
        return False  # not Linux, or can't read; assume not zombie


def _clean_stale_lock(reason: str) -> None:
    """Remove the lockfile and log why."""
    logger.info("[LOCAL CHECK] cleaning stale retrain lockfile — %s", reason)
    Path(LOCKFILE_PATH).unlink(missing_ok=True)


def _is_retrain_in_progress() -> bool:
    """
    Decide whether a retrain is currently active.

    In local mode the lockfile is the source of truth. Several stale states
    must be detected explicitly — `os.kill(pid, 0)` alone is not enough:
      • zombie children (Z state) — process is dead but still in the table
      • lock older than _MAX_LOCK_AGE_SECONDS — assume crashed without cleanup
      • PID dead (ProcessLookupError) — original case
    """
    if os.getenv("LOCAL_MODE", "false").lower() != "true":
        return _is_retrain_in_progress_gh()

    lock_file = Path(LOCKFILE_PATH)
    if not lock_file.exists():
        return False

    try:
        lock_data = json.loads(lock_file.read_text())
    except Exception as exc:
        _clean_stale_lock(f"unreadable lockfile ({exc})")
        return False

    pid = lock_data.get("pid")
    if not pid:
        _clean_stale_lock("no pid field")
        return False

    # 1. Age check — covers crashes that left the lockfile behind even when the
    #    PID was recycled by an unrelated process.
    started_at_raw = lock_data.get("started_at")
    if started_at_raw:
        try:
            started = datetime.fromisoformat(started_at_raw)
            age = (datetime.now(timezone.utc) - started).total_seconds()
            if age > _MAX_LOCK_AGE_SECONDS:
                _clean_stale_lock(
                    f"age {int(age)}s > max {_MAX_LOCK_AGE_SECONDS}s (pid={pid})"
                )
                return False
        except (TypeError, ValueError):
            # Bad timestamp — don't trust the lock either way; clean it.
            _clean_stale_lock(f"unparseable started_at={started_at_raw!r}")
            return False

    # 2. Liveness check.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _clean_stale_lock(f"pid {pid} dead")
        return False
    except PermissionError:
        # PID exists but is owned by another user. Can't tell what it is —
        # safest to treat as active and wait for the age check to age it out.
        logger.info("[LOCAL CHECK] pid %s owned by another user — treating as active", pid)
        return True
    except Exception as exc:
        logger.info("[LOCAL CHECK] unexpected error checking pid %s: %s — treating as active", pid, exc)
        return True

    # 3. Zombie check — kill(pid,0) succeeds on zombies. Reap and clean.
    if _is_zombie(pid):
        _clean_stale_lock(f"pid {pid} is a zombie (defunct)")
        return False

    logger.info("[LOCAL CHECK] retrain active — pid=%s started_at=%s", pid, started_at_raw)
    return True


def _is_retrain_in_progress_gh() -> bool:
    """Check GitHub Actions for an already-running retrain workflow."""
    try:
        import requests
        token       = os.environ["GITHUB_TOKEN"]
        owner       = os.environ["GITHUB_OWNER"]
        repo        = os.environ["GITHUB_REPO"]
        workflow_id = os.environ["GITHUB_RETRAIN_WORKFLOW_ID"]

        url  = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/actions/workflows/{workflow_id}/runs"
        )
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params={"status": "in_progress"},
            timeout=10,
        )
        resp.raise_for_status()
        return len(resp.json().get("workflow_runs", [])) > 0
    except Exception as exc:
        logger.warning("Could not check workflow status: %s", exc)
        return False   # assume not in progress if check fails

def rollback_deployment(
    model_id: str,
    environment: str,
    reason: str,
) -> dict[str, Any]:
    """
    Roll back a Kubernetes deployment to the previous revision using Helm.

    Required env vars:
        KUBECONFIG                   — path to kubeconfig, or in-cluster auth
        K8S_NAMESPACE                — target namespace
        HELM_RELEASE_NAME_TEMPLATE   — optional, default "{model_id}-{environment}"
    """
    try:
        import subprocess

        namespace = os.environ["K8S_NAMESPACE"]
        release_template = os.getenv(
            "HELM_RELEASE_NAME_TEMPLATE", "{model_id}-{environment}"
        )
        release = release_template.format(
            model_id=model_id, environment=environment
        )
        kubeconfig = os.getenv("KUBECONFIG", "")

        env = {**os.environ}
        cmd = ["helm", "rollback", release, "--namespace", namespace, "--wait"]
        if kubeconfig:
            env["KUBECONFIG"] = kubeconfig

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise ToolExecutionError(result.stderr.strip())

        logger.info("Rolled back Helm release '%s' in namespace '%s'", release, namespace)
        return {
            "status": "success",
            "detail": f"Helm release '{release}' rolled back. stdout: {result.stdout.strip()}",
        }

    except Exception as exc:
        logger.error("rollback_deployment failed: %s", exc)
        return {"status": "failed", "detail": str(exc)}


def scale_deployment(
    model_id: str,
    environment: str,
    replicas: int | None = None,
) -> dict[str, Any]:
    """
    Scale a Kubernetes deployment horizontally.

    If replicas is None, doubles the current replica count (up to MAX_REPLICAS).

    Required env vars:
        K8S_NAMESPACE
        K8S_DEPLOYMENT_NAME_TEMPLATE  — default "{model_id}-{environment}"
        K8S_MAX_REPLICAS              — default 20
    """
    try:
        from kubernetes import client as k8s_client, config as k8s_config

        namespace = os.environ["K8S_NAMESPACE"]
        dep_template = os.getenv(
            "K8S_DEPLOYMENT_NAME_TEMPLATE", "{model_id}-{environment}"
        )
        deployment_name = dep_template.format(
            model_id=model_id, environment=environment
        )
        max_replicas = int(os.getenv("K8S_MAX_REPLICAS", "20"))

        # Load kubeconfig (falls back to in-cluster automatically)
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()

        apps_v1 = k8s_client.AppsV1Api()
        dep = apps_v1.read_namespaced_deployment(deployment_name, namespace)
        current = dep.spec.replicas or 1

        if replicas is None:
            replicas = min(current * 2, max_replicas)

        replicas = min(replicas, max_replicas)
        dep.spec.replicas = replicas
        apps_v1.patch_namespaced_deployment(deployment_name, namespace, dep)

        logger.info(
            "Scaled deployment '%s' from %d → %d replicas",
            deployment_name, current, replicas,
        )
        return {
            "status": "success",
            "detail": (
                f"Deployment '{deployment_name}' scaled from {current} "
                f"to {replicas} replicas."
            ),
            "previous_replicas": current,
            "new_replicas": replicas,
        }

    except Exception as exc:
        logger.error("scale_deployment failed: %s", exc)
        return {"status": "failed", "detail": str(exc)}


def open_github_issue(
    model_id: str,
    environment: str,
    diagnosis: str,
    severity: str,
    metrics: dict,
) -> dict[str, Any]:
    """
    Open a GitHub issue to track an ambiguous incident.

    Required env vars:
        GITHUB_TOKEN
        GITHUB_OWNER
        GITHUB_REPO
        GITHUB_ISSUE_LABELS          — optional, comma-separated, default "mlops,auto"
        GITHUB_ISSUE_ASSIGNEES       — optional, comma-separated
    """
    try:
        import requests

        token = os.environ["GITHUB_TOKEN"]
        owner = os.environ["GITHUB_OWNER"]
        repo = os.environ["GITHUB_REPO"]
        labels = [
            lbl.strip()
            for lbl in os.getenv("GITHUB_ISSUE_LABELS", "mlops,auto").split(",")
            if lbl.strip()
        ]
        assignees = [
            a.strip()
            for a in os.getenv("GITHUB_ISSUE_ASSIGNEES", "").split(",")
            if a.strip()
        ]

        title = f"[MLOps {severity.upper()}] Model degradation — {model_id} ({environment})"
        body = _build_github_issue_body(
            model_id=model_id,
            environment=environment,
            severity=severity,
            diagnosis=diagnosis,
            metrics=metrics,
        )

        resp = requests.post(
            f"https://api.github.com/repos/{owner}/{repo}/issues",
            json={"title": title, "body": body, "labels": labels, "assignees": assignees},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )
        resp.raise_for_status()
        issue = resp.json()
        logger.info("Opened GitHub issue #%s: %s", issue["number"], issue["html_url"])
        return {
            "status": "success",
            "detail": f"Issue #{issue['number']} opened.",
            "issue_url": issue["html_url"],
            "issue_number": issue["number"],
        }

    except Exception as exc:
        logger.error("open_github_issue failed: %s", exc)
        return {"status": "failed", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Notification tools
# ---------------------------------------------------------------------------

def send_slack_notification(
    message: str,
    severity: str,
    incident_id: str | None = None,
    channel: str | None = None,
) -> dict[str, Any]:
    """
    Post a message to Slack via Incoming Webhook or Bot Token.

    Required env vars:
        SLACK_WEBHOOK_URL  OR  (SLACK_BOT_TOKEN + SLACK_DEFAULT_CHANNEL)
        SLACK_SEVERITY_CHANNELS   — optional JSON, e.g.
                                    '{"critical":"#incidents","major":"#mlops-alerts"}'
    """
    try:
        import json
        import requests

        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        bot_token = os.getenv("SLACK_BOT_TOKEN", "")

        severity_channels_raw = os.getenv("SLACK_SEVERITY_CHANNELS", "{}")
        severity_channels: dict = json.loads(severity_channels_raw)

        target_channel = (
            channel
            or severity_channels.get(severity)
            or os.getenv("SLACK_DEFAULT_CHANNEL", "#mlops-alerts")
        )

        emoji = {
            "critical": ":rotating_light:",
            "major": ":warning:",
            "minor": ":information_source:",
            "none": ":white_check_mark:",
        }.get(severity, ":robot_face:")

        blocks = _build_slack_blocks(
            message=message,
            severity=severity,
            emoji=emoji,
            incident_id=incident_id,
        )

        if webhook_url:
            resp = requests.post(
                webhook_url,
                json={"blocks": blocks, "channel": target_channel},
                timeout=10,
            )
            resp.raise_for_status()
        elif bot_token:
            resp = requests.post(
                "https://slack.com/api/chat.postMessage",
                json={"channel": target_channel, "blocks": blocks},
                headers={"Authorization": f"Bearer {bot_token}"},
                timeout=10,
            )
            resp.raise_for_status()
            if not resp.json().get("ok"):
                raise ToolExecutionError(resp.json().get("error", "Slack API error"))
        else:
            raise ToolExecutionError(
                "Neither SLACK_WEBHOOK_URL nor SLACK_BOT_TOKEN is configured."
            )

        logger.info("Slack notification sent (severity=%s, channel=%s)", severity, target_channel)
        return {"status": "success", "detail": f"Sent to {target_channel}"}

    except Exception as exc:
        logger.error("send_slack_notification failed: %s", exc)
        return {"status": "failed", "detail": str(exc)}


def send_email_alert(
    subject: str,
    body: str,
    severity: str,
    incident_id: str | None = None,
) -> dict[str, Any]:
    """
    Send an incident email alert via SMTP or SendGrid.

    Required env vars (SMTP mode):
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
        EMAIL_FROM, EMAIL_TO_CRITICAL, EMAIL_TO_MAJOR, EMAIL_TO_MINOR

    Required env vars (SendGrid mode):
        SENDGRID_API_KEY, EMAIL_FROM, EMAIL_TO_CRITICAL, ...
        Set ALERT_EMAIL_PROVIDER=sendgrid to enable.
    """
    try:
        provider = os.getenv("ALERT_EMAIL_PROVIDER", "smtp").lower()

        to_addresses = _resolve_email_recipients(severity)
        if not to_addresses:
            logger.info("No email recipients configured for severity '%s'. Skipping.", severity)
            return {"status": "skipped", "detail": "No recipients configured."}

        if provider == "sendgrid":
            return _send_via_sendgrid(subject, body, to_addresses)
        return _send_via_smtp(subject, body, to_addresses)

    except Exception as exc:
        logger.error("send_email_alert failed: %s", exc)
        return {"status": "failed", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_email_recipients(severity: str) -> list[str]:
    key_map = {
        "critical": "EMAIL_TO_CRITICAL",
        "major": "EMAIL_TO_MAJOR",
        "minor": "EMAIL_TO_MINOR",
    }
    env_key = key_map.get(severity, "EMAIL_TO_MINOR")
    raw = os.getenv(env_key, os.getenv("EMAIL_TO_MINOR", ""))
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def _send_via_smtp(subject: str, body: str, to_addrs: list[str]) -> dict:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    from_addr = os.environ["EMAIL_FROM"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.sendmail(from_addr, to_addrs, msg.as_string())

    return {"status": "success", "detail": f"Email sent to {to_addrs}"}


def _send_via_sendgrid(subject: str, body: str, to_addrs: list[str]) -> dict:
    import sendgrid
    from sendgrid.helpers.mail import Mail

    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    mail = Mail(
        from_email=os.environ["EMAIL_FROM"],
        to_emails=to_addrs,
        subject=subject,
        plain_text_content=body,
    )
    response = sg.send(mail)
    if response.status_code not in (200, 202):
        raise ToolExecutionError(f"SendGrid returned status {response.status_code}")
    return {"status": "success", "detail": f"Email sent via SendGrid to {to_addrs}"}


def _build_github_issue_body(
    model_id: str,
    environment: str,
    severity: str,
    diagnosis: str,
    metrics: dict,
) -> str:
    return f"""## MLOps Incident — Automated Report

**Model:** `{model_id}`  
**Environment:** `{environment}`  
**Severity:** `{severity.upper()}`  
**Detected at:** {datetime.now(timezone.utc).isoformat()}

### Diagnosis
{diagnosis}

### Metrics at Detection
| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A')} |
| Drift Score | {metrics.get('drift_score', 'N/A')} |
| p99 Latency | {metrics.get('latency_p99_ms', 'N/A')} ms |
| Error Rate | {metrics.get('error_rate', 'N/A')} |
| Predictions (window) | {metrics.get('prediction_count', 'N/A')} |

---
*This issue was opened automatically by the MLOps agent system. 
Please investigate and close once resolved.*
"""


def _build_slack_blocks(
    message: str,
    severity: str,
    emoji: str,
    incident_id: str | None,
) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} MLOps Alert — {severity.upper()}",
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": message},
        },
    ]
    if incident_id:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Incident ID: `{incident_id}`",
                    }
                ],
            }
        )
    return blocks
