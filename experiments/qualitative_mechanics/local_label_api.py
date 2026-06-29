#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal local HTTP API for the qualitative mechanics web-app prototype."""

from __future__ import annotations

import argparse
import json
import secrets
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import analyze_label_db
import local_label_db


SESSIONS: dict[str, str] = {}


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "")
    if raw_length == "":
        return {}
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise RuntimeError("Invalid Content-Length.") from exc
    body = handler.rfile.read(length)
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("JSON body must be an object.")
    return payload


def require_session(handler: BaseHTTPRequestHandler) -> str:
    header = handler.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        raise PermissionError("Missing bearer token.")
    token = header[len(prefix) :].strip()
    coach_id = SESSIONS.get(token)
    if coach_id is None:
        raise PermissionError("Invalid bearer token.")
    return coach_id


def row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["id"],
        "display_order": row["display_order"],
        "session_pitch": row["session_pitch"],
        "pitcher_id": row["pitcher_id"],
        "p_throws": row["p_throws"],
        "filename_new": row["filename_new"],
        "active_label_fields": row["active_label_fields"].split(";"),
    }


def save_pitch_labels(coach_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    session_pitch = str(payload.get("session_pitch", "")).strip()
    labels = payload.get("labels")
    view_used = str(payload.get("view_used", "")).strip()
    playback_speed = str(payload.get("playback_speed", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    if not session_pitch:
        raise RuntimeError("session_pitch is required.")
    if not isinstance(labels, dict):
        raise RuntimeError("labels must be an object.")
    if not view_used:
        raise RuntimeError("view_used is required.")
    if not playback_speed:
        raise RuntimeError("playback_speed is required.")

    expected_fields = set(local_label_db.ACTIVE_LABEL_FIELDS)
    provided_fields = set(labels)
    if provided_fields != expected_fields:
        missing = sorted(expected_fields.difference(provided_fields))
        extra = sorted(provided_fields.difference(expected_fields))
        raise RuntimeError(f"labels must contain exactly the active fields. missing={missing} extra={extra}")

    created_at = datetime.now(timezone.utc).isoformat()
    with local_label_db.connect() as conn:
        task = conn.execute(
            "SELECT session_pitch FROM label_tasks WHERE session_pitch = ? AND active = 1",
            (session_pitch,),
        ).fetchone()
        if task is None:
            raise RuntimeError(f"No active task for session_pitch={session_pitch}.")
        for item_name, label_value_raw in labels.items():
            label_value = str(label_value_raw).strip()
            if label_value not in local_label_db.FIELD_ALLOWED_VALUES[item_name]:
                raise RuntimeError(f"Invalid label value for {item_name}: {label_value}")
        before_pending = len(local_label_db.pending_tasks(conn, coach_id))
        for item_name in local_label_db.ACTIVE_LABEL_FIELDS:
            conn.execute(
                """
                INSERT INTO labels (
                    coach_id, session_pitch, item_name, label_value, view_used,
                    playback_speed, skipped, skip_reason, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, '', ?, ?)
                """,
                (
                    coach_id,
                    session_pitch,
                    item_name,
                    str(labels[item_name]).strip(),
                    view_used,
                    playback_speed,
                    notes,
                    created_at,
                ),
            )
        after_pending = len(local_label_db.pending_tasks(conn, coach_id))
    return {
        "ok": True,
        "session_pitch": session_pitch,
        "inserted_labels": len(local_label_db.ACTIVE_LABEL_FIELDS),
        "pending_before": before_pending,
        "pending_after": after_pending,
    }


class LabelApiHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        try:
            if self.path == "/api/pending":
                coach_id = require_session(self)
                with local_label_db.connect() as conn:
                    rows = local_label_db.pending_tasks(conn, coach_id)
                json_response(self, 200, {"tasks": [row_to_task(row) for row in rows]})
                return
            if self.path == "/api/analysis":
                require_session(self)
                json_response(self, 200, analyze_label_db.build_summary())
                return
            json_response(self, 404, {"error": "Not found."})
        except PermissionError as exc:
            json_response(self, 401, {"error": str(exc)})
        except Exception as exc:
            json_response(self, 400, {"error": str(exc)})

    def do_POST(self) -> None:
        try:
            payload = parse_json_body(self)
            if self.path == "/api/login":
                name = str(payload.get("name", "")).strip()
                password = str(payload.get("password", ""))
                with local_label_db.connect() as conn:
                    coach_id = local_label_db.login(conn, name, password)
                token = secrets.token_urlsafe(32)
                SESSIONS[token] = coach_id
                json_response(self, 200, {"token": token, "coach_id": coach_id})
                return
            if self.path == "/api/labels":
                coach_id = require_session(self)
                json_response(self, 200, save_pitch_labels(coach_id, payload))
                return
            json_response(self, 404, {"error": "Not found."})
        except PermissionError as exc:
            json_response(self, 401, {"error": str(exc)})
        except sqlite3.IntegrityError as exc:
            json_response(self, 409, {"error": f"Duplicate or invalid label insert: {exc}"})
        except Exception as exc:
            json_response(self, 400, {"error": str(exc)})


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None, token: str | None = None) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def smoke_test() -> None:
    local_label_db.init_db()
    SESSIONS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabelApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, login_payload = request_json(
            f"{base_url}/api/login",
            "POST",
            {"name": "pilot_coach_1", "password": "local-only-test-password"},
        )
        if status != 200:
            raise RuntimeError(f"Login failed: status={status} payload={login_payload}")
        token = str(login_payload["token"])
        status, pending_payload = request_json(f"{base_url}/api/pending", token=token)
        if status != 200 or len(pending_payload["tasks"]) != 28:
            raise RuntimeError(f"Unexpected pending response: status={status} payload={pending_payload}")
        first_task = pending_payload["tasks"][0]
        labels = {field: sorted(local_label_db.FIELD_ALLOWED_VALUES[field])[0] for field in local_label_db.ACTIVE_LABEL_FIELDS}
        status, save_payload = request_json(
            f"{base_url}/api/labels",
            "POST",
            {
                "session_pitch": first_task["session_pitch"],
                "view_used": "test",
                "playback_speed": "1",
                "labels": labels,
            },
            token=token,
        )
        if status != 200 or save_payload["inserted_labels"] != 8 or save_payload["pending_after"] != 27:
            raise RuntimeError(f"Unexpected save response: status={status} payload={save_payload}")
        status, duplicate_payload = request_json(
            f"{base_url}/api/labels",
            "POST",
            {
                "session_pitch": first_task["session_pitch"],
                "view_used": "test",
                "playback_speed": "1",
                "labels": labels,
            },
            token=token,
        )
        if status != 409:
            raise RuntimeError(f"Duplicate save was not blocked: status={status} payload={duplicate_payload}")
        duplicate_status = status
        status, analysis_payload = request_json(f"{base_url}/api/analysis", token=token)
        if status != 200 or len(analysis_payload["item_summaries"]) != 8:
            raise RuntimeError(f"Unexpected analysis response: status={status} payload={analysis_payload}")
        print(f"server={base_url}")
        print(f"login_status=200")
        print(f"pending_before=28")
        print(f"save_inserted_labels=8")
        print(f"pending_after_save={save_payload['pending_after']}")
        print(f"duplicate_status={duplicate_status}")
        print(f"analysis_items={len(analysis_payload['item_summaries'])}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), LabelApiHandler)
    print(f"Serving qualitative label API on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run the local API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--smoke-test", action="store_true", help="Run API smoke tests against a local server thread.")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    if args.serve:
        serve(args.host, args.port)
    if not args.smoke_test and not args.serve:
        parser.error("Choose --serve, --smoke-test, or both.")


if __name__ == "__main__":
    main()
