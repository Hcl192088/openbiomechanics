#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal local HTTP API for the qualitative mechanics web-app prototype."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import analyze_label_db
import local_label_db


EXPERIMENT_DIR = local_label_db.EXPERIMENT_DIR
WEB_MOTION_MANIFEST_PATH = EXPERIMENT_DIR / "web_motion_manifest.csv"
SESSIONS: dict[str, str] = {}
WEB_MOTION_INDEX: dict[str, str] | None = None

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qualitative Mechanics Labeling</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #1d252d; }
    main { max-width: 1280px; margin: 0 auto; padding: 24px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 20px; }
    h1 { font-size: 24px; margin: 0; }
    h2 { font-size: 18px; margin: 0 0 12px; }
    section { background: #fff; border: 1px solid #d9dee5; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 700; margin-bottom: 6px; }
    input, select, textarea, button { font: inherit; }
    input, select, textarea { box-sizing: border-box; width: 100%; border: 1px solid #b9c1cc; border-radius: 6px; padding: 8px; background: #fff; }
    textarea { min-height: 72px; resize: vertical; }
    button { border: 0; border-radius: 6px; background: #1f6feb; color: #fff; padding: 9px 13px; cursor: pointer; }
    button.secondary { background: #52616f; }
    button:disabled { background: #9aa5b1; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .muted { color: #64707d; font-size: 13px; }
    .status { font-size: 13px; min-height: 20px; }
    .error { color: #b42318; }
    .ok { color: #067647; }
    .task-meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    .task-meta div { background: #eef2f6; border-radius: 6px; padding: 8px; font-size: 13px; }
    .hidden { display: none; }
    #viewer { height: 58vh; min-height: 420px; position: relative; background: #111418; border-radius: 8px; overflow: hidden; }
    #loading { position: absolute; top: 12px; left: 12px; background: #171b21; color: #e8eaed; padding: 8px 10px; border: 1px solid #3a424d; border-radius: 6px; z-index: 2; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e1e6ec; padding: 8px; text-align: left; }
    th { background: #f0f3f6; }
    @media (max-width: 720px) {
      main { padding: 14px; }
      header, .grid, .task-meta { display: block; }
      .grid > div, .task-meta div { margin-bottom: 10px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Qualitative Mechanics Labeling</h1>
      <div class="muted">Local prototype</div>
    </div>
    <button id="refreshBtn" class="secondary" disabled>Refresh</button>
  </header>

  <section id="loginSection">
    <h2>Login</h2>
    <div class="grid">
      <div>
        <label for="name">Coach name</label>
        <input id="name" value="pilot_coach_1">
      </div>
      <div>
        <label for="password">Password</label>
        <input id="password" type="password" value="local-only-test-password">
      </div>
    </div>
    <div class="actions" style="margin-top: 12px;">
      <button id="loginBtn">Login</button>
      <span id="loginStatus" class="status"></span>
    </div>
  </section>

  <section id="taskSection" class="hidden">
    <h2>Current Task</h2>
    <div id="viewer"><div id="loading">Loading motion...</div></div>
    <div class="actions" style="margin: 12px 0;">
      <button type="button" class="secondary" id="playBtn">Play/Pause</button>
      <button type="button" class="secondary" data-view="home">Home</button>
      <button type="button" class="secondary" data-view="side">Open side</button>
      <button type="button" class="secondary" data-view="second">Second base</button>
      <button type="button" class="secondary" data-view="free">Free</button>
      <span class="muted" id="viewPill">view: home</span>
    </div>
    <div class="actions" style="margin-bottom: 12px;">
      <input id="frameSlider" type="range" min="0" max="0" value="0" style="flex: 1; min-width: 220px;">
      <span class="muted" id="frameText">0 / 0</span>
    </div>
    <div id="taskMeta" class="task-meta"></div>
    <form id="labelForm">
      <div id="labelFields" class="grid"></div>
      <div class="grid" style="margin-top: 12px;">
        <div>
          <label for="viewUsed">View used</label>
          <select id="viewUsed">
            <option value="side">side</option>
            <option value="second">second</option>
            <option value="home">home</option>
            <option value="free">free</option>
          </select>
        </div>
        <div>
          <label for="playbackSpeed">Playback speed</label>
          <select id="playbackSpeed">
            <option value="1">1</option>
            <option value="0.5">0.5</option>
            <option value="0.25">0.25</option>
          </select>
        </div>
      </div>
      <div style="margin-top: 12px;">
        <label for="notes">Notes</label>
        <textarea id="notes"></textarea>
      </div>
      <div class="actions" style="margin-top: 12px;">
        <button id="submitBtn" type="submit">Submit task</button>
        <span id="taskStatus" class="status"></span>
      </div>
    </form>
  </section>

  <section id="analysisSection" class="hidden">
    <h2>Agreement Gate</h2>
    <table>
      <thead>
        <tr>
          <th>Item</th>
          <th>Coaches</th>
          <th>Shared tasks</th>
          <th>Agreement</th>
          <th>Gate</th>
        </tr>
      </thead>
      <tbody id="analysisRows"></tbody>
    </table>
  </section>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const allowedValues = {
  hip_shoulder_separation: ["good", "average", "bad", "unclear"],
  lower_body_dominance: ["glute", "quad", "mixed", "unclear"],
  direction: ["good", "bad", "unclear"],
  shoulder_horizontal_abduction: ["good", "average", "bad", "unclear"],
  torso_velo_z: ["fast", "slow", "unclear"],
  hip_extension: ["good", "bad", "unclear"],
  heel_connection: ["connected", "early_extension", "unclear"],
  drift: ["good", "average", "bad", "unclear"]
};

let token = "";
let tasks = [];
let currentTask = null;
let scene, camera, renderer, controls, lines = [];
let motion = null;
let currentFrame = 0;
let playing = true;
let lastTime = 0;
let currentView = "home";
let threeReady = false;

function initThree() {
  if (threeReady) return;
  const el = document.getElementById("viewer");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111418);
  camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 1, 100000);
  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(renderer.domElement);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.GridHelper(6000, 20, 0x3a424d, 0x222830));
  window.addEventListener("resize", () => {
    camera.aspect = el.clientWidth / el.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(el.clientWidth, el.clientHeight);
  });
  threeReady = true;
  requestAnimationFrame(animate);
}

function clearLines() {
  for (const line of lines) scene.remove(line);
  lines = [];
}

function buildSkeleton() {
  clearLines();
  const mat = new THREE.LineBasicMaterial({color: 0xe8eaed, linewidth: 2});
  for (const pair of motion.connections) {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const line = new THREE.Line(geom, mat);
    line.userData = {a: pair[0], b: pair[1]};
    scene.add(line);
    lines.push(line);
  }
}

function updateSkeleton() {
  if (!motion) return;
  const frame = motion.frames[currentFrame] || {};
  for (const line of lines) {
    const a = frame[line.userData.a], b = frame[line.userData.b];
    line.visible = !!(a && b);
    if (line.visible) line.geometry.setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
  }
  document.getElementById("frameSlider").value = currentFrame;
  text(document.getElementById("frameText"), `${currentFrame} / ${motion.frames.length - 1}`);
}

function centerOfFrame() {
  const frame = motion?.frames?.[0] || {};
  const vals = Object.values(frame);
  const avg = vals.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0]);
  return vals.length ? new THREE.Vector3(avg[0] / vals.length, avg[1] / vals.length, avg[2] / vals.length) : new THREE.Vector3();
}

function setView(view) {
  currentView = view;
  text(document.getElementById("viewPill"), `view: ${view}`);
  if (!motion || !currentTask) return;
  const target = centerOfFrame();
  controls.target.copy(target);
  controls.enableRotate = view === "free";
  controls.enablePan = view === "free";
  if (view === "home") {
    camera.position.set(target.x + 8000, target.y + 1000, target.z);
  } else if (view === "side") {
    const lateral = currentTask.p_throws === "L" ? -5000 : 5000;
    camera.position.set(target.x, target.y + 1000, target.z + lateral);
  } else if (view === "second") {
    camera.position.set(target.x - 8000, target.y + 1000, target.z);
  }
  camera.lookAt(target);
  controls.update();
}

async function loadMotion(task) {
  initThree();
  const loading = document.getElementById("loading");
  loading.style.display = "block";
  text(loading, "Loading motion...");
  try {
    motion = await api(`/api/motion?session_pitch=${encodeURIComponent(task.session_pitch)}`);
    currentFrame = 0;
    document.getElementById("frameSlider").max = motion.frames.length - 1;
    buildSkeleton();
    updateSkeleton();
    setView(currentView);
    loading.style.display = "none";
  } catch (error) {
    motion = null;
    clearLines();
    text(loading, error.message);
  }
}

function animate(t) {
  requestAnimationFrame(animate);
  if (motion && playing && t - lastTime > (1000 / motion.fps)) {
    currentFrame = (currentFrame + 1) % motion.frames.length;
    updateSkeleton();
    lastTime = t;
  }
  if (controls) controls.update();
  if (renderer) renderer.render(scene, camera);
}

function text(el, value) { el.textContent = value; }
function show(el, visible) { el.classList.toggle("hidden", !visible); }

async function api(path, options = {}) {
  const headers = {"Accept": "application/json"};
  if (options.body) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const response = await fetch(path, {...options, headers});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function setStatus(id, message, ok = true) {
  const el = document.getElementById(id);
  text(el, message);
  el.className = `status ${ok ? "ok" : "error"}`;
}

function renderTask(task) {
  currentTask = task || null;
  const meta = document.getElementById("taskMeta");
  const fields = document.getElementById("labelFields");
  meta.replaceChildren();
  fields.replaceChildren();
  if (!task) {
    const done = document.createElement("div");
    text(done, "No pending tasks.");
    meta.appendChild(done);
    document.getElementById("submitBtn").disabled = true;
    return;
  }
  document.getElementById("submitBtn").disabled = false;
  [
    ["Order", task.display_order],
    ["Session pitch", task.session_pitch],
    ["Pitcher", task.pitcher_id],
    ["Throws", task.p_throws]
  ].forEach(([label, value]) => {
    const box = document.createElement("div");
    text(box, `${label}: ${value}`);
    meta.appendChild(box);
  });
  task.active_label_fields.forEach((field) => {
    const wrap = document.createElement("div");
    const label = document.createElement("label");
    label.setAttribute("for", `field_${field}`);
    text(label, field);
    const select = document.createElement("select");
    select.id = `field_${field}`;
    allowedValues[field].forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      text(option, value);
      select.appendChild(option);
    });
    wrap.appendChild(label);
    wrap.appendChild(select);
    fields.appendChild(wrap);
  });
  loadMotion(task);
}

async function loadPending() {
  const payload = await api("/api/pending");
  tasks = payload.tasks;
  renderTask(tasks[0]);
  setStatus("taskStatus", `${tasks.length} pending tasks.`);
}

async function loadAnalysis() {
  const payload = await api("/api/analysis");
  const body = document.getElementById("analysisRows");
  body.replaceChildren();
  Object.entries(payload.item_summaries).forEach(([item, summary]) => {
    const agreement = summary.agreement;
    const row = document.createElement("tr");
    [item, agreement.coach_count, agreement.shared_tasks, agreement.exact_agreement_rate ?? "n/a", agreement.pooled_analysis_gate_reason].forEach((value) => {
      const cell = document.createElement("td");
      text(cell, String(value));
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  show(document.getElementById("analysisSection"), true);
}

document.getElementById("loginBtn").addEventListener("click", async () => {
  try {
    const payload = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({
        name: document.getElementById("name").value,
        password: document.getElementById("password").value
      })
    });
    token = payload.token;
    setStatus("loginStatus", `Logged in as coach ${payload.coach_id}.`);
    show(document.getElementById("taskSection"), true);
    document.getElementById("refreshBtn").disabled = false;
    await loadPending();
    await loadAnalysis();
  } catch (error) {
    setStatus("loginStatus", error.message, false);
  }
});

document.getElementById("refreshBtn").addEventListener("click", async () => {
  try {
    await loadPending();
    await loadAnalysis();
  } catch (error) {
    setStatus("taskStatus", error.message, false);
  }
});

document.getElementById("playBtn").addEventListener("click", () => {
  playing = !playing;
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

document.getElementById("frameSlider").addEventListener("input", (event) => {
  if (!motion) return;
  currentFrame = Number(event.target.value);
  playing = false;
  updateSkeleton();
});

document.getElementById("labelForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentTask) return;
  const labels = {};
  currentTask.active_label_fields.forEach((field) => {
    labels[field] = document.getElementById(`field_${field}`).value;
  });
  try {
    const payload = await api("/api/labels", {
      method: "POST",
      body: JSON.stringify({
        session_pitch: currentTask.session_pitch,
        view_used: document.getElementById("viewUsed").value,
        playback_speed: document.getElementById("playbackSpeed").value,
        notes: document.getElementById("notes").value,
        labels
      })
    });
    setStatus("taskStatus", `Saved ${payload.inserted_labels} labels. Pending: ${payload.pending_after}.`);
    document.getElementById("notes").value = "";
    await loadPending();
    await loadAnalysis();
  } catch (error) {
    setStatus("taskStatus", error.message, false);
  }
});
</script>
</body>
</html>
"""


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, status: int, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
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


def load_web_motion_index() -> dict[str, str]:
    global WEB_MOTION_INDEX
    if WEB_MOTION_INDEX is not None:
        return WEB_MOTION_INDEX
    if not WEB_MOTION_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing web motion manifest: {WEB_MOTION_MANIFEST_PATH}")
    with WEB_MOTION_MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("web_motion_manifest.csv has no rows.")
    required = {"session_pitch", "motion_path"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise RuntimeError(f"web_motion_manifest.csv missing required columns: {sorted(missing)}")
    WEB_MOTION_INDEX = {row["session_pitch"]: row["motion_path"] for row in rows}
    return WEB_MOTION_INDEX


def load_static_motion(session_pitch: str) -> dict[str, Any]:
    motion_path = load_web_motion_index().get(session_pitch)
    if motion_path is None:
        raise FileNotFoundError(f"No static motion JSON registered for {session_pitch}.")
    full_path = EXPERIMENT_DIR / motion_path
    if not full_path.exists():
        raise FileNotFoundError(f"Missing static motion JSON for {session_pitch}: {full_path}")
    with full_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "frames" not in payload or "connections" not in payload:
        raise RuntimeError(f"Invalid static motion JSON for {session_pitch}: {full_path}")
    return payload


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
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html_response(self, 200, INDEX_HTML)
                return
            if parsed.path == "/api/pending":
                coach_id = require_session(self)
                with local_label_db.connect() as conn:
                    rows = local_label_db.pending_tasks(conn, coach_id)
                json_response(self, 200, {"tasks": [row_to_task(row) for row in rows]})
                return
            if parsed.path == "/api/analysis":
                require_session(self)
                json_response(self, 200, analyze_label_db.build_summary())
                return
            if parsed.path == "/api/motion":
                require_session(self)
                qs = parse_qs(parsed.query)
                session_pitch = qs.get("session_pitch", [""])[0]
                with local_label_db.connect() as conn:
                    row = conn.execute(
                        "SELECT session_pitch FROM label_tasks WHERE session_pitch = ? AND active = 1",
                        (session_pitch,),
                    ).fetchone()
                if row is None:
                    json_response(self, 404, {"error": f"No active task for {session_pitch}."})
                    return
                json_response(self, 200, load_static_motion(session_pitch))
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


def request_text(url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"Accept": "text/html"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def smoke_test() -> None:
    local_label_db.init_db()
    SESSIONS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabelApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, html = request_text(f"{base_url}/")
        if status != 200 or "Qualitative Mechanics Labeling" not in html or "/api/login" not in html:
            raise RuntimeError(f"UI page check failed: status={status}")
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
        print(f"ui_status=200")
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
    print(f"Serving qualitative label API on http://{host}:{server.server_port}", flush=True)
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
