#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local qualitative pitching-mechanics labeling experiment.

This tool serves a browser UI for labeling C3D skeleton motion clips. It writes
only experiment artifacts in this folder: manifest.csv and labels.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT / "baseball_pitching"
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.csv"
LABELS_PATH = EXPERIMENT_DIR / "labels.csv"

SKELETON_CONNECTIONS = [
    ("C7", "CLAV"), ("CLAV", "STRN"), ("STRN", "T10"),
    ("T10", "LASI"), ("T10", "RASI"), ("LASI", "RASI"),
    ("LASI", "LPSI"), ("RASI", "RPSI"), ("LPSI", "RPSI"),
    ("C7", "RBAK"), ("LFHD", "RFHD"), ("LBHD", "RBHD"),
    ("LFHD", "LBHD"), ("RFHD", "RBHD"),
    ("CLAV", "RSHO"), ("RSHO", "RUPA"), ("RUPA", "RELB"),
    ("RELB", "RMELB"), ("RMELB", "RFRM"), ("RFRM", "RWRA"),
    ("RFRM", "RWRB"), ("RWRA", "RWRB"), ("RWRA", "RFIN"),
    ("CLAV", "LSHO"), ("LSHO", "LUPA"), ("LUPA", "LELB"),
    ("LELB", "LMELB"), ("LMELB", "LFRM"), ("LFRM", "LWRA"),
    ("LFRM", "LWRB"), ("LWRA", "LWRB"), ("LWRA", "LFIN"),
    ("RASI", "RTHI"), ("RTHI", "RKNE"), ("RKNE", "RMKNE"),
    ("RMKNE", "RTIB"), ("RTIB", "RANK"), ("RANK", "RMANK"),
    ("RMANK", "RHEE"), ("RANK", "RTOE"), ("RHEE", "RTOE"),
    ("LASI", "LTHI"), ("LTHI", "LKNE"), ("LKNE", "LMKNE"),
    ("LMKNE", "LTIB"), ("LTIB", "LANK"), ("LANK", "LMANK"),
    ("LMANK", "LHEE"), ("LANK", "LTOE"), ("LHEE", "LTOE"),
]

LABEL_FIELDS = [
    "hip_shoulder_separation",
    "lower_body_dominance",
    "direction",
    "shoulder_horizontal_abduction",
    "heel_connection",
    "drift",
]

LABEL_COLUMNS = [
    "saved_at_utc",
    "rater_id",
    "session_pitch",
    "pitcher_id",
    "p_throws",
    "order",
    "view_used",
    "playback_speed",
    *LABEL_FIELDS,
    "skipped",
    "skip_reason",
    "notes",
]

_CFG: dict[str, object] = {}


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_manifest(rows: list[dict[str, object]]) -> None:
    fields = ["order", "session_pitch", "pitcher_id", "p_throws", "filename_new", "c3d_path"]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_c3d_path(row: pd.Series) -> Path:
    pitcher_id = f"{int(row['user']):06d}"
    return DATA_ROOT / "c3d" / pitcher_id / str(row["filename_new"])


def ensure_manifest(seed: int, pitchers: int, pitches_per_pitcher: int, rebuild: bool) -> list[dict[str, str]]:
    if MANIFEST_PATH.exists() and not rebuild:
        return _read_csv_records(MANIFEST_PATH)

    meta_path = DATA_ROOT / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    meta = pd.read_csv(meta_path)
    required = {"session_pitch", "user", "filename_new"}
    missing = required.difference(meta.columns)
    if missing:
        raise RuntimeError(f"metadata.csv missing required columns: {sorted(missing)}")
    if "p_throws" not in meta.columns:
        poi_path = DATA_ROOT / "poi" / "poi_metrics.csv"
        if not poi_path.exists():
            raise FileNotFoundError(f"Missing POI metrics for p_throws: {poi_path}")
        poi = pd.read_csv(poi_path, usecols=["session_pitch", "p_throws"])
        meta = meta.merge(poi.drop_duplicates("session_pitch"), on="session_pitch", how="left")
        if meta["p_throws"].isna().any():
            missing_throws = meta.loc[meta["p_throws"].isna(), "session_pitch"].head(10).tolist()
            raise RuntimeError(f"Missing p_throws after POI merge for session_pitch values: {missing_throws}")

    rows: list[dict[str, object]] = []
    rng = random.Random(seed)
    grouped = list(meta.groupby("user", sort=False))
    rng.shuffle(grouped)

    for user, group in grouped[:pitchers]:
        candidates = []
        for _, row in group.iterrows():
            c3d_path = build_c3d_path(row)
            if c3d_path.exists():
                candidates.append((row, c3d_path))
        rng.shuffle(candidates)
        for row, c3d_path in candidates[:pitches_per_pitcher]:
            rows.append({
                "session_pitch": str(row["session_pitch"]),
                "pitcher_id": f"{int(user):06d}",
                "p_throws": str(row["p_throws"]),
                "filename_new": str(row["filename_new"]),
                "c3d_path": str(c3d_path),
            })

    if not rows:
        raise RuntimeError("No displayable C3D files found for manifest generation.")

    rng.shuffle(rows)
    for idx, row in enumerate(rows, 1):
        row["order"] = idx
    _write_manifest(rows)
    return _read_csv_records(MANIFEST_PATH)


def ensure_labels_file() -> None:
    if LABELS_PATH.exists():
        return
    with LABELS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()


def load_motion_data(c3d_path: Path, frame_step: int) -> dict[str, object]:
    import ezc3d

    c3d = ezc3d.c3d(str(c3d_path))
    pts = c3d["data"]["points"]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    marker_labels = [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in labels]
    indices = {name: marker_labels.index(name) for pair in SKELETON_CONNECTIONS for name in pair if name in marker_labels}
    frames = []

    for frame in range(0, pts.shape[2], frame_step):
        frame_data = {}
        for label, idx in indices.items():
            val = pts[:3, idx, frame]
            if not any(pd.isna(val)) and not any(val == 0):
                frame_data[label] = [float(val[0]), float(val[2]), -float(val[1])]
        frames.append(frame_data)

    if not frames:
        raise RuntimeError("No valid marker frames found in C3D.")

    return {
        "fps": float(c3d["header"]["points"]["frame_rate"]) / frame_step,
        "connections": SKELETON_CONNECTIONS,
        "frames": frames,
    }


def html_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qualitative Mechanics Experiment</title>
<style>
body { margin:0; background:#111418; color:#e8eaed; font-family:Segoe UI, Arial, sans-serif; }
.app { display:grid; grid-template-columns:minmax(0,1fr) 360px; height:100vh; }
#viewer { position:relative; min-width:0; }
.side { border-left:1px solid #2c3138; padding:16px; overflow:auto; background:#171b21; }
.row { display:flex; gap:8px; align-items:center; margin:8px 0; flex-wrap:wrap; }
button, select, input, textarea { background:#242a33; color:#f2f4f8; border:1px solid #3a424d; border-radius:6px; padding:8px; }
button { cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button.warn { background:#6e2c2c; border-color:#a54242; }
label { display:block; margin-top:12px; font-size:13px; color:#bdc7d5; }
select, input, textarea { width:100%; box-sizing:border-box; }
textarea { min-height:64px; resize:vertical; }
.status { color:#9fb0c4; font-size:13px; line-height:1.4; }
.pill { padding:3px 8px; border-radius:999px; background:#2d333b; font-size:12px; }
#loading { position:absolute; top:16px; left:16px; background:#171b21; padding:8px 10px; border:1px solid #3a424d; border-radius:6px; }
</style>
</head>
<body>
<div class="app">
  <div id="viewer"><div id="loading">Loading...</div></div>
  <div class="side">
    <h2>Qualitative Mechanics</h2>
    <div class="status" id="itemStatus"></div>
    <label>Rater ID<input id="raterId" placeholder="required"></label>
    <div class="row">
      <button onclick="prevItem()">Prev</button>
      <button onclick="nextItem()">Next</button>
      <button onclick="setView('home')">Home</button>
      <button onclick="setView('side')">Open side</button>
      <button onclick="setView('free')">Free</button>
    </div>
    <div class="row">
      <button onclick="togglePlay()">Play/Pause</button>
      <span class="pill" id="viewPill">view: home</span>
      <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
    </div>
    <form id="labelForm">
      <label>Hip-shoulder separation<select name="hip_shoulder_separation"><option>unclear</option><option>present</option><option>absent</option></select></label>
      <label>Glute / quad dominance<select name="lower_body_dominance"><option>unclear</option><option>glute</option><option>quad</option><option>mixed</option></select></label>
      <label>Direction<select name="direction"><option>unclear</option><option>stride</option><option>hip_extension</option></select></label>
      <label>Shoulder horizontal abduction<select name="shoulder_horizontal_abduction"><option>unclear</option><option>early</option><option>neutral</option><option>excessive</option></select></label>
      <label>Heel connection<select name="heel_connection"><option>unclear</option><option>connected</option><option>early_extension</option></select></label>
      <label>Drift<select name="drift"><option>unclear</option><option>present</option><option>absent</option></select></label>
      <label>Skip reason<input name="skip_reason" placeholder="only when skipped"></label>
      <label>Notes<textarea name="notes"></textarea></label>
    </form>
    <div class="row">
      <button class="primary" onclick="saveLabel(false)">Save label</button>
      <button class="warn" onclick="saveLabel(true)">Skip / bad display</button>
    </div>
    <div class="status" id="saveStatus"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
let manifest = [];
let idx = 0;
let motion = null;
let scene, camera, renderer, controls, lines = [];
let currentFrame = 0, playing = true, lastTime = 0, currentView = 'home';

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function initThree() {
  const el = document.getElementById('viewer');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f141a);
  camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 1, 100000);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(renderer.domElement);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.GridHelper(6000, 20, 0x3a424d, 0x222830));
  window.addEventListener('resize', () => {
    camera.aspect = el.clientWidth / el.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(el.clientWidth, el.clientHeight);
  });
}

function clearLines() {
  for (const line of lines) scene.remove(line);
  lines = [];
}

function buildSkeleton() {
  clearLines();
  const mat = new THREE.LineBasicMaterial({color:0xe8eaed, linewidth:2});
  for (const pair of motion.connections) {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const line = new THREE.Line(geom, mat);
    line.userData = {a:pair[0], b:pair[1]};
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
    if (line.visible) {
      line.geometry.setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
    }
  }
}

function centerOfFrame() {
  const frame = motion.frames[0] || {};
  const vals = Object.values(frame);
  const avg = vals.reduce((acc, p) => [acc[0]+p[0], acc[1]+p[1], acc[2]+p[2]], [0,0,0]);
  return vals.length ? new THREE.Vector3(avg[0]/vals.length, avg[1]/vals.length, avg[2]/vals.length) : new THREE.Vector3();
}

function setView(view) {
  currentView = view;
  document.getElementById('viewPill').textContent = 'view: ' + view;
  if (!motion) return;
  const target = centerOfFrame();
  controls.target.copy(target);
  controls.enableRotate = view === 'free';
  controls.enablePan = view === 'free';
  const item = manifest[idx] || {};
  if (view === 'home') {
    camera.position.set(target.x + 8000, target.y + 1000, target.z);
  } else if (view === 'side') {
    const lateral = item.p_throws === 'L' ? -5000 : 5000;
    camera.position.set(target.x, target.y + 1000, target.z + lateral);
  }
  camera.lookAt(target);
  controls.update();
}

async function loadItem() {
  const item = manifest[idx];
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = 'Loading ' + item.session_pitch + '...';
  document.getElementById('itemStatus').textContent =
    `Item ${idx + 1}/${manifest.length} | pitch ${item.session_pitch} | pitcher ${item.pitcher_id} | throws ${item.p_throws}`;
  try {
    motion = await api('/api/motion?session_pitch=' + encodeURIComponent(item.session_pitch));
    currentFrame = 0;
    buildSkeleton();
    updateSkeleton();
    setView(currentView);
    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    motion = null;
    clearLines();
    document.getElementById('loading').textContent = err.message;
  }
}

function nextItem() { if (idx < manifest.length - 1) { idx++; loadItem(); } }
function prevItem() { if (idx > 0) { idx--; loadItem(); } }
function togglePlay() { playing = !playing; }

async function saveLabel(skipped) {
  const rater = document.getElementById('raterId').value.trim();
  if (!rater) { document.getElementById('saveStatus').textContent = 'Rater ID is required.'; return; }
  const form = new FormData(document.getElementById('labelForm'));
  const payload = Object.fromEntries(form.entries());
  Object.assign(payload, {
    rater_id: rater,
    session_pitch: manifest[idx].session_pitch,
    view_used: currentView,
    playback_speed: document.getElementById('speed').value,
    skipped: skipped ? 'true' : 'false',
  });
  try {
    await api('/api/label', {method:'POST', body:JSON.stringify(payload)});
    document.getElementById('saveStatus').textContent = 'Saved.';
    if (idx < manifest.length - 1) nextItem();
  } catch (err) {
    document.getElementById('saveStatus').textContent = err.message;
  }
}

function animate(t) {
  requestAnimationFrame(animate);
  if (motion && playing && t - lastTime > (1000 / motion.fps) / parseFloat(document.getElementById('speed').value)) {
    currentFrame = (currentFrame + 1) % motion.frames.length;
    updateSkeleton();
    lastTime = t;
  }
  if (controls) controls.update();
  if (renderer) renderer.render(scene, camera);
}

async function main() {
  initThree();
  manifest = await api('/api/manifest');
  await loadItem();
  animate(0);
}
main().catch(err => document.getElementById('loading').textContent = err.message);
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(fmt % args + "\n")

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = html_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/manifest":
            self._json(_CFG["manifest"])
            return
        if parsed.path == "/api/motion":
            qs = parse_qs(parsed.query)
            session_pitch = qs.get("session_pitch", [""])[0]
            item = _CFG["c3d_map"].get(session_pitch)  # type: ignore[index]
            if not item:
                self._json({"error": f"No manifest item for {session_pitch}"}, 404)
                return
            try:
                self._json(load_motion_data(Path(item["c3d_path"]), int(_CFG["frame_step"])))  # type: ignore[index]
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            item = _CFG["c3d_map"].get(str(payload.get("session_pitch", "")))  # type: ignore[index]
            if not item:
                self._json({"error": "Label references a pitch outside the manifest."}, 400)
                return
            row = {col: "" for col in LABEL_COLUMNS}
            row.update({
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "pitcher_id": item["pitcher_id"],
                "p_throws": item["p_throws"],
                "order": item["order"],
            })
            for col in LABEL_COLUMNS:
                if col in payload:
                    row[col] = str(payload[col])
            if row["skipped"] == "true" and not row["skip_reason"].strip():
                self._json({"error": "Skip reason is required for skipped items."}, 400)
                return
            with LABELS_PATH.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
                writer.writerow(row)
            self._json({"ok": True})
        except Exception as exc:
            self._json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the qualitative mechanics labeling experiment.")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--pitchers", type=int, default=20)
    parser.add_argument("--pitches-per-pitcher", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--check-first-load", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest(args.seed, args.pitchers, args.pitches_per_pitcher, args.rebuild_manifest)
    ensure_labels_file()

    _CFG["manifest"] = manifest
    _CFG["c3d_map"] = {row["session_pitch"]: row for row in manifest}
    _CFG["frame_step"] = args.frame_step

    if args.check_first_load:
        first = manifest[0]
        motion = load_motion_data(Path(first["c3d_path"]), args.frame_step)
        print(
            "Loaded first manifest pitch:",
            first["session_pitch"],
            f"frames={len(motion['frames'])}",
            f"fps={motion['fps']}",
        )
        return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Qualitative Mechanics Experiment -> {url}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Labels:   {LABELS_PATH}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
