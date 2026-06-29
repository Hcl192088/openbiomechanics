const ACTIVE_LABEL_FIELDS = [
  "hip_shoulder_separation",
  "lower_body_dominance",
  "direction",
  "shoulder_horizontal_abduction",
  "torso_velo_z",
  "hip_extension",
  "heel_connection",
  "drift",
];

const FIELD_ALLOWED_VALUES = {
  hip_shoulder_separation: new Set(["good", "average", "bad", "unclear"]),
  lower_body_dominance: new Set(["glute", "quad", "mixed", "unclear"]),
  direction: new Set(["good", "bad", "unclear"]),
  shoulder_horizontal_abduction: new Set(["good", "average", "bad", "unclear"]),
  torso_velo_z: new Set(["fast", "slow", "unclear"]),
  hip_extension: new Set(["good", "bad", "unclear"]),
  heel_connection: new Set(["connected", "early_extension", "unclear"]),
  drift: new Set(["good", "average", "bad", "unclear"]),
};

const AGREEMENT_THRESHOLD = 0.70;
const MIN_SHARED_TASKS = 5;
const MIN_COACHES = 2;
const MAX_WORKERS_PBKDF2_ITERATIONS = 100000;
const MAX_JSON_BODY_BYTES = 16 * 1024;
const MAX_NAME_LENGTH = 64;
const MAX_PASSWORD_LENGTH = 128;
const MIN_PASSWORD_LENGTH = 4;
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      if (url.pathname === "/api/login" && request.method === "POST") {
        return json(await login(request, env));
      }
      if (url.pathname === "/api/password" && request.method === "POST") {
        const coachId = await requireSession(request, env);
        return json(await changePassword(request, env, coachId));
      }
      if (url.pathname === "/api/pending" && request.method === "GET") {
        const coachId = await requireSession(request, env);
        return json({ tasks: await pendingTasks(env, coachId) });
      }
      if (url.pathname === "/api/labels" && request.method === "POST") {
        const coachId = await requireSession(request, env);
        return json(await saveLabels(request, env, coachId));
      }
      if (url.pathname === "/api/analysis" && request.method === "GET") {
        const coachId = await requireSession(request, env);
        return json(await analysis(env, coachId));
      }
      if (url.pathname === "/api/motion" && request.method === "GET") {
        await requireSession(request, env);
        return motion(request, env);
      }
      if (url.pathname.startsWith("/api/")) {
        return json({ error: "Not found." }, 404);
      }
      return env.ASSETS.fetch(request);
    } catch (error) {
      const status = error.status || 400;
      return json({ error: error.message || String(error) }, status);
    }
  },
};

async function login(request, env) {
  const payload = await readObject(request);
  const name = String(payload.name || "").trim();
  const password = String(payload.password || "");
  validateCoachName(name);
  validatePasswordInput(password, "password");
  const coach = await env.DB.prepare(
    "SELECT id, name, password_hash, must_change_password FROM coaches WHERE name = ?"
  ).bind(name).first();
  if (!coach || !(await verifyPassword(password, coach.password_hash))) {
    throw httpError("Invalid login.", 401);
  }
  const token = crypto.randomUUID() + "." + crypto.randomUUID();
  const createdAt = new Date().toISOString();
  const expiresAt = new Date(Date.now() + SESSION_TTL_MS).toISOString();
  await env.DB.prepare(
    "INSERT INTO sessions (token, coach_id, created_at, expires_at) VALUES (?, ?, ?, ?)"
  ).bind(token, coach.id, createdAt, expiresAt).run();
  return {
    token,
    coach_id: coach.id,
    coach_name: coach.name,
    must_change_password: Boolean(coach.must_change_password),
  };
}

async function requireSession(request, env) {
  const header = request.headers.get("Authorization") || "";
  const prefix = "Bearer ";
  if (!header.startsWith(prefix)) throw httpError("Missing bearer token.", 401);
  const token = header.slice(prefix.length).trim();
  const row = await env.DB.prepare(
    "SELECT coach_id, expires_at FROM sessions WHERE token = ?"
  ).bind(token).first();
  if (!row) throw httpError("Invalid bearer token.", 401);
  if (Date.parse(row.expires_at) <= Date.now()) {
    await env.DB.prepare("DELETE FROM sessions WHERE token = ?").bind(token).run();
    throw httpError("Expired bearer token.", 401);
  }
  return row.coach_id;
}

async function changePassword(request, env, coachId) {
  const payload = await readObject(request);
  const currentPassword = String(payload.current_password || "");
  const newPassword = String(payload.new_password || "");
  validatePasswordInput(currentPassword, "current_password");
  validatePasswordInput(newPassword, "new_password");
  if (newPassword === "0000") throw httpError("new_password cannot remain 0000.", 400);
  const coach = await env.DB.prepare(
    "SELECT password_hash FROM coaches WHERE id = ?"
  ).bind(coachId).first();
  if (!coach || !(await verifyPassword(currentPassword, coach.password_hash))) {
    throw httpError("Invalid current password.", 401);
  }
  const salt = crypto.randomUUID().replaceAll("-", "");
  const passwordHash = await hashPassword(newPassword, salt);
  await env.DB.prepare(
    "UPDATE coaches SET password_hash = ?, must_change_password = 0 WHERE id = ?"
  ).bind(passwordHash, coachId).run();
  return { ok: true };
}

async function pendingTasks(env, coachId) {
  const rows = await env.DB.prepare(
    `SELECT t.*
     FROM label_tasks t
     WHERE t.active = 1
       AND (
         SELECT COUNT(DISTINCT l.item_name)
         FROM labels l
         WHERE l.coach_id = ?
           AND l.session_pitch = t.session_pitch
       ) < ?
     ORDER BY t.display_order`
  ).bind(coachId, ACTIVE_LABEL_FIELDS.length).all();
  return rows.results.map((row) => ({
    task_id: row.id,
    display_order: row.display_order,
    session_pitch: row.session_pitch,
    p_throws: row.p_throws,
    active_label_fields: String(row.active_label_fields).split(";"),
  }));
}

async function saveLabels(request, env, coachId) {
  const payload = await readObject(request);
  const sessionPitch = String(payload.session_pitch || "").trim();
  const playbackSpeed = String(payload.playback_speed || "").trim();
  const labels = payload.labels;
  if (!sessionPitch) throw httpError("session_pitch is required.", 400);
  if (!playbackSpeed) throw httpError("playback_speed is required.", 400);
  if (!labels || typeof labels !== "object" || Array.isArray(labels)) {
    throw httpError("labels must be an object.", 400);
  }
  const provided = Object.keys(labels).sort();
  const expected = [...ACTIVE_LABEL_FIELDS].sort();
  if (provided.join("\n") !== expected.join("\n")) {
    throw httpError("labels must contain exactly the active fields.", 400);
  }
  const task = await env.DB.prepare(
    "SELECT session_pitch FROM label_tasks WHERE session_pitch = ? AND active = 1"
  ).bind(sessionPitch).first();
  if (!task) throw httpError(`No active task for session_pitch=${sessionPitch}.`, 404);
  for (const itemName of ACTIVE_LABEL_FIELDS) {
    const value = String(labels[itemName] || "").trim();
    if (!FIELD_ALLOWED_VALUES[itemName].has(value)) {
      throw httpError(`Invalid label value for ${itemName}: ${value}`, 400);
    }
  }
  const beforePending = (await pendingTasks(env, coachId)).length;
  const createdAt = new Date().toISOString();
  const statements = ACTIVE_LABEL_FIELDS.map((itemName) =>
    env.DB.prepare(
      `INSERT INTO labels (
        coach_id, session_pitch, item_name, label_value, view_used,
        playback_speed, skipped, skip_reason, notes, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, 0, '', ?, ?)`
    ).bind(
      coachId,
      sessionPitch,
      itemName,
      String(labels[itemName]).trim(),
      "",
      playbackSpeed,
      "",
      createdAt
    )
  );
  try {
    await env.DB.batch(statements);
  } catch (error) {
    throw httpError(`Duplicate or invalid label insert: ${error.message}`, 409);
  }
  const afterPending = (await pendingTasks(env, coachId)).length;
  return {
    ok: true,
    session_pitch: sessionPitch,
    inserted_labels: ACTIVE_LABEL_FIELDS.length,
    pending_before: beforePending,
    pending_after: afterPending,
  };
}

async function motion(request, env) {
  const url = new URL(request.url);
  const sessionPitch = url.searchParams.get("session_pitch") || "";
  const task = await env.DB.prepare(
    "SELECT session_pitch FROM label_tasks WHERE session_pitch = ? AND active = 1"
  ).bind(sessionPitch).first();
  if (!task) return json({ error: `No active task for ${sessionPitch}.` }, 404);
  const assetUrl = new URL(`/web_motion/${sessionPitch}.json`, request.url);
  const assetRequest = new Request(assetUrl.toString(), request);
  const response = await env.ASSETS.fetch(assetRequest);
  if (!response.ok) return json({ error: `Missing static motion JSON for ${sessionPitch}.` }, 404);
  return new Response(response.body, {
    status: response.status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

async function analysis(env, coachId) {
  const rows = await env.DB.prepare(
    "SELECT coach_id, session_pitch, item_name, label_value, skipped FROM labels ORDER BY item_name, session_pitch, coach_id"
  ).all();
  if (!rows.results.length) throw httpError("No labels in database.", 400);
  const byItem = new Map();
  for (const row of rows.results) {
    if (!byItem.has(row.item_name)) byItem.set(row.item_name, []);
    byItem.get(row.item_name).push(row);
  }
  const itemSummaries = {};
  for (const [itemName, itemRows] of [...byItem.entries()].sort()) {
    itemSummaries[itemName] = {
      agreement: itemAgreement(itemRows),
      pooled_metric_summary: null,
    };
  }
  return {
    dashboard: await dashboardStats(env, coachId),
    agreement_threshold: AGREEMENT_THRESHOLD,
    min_shared_tasks: MIN_SHARED_TASKS,
    min_coaches: MIN_COACHES,
    metric_columns: [],
    item_summaries: itemSummaries,
  };
}

async function dashboardStats(env, coachId) {
  const totalTasks = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM label_tasks WHERE active = 1"
  ).first();
  const myCompleted = await env.DB.prepare(
    `SELECT COUNT(*) AS n
     FROM (
       SELECT session_pitch
       FROM labels
       WHERE coach_id = ?
       GROUP BY session_pitch
       HAVING COUNT(DISTINCT item_name) = ?
     )`
  ).bind(coachId, ACTIVE_LABEL_FIELDS.length).first();
  const totalLabels = await env.DB.prepare("SELECT COUNT(*) AS n FROM labels").first();
  const coachCount = await env.DB.prepare("SELECT COUNT(*) AS n FROM coaches").first();
  return {
    total_tasks: Number(totalTasks.n),
    my_completed_tasks: Number(myCompleted.n),
    my_pending_tasks: Number(totalTasks.n) - Number(myCompleted.n),
    total_labels: Number(totalLabels.n),
    coach_count: Number(coachCount.n),
  };
}

function itemAgreement(rows) {
  const coaches = new Set(rows.map((row) => row.coach_id));
  const unclearCount = rows.filter((row) => row.label_value === "unclear").length;
  const skippedCount = rows.filter((row) => Number(row.skipped)).length;
  const byPitch = new Map();
  for (const row of rows) {
    if (Number(row.skipped)) continue;
    if (!byPitch.has(row.session_pitch)) byPitch.set(row.session_pitch, new Map());
    byPitch.get(row.session_pitch).set(row.coach_id, row.label_value);
  }
  let comparedPairs = 0;
  let exactMatches = 0;
  let sharedTasks = 0;
  for (const coachLabels of byPitch.values()) {
    const pitchCoaches = [...coachLabels.keys()].sort();
    if (pitchCoaches.length < 2) continue;
    sharedTasks += 1;
    for (let i = 0; i < pitchCoaches.length; i += 1) {
      for (let j = i + 1; j < pitchCoaches.length; j += 1) {
        comparedPairs += 1;
        if (coachLabels.get(pitchCoaches[i]) === coachLabels.get(pitchCoaches[j])) {
          exactMatches += 1;
        }
      }
    }
  }
  const exactAgreementRate = comparedPairs === 0 ? null : exactMatches / comparedPairs;
  let gateReason = "pass";
  if (coaches.size < MIN_COACHES) gateReason = "fewer_than_two_coaches";
  else if (sharedTasks < MIN_SHARED_TASKS) gateReason = "not_enough_shared_tasks";
  else if (exactAgreementRate === null || exactAgreementRate < AGREEMENT_THRESHOLD) {
    gateReason = "below_agreement_threshold";
  }
  return {
    coach_count: coaches.size,
    shared_tasks: sharedTasks,
    compared_pairs: comparedPairs,
    exact_matches: exactMatches,
    exact_agreement_rate: exactAgreementRate,
    unclear_count: unclearCount,
    skipped_count: skippedCount,
    pooled_analysis_enabled: gateReason === "pass",
    pooled_analysis_gate_reason: gateReason,
  };
}

async function readObject(request) {
  const length = Number(request.headers.get("Content-Length") || "0");
  if (length > MAX_JSON_BODY_BYTES) {
    throw httpError(`JSON body must be ${MAX_JSON_BODY_BYTES} bytes or fewer.`, 413);
  }
  let payload;
  try {
    payload = await request.json();
  } catch {
    throw httpError("Invalid JSON body.", 400);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw httpError("JSON body must be an object.", 400);
  }
  return payload;
}

function validateCoachName(name) {
  if (!name || name.length > MAX_NAME_LENGTH || !/^[A-Za-z0-9_-]+$/.test(name)) {
    throw httpError("name must be 1-64 characters and contain only letters, numbers, underscores, or hyphens.", 400);
  }
}

function validatePasswordInput(password, fieldName) {
  if (password.length < MIN_PASSWORD_LENGTH || password.length > MAX_PASSWORD_LENGTH) {
    throw httpError(`${fieldName} must be ${MIN_PASSWORD_LENGTH}-${MAX_PASSWORD_LENGTH} characters.`, 400);
  }
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function httpError(message, status) {
  const error = new Error(message);
  error.status = status;
  return error;
}

async function verifyPassword(password, storedHash) {
  const [algorithm, iterationsRaw, salt, expectedHex] = String(storedHash).split("$");
  if (algorithm !== "pbkdf2_sha256") throw httpError(`Unsupported password hash algorithm: ${algorithm}`, 400);
  const iterations = Number(iterationsRaw);
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > MAX_WORKERS_PBKDF2_ITERATIONS) {
    throw httpError(`Unsupported PBKDF2 iteration count: ${iterationsRaw}`, 400);
  }
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveBits"]
  );
  const bits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      hash: "SHA-256",
      salt: new TextEncoder().encode(salt),
      iterations,
    },
    keyMaterial,
    256
  );
  return timingSafeEqual(hex(new Uint8Array(bits)), expectedHex);
}

async function hashPassword(password, salt) {
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveBits"]
  );
  const bits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      hash: "SHA-256",
      salt: new TextEncoder().encode(salt),
      iterations: MAX_WORKERS_PBKDF2_ITERATIONS,
    },
    keyMaterial,
    256
  );
  return `pbkdf2_sha256$${MAX_WORKERS_PBKDF2_ITERATIONS}$${salt}$${hex(new Uint8Array(bits))}`;
}

function hex(bytes) {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i += 1) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}
