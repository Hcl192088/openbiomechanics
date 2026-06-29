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

const FIELD_VALUE_ORDER = {
  hip_shoulder_separation: ["good", "average", "bad", "unclear"],
  lower_body_dominance: ["glute", "mixed", "quad", "unclear"],
  direction: ["good", "bad", "unclear"],
  shoulder_horizontal_abduction: ["good", "average", "bad", "unclear"],
  torso_velo_z: ["fast", "slow", "unclear"],
  hip_extension: ["good", "bad", "unclear"],
  heel_connection: ["connected", "early_extension", "unclear"],
  drift: ["good", "average", "bad", "unclear"],
};

const PILOT_FIELD_METRICS = {
  hip_shoulder_separation: [
    "pitch_speed_mph",
    "max_rotation_hip_shoulder_separation",
    "rotation_hip_shoulder_separation_fp",
  ],
  shoulder_horizontal_abduction: [
    "pitch_speed_mph",
    "shoulder_horizontal_abduction_fp",
    "max_shoulder_horizontal_abduction",
  ],
  torso_velo_z: ["pitch_speed_mph", "max_torso_rotational_velo"],
  hip_extension: [
    "pitch_speed_mph",
    "pelvis_rotation_fp",
    "rotation_hip_shoulder_separation_fp",
    "max_rotation_hip_shoulder_separation",
    "max_torso_rotational_velo",
    "cog_velo_pkh",
    "stride_length",
    "stride_angle",
    "max_rear_hip_flexion",
    "max_rear_hip_internal_rotation_velo",
    "rear_hip_transfer_pkh_fp",
    "rear_hip_generation_pkh_fp",
    "rear_hip_absorption_pkh_fp",
    "lead_hip_transfer_fp_br",
    "lead_hip_generation_fp_br",
    "lead_hip_absorption_fp_br",
    "lead_knee_extension_from_fp_to_br",
    "lead_knee_extension_angular_velo_fp",
    "lead_grf_x_max",
    "lead_grf_y_max",
    "lead_grf_z_max",
    "rear_grf_x_max",
    "rear_grf_y_max",
    "rear_grf_z_max",
  ],
  direction: ["pitch_speed_mph", "stride_length", "stride_angle", "max_cog_velo_x"],
  heel_connection: [
    "pitch_speed_mph",
    "lead_knee_extension_from_fp_to_br",
    "lead_knee_extension_angular_velo_fp",
    "lead_grf_z_max",
  ],
  drift: ["pitch_speed_mph", "cog_velo_pkh", "max_cog_velo_x", "stride_angle"],
};

const PILOT_INTERPRETATION = {
  hip_shoulder_separation: "Highest-priority validation candidate: visual groups align with direct hip-shoulder separation POI metrics and pitch speed.",
  shoulder_horizontal_abduction: "Promising but sample-limited: direct shoulder horizontal abduction metrics move in the expected direction, but the bad group is small.",
  torso_velo_z: "Visual fast/slow maps better to torso rotational velocity than to pitch speed.",
  hip_extension: "Pilot-positive but indirect: good/bad groups separate on speed and several transfer/lead-leg metrics, but POI still lacks direct FP hip extension angles.",
  direction: "Current good/bad rubric likely mixes open stride and cross-fire into one bad group; angle-based categories should be split before strong interpretation.",
  heel_connection: "Contested label: pitch-speed separation should not be treated as a clean mechanism until the rubric is tightened and matched to POI metrics.",
  drift: "More consistent with center-of-mass velocity at PKH than with max COM velocity or pitch speed.",
};

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
      if (url.pathname === "/dashboard") {
        return env.ASSETS.fetch(new Request(new URL("/dashboard.html", request.url), request));
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
  return {
    dashboard: await dashboardStats(env, coachId),
    metric_columns: [],
    my_stats: await pilotStats(env, coachId),
    all_stats: await pilotStats(env),
  };
}

async function pilotStats(env, coachId = null) {
  const out = [];
  for (const [field, metrics] of Object.entries(PILOT_FIELD_METRICS)) {
    const fieldOut = {
      field,
      interpretation: PILOT_INTERPRETATION[field] || "",
      metrics: [],
    };
    for (const metric of metrics) fieldOut.metrics.push(await groupedMetric(env, field, metric, coachId));
    out.push(fieldOut);
  }
  return out;
}

async function groupedMetric(env, field, metric, coachId = null) {
  let query =
    `SELECT l.label_value AS value, p.${metric} AS metric_value
     FROM labels l
     JOIN poi_metrics p ON p.session_pitch = l.session_pitch
     WHERE l.item_name = ?
       AND l.skipped = 0
       AND l.label_value <> 'unclear'
       AND p.${metric} IS NOT NULL`;
  const params = [field];
  if (coachId !== null) {
    query += " AND l.coach_id = ?";
    params.push(coachId);
  }
  const rows = await env.DB.prepare(query).bind(...params).all();
  const grouped = new Map();
  for (const row of rows.results) {
    if (!grouped.has(row.value)) grouped.set(row.value, []);
    grouped.get(row.value).push(Number(row.metric_value));
  }
  const orderedValues = orderedFieldValues(field, [...grouped.keys()]);
  const groups = orderedValues.map((value) => {
    const values = grouped.get(value);
    return {
      value,
      values,
      summary: {
        value,
        n: values.length,
        mean: round(mean(values), 4),
        sd: values.length > 1 ? round(sd(values), 4) : null,
      },
    };
  });
  return {
    metric,
    groups: groups.map((group) => group.summary),
    tests: pairwiseWelch(groups),
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

function orderedFieldValues(field, values) {
  const preferred = FIELD_VALUE_ORDER[field] || [];
  const seen = new Set(values);
  const ordered = preferred.filter((value) => seen.has(value));
  for (const value of values.sort()) {
    if (!ordered.includes(value)) ordered.push(value);
  }
  return ordered;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sd(values) {
  const avg = mean(values);
  const variance = values.reduce((sum, value) => sum + ((value - avg) ** 2), 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function round(value, digits) {
  return Math.round(value * (10 ** digits)) / (10 ** digits);
}

function pairwiseWelch(groups) {
  const tests = [];
  for (let i = 0; i < groups.length; i += 1) {
    for (let j = i + 1; j < groups.length; j += 1) {
      tests.push(welchSummary(groups[i], groups[j]));
    }
  }
  return tests;
}

function welchSummary(leftGroup, rightGroup) {
  const left = leftGroup.values;
  const right = rightGroup.values;
  if (left.length < 2 || right.length < 2) {
    return {
      test: "welch",
      comparison: `${leftGroup.value}-${rightGroup.value}`,
      reason: "need_at_least_two_values_per_group",
    };
  }
  const leftMean = mean(left);
  const rightMean = mean(right);
  const leftVar = variance(left);
  const rightVar = variance(right);
  const seSquared = (leftVar / left.length) + (rightVar / right.length);
  if (seSquared <= 0) {
    return {
      test: "welch",
      comparison: `${leftGroup.value}-${rightGroup.value}`,
      reason: "zero_standard_error",
    };
  }
  const t = (leftMean - rightMean) / Math.sqrt(seSquared);
  const numerator = seSquared ** 2;
  const denominator = ((leftVar / left.length) ** 2 / (left.length - 1)) + ((rightVar / right.length) ** 2 / (right.length - 1));
  const df = numerator / denominator;
  const p = 2 * (1 - studentTCdf(Math.abs(t), df));
  return {
    test: "welch",
    comparison: `${leftGroup.value}-${rightGroup.value}`,
    t: round(t, 4),
    df: round(df, 4),
    p: round(Math.max(0, Math.min(1, p)), 4),
    mean_diff: round(leftMean - rightMean, 4),
    cohen_d: round(cohenD(left, right), 4),
  };
}

function variance(values) {
  const avg = mean(values);
  return values.reduce((sum, value) => sum + ((value - avg) ** 2), 0) / (values.length - 1);
}

function cohenD(left, right) {
  const pooled = (((left.length - 1) * variance(left)) + ((right.length - 1) * variance(right))) / (left.length + right.length - 2);
  return pooled > 0 ? (mean(left) - mean(right)) / Math.sqrt(pooled) : null;
}

function studentTCdf(t, df) {
  if (df <= 0) return NaN;
  const x = df / (df + (t * t));
  const ib = regularizedIncompleteBeta(x, df / 2, 0.5);
  return 1 - (0.5 * ib);
}

function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + (a * Math.log(x)) + (b * Math.log(1 - x)));
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betaContinuedFraction(x, a, b)) / a;
  }
  return 1 - ((bt * betaContinuedFraction(1 - x, b, a)) / b);
}

function betaContinuedFraction(x, a, b) {
  const maxIterations = 100;
  const epsilon = 3e-7;
  const fpMin = 1e-30;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x / qap);
  if (Math.abs(d) < fpMin) d = fpMin;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= maxIterations; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + (aa * d);
    if (Math.abs(d) < fpMin) d = fpMin;
    c = 1 + (aa / c);
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;
    h *= d * c;
    aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + (aa * d);
    if (Math.abs(d) < fpMin) d = fpMin;
    c = 1 + (aa / c);
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < epsilon) break;
  }
  return h;
}

function logGamma(value) {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  let x = 0.9999999999998099;
  const z = value - 1;
  for (let i = 0; i < coefficients.length; i += 1) {
    x += coefficients[i] / (z + i + 1);
  }
  const t = z + coefficients.length - 0.5;
  return (0.5 * Math.log(2 * Math.PI)) + ((z + 0.5) * Math.log(t)) - t + Math.log(x);
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
