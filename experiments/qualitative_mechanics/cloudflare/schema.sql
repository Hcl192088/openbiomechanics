DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS label_tasks;
DROP TABLE IF EXISTS coaches;

CREATE TABLE coaches (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  is_admin INTEGER NOT NULL DEFAULT 0,
  must_change_password INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE label_tasks (
  id TEXT PRIMARY KEY,
  session_pitch TEXT NOT NULL UNIQUE,
  display_order INTEGER NOT NULL UNIQUE,
  pitcher_id TEXT NOT NULL,
  p_throws TEXT NOT NULL,
  filename_new TEXT NOT NULL,
  c3d_path TEXT NOT NULL,
  active_label_fields TEXT NOT NULL,
  active INTEGER NOT NULL
);

CREATE TABLE poi_metrics (
  session_pitch TEXT PRIMARY KEY,
  pitch_speed_mph REAL,
  max_rotation_hip_shoulder_separation REAL,
  rotation_hip_shoulder_separation_fp REAL,
  shoulder_horizontal_abduction_fp REAL,
  max_shoulder_horizontal_abduction REAL,
  max_torso_rotational_velo REAL,
  pelvis_rotation_fp REAL,
  cog_velo_pkh REAL,
  stride_length REAL,
  stride_angle REAL,
  max_rear_hip_flexion REAL,
  max_rear_hip_internal_rotation_velo REAL,
  rear_hip_transfer_pkh_fp REAL,
  rear_hip_generation_pkh_fp REAL,
  rear_hip_absorption_pkh_fp REAL,
  lead_hip_transfer_fp_br REAL,
  lead_hip_generation_fp_br REAL,
  lead_hip_absorption_fp_br REAL,
  lead_knee_extension_from_fp_to_br REAL,
  lead_knee_extension_angular_velo_fp REAL,
  lead_grf_x_max REAL,
  lead_grf_y_max REAL,
  lead_grf_z_max REAL,
  rear_grf_x_max REAL,
  rear_grf_y_max REAL,
  rear_grf_z_max REAL,
  max_cog_velo_x REAL
);

CREATE TABLE labels (
  coach_id TEXT NOT NULL,
  session_pitch TEXT NOT NULL,
  item_name TEXT NOT NULL,
  label_value TEXT NOT NULL,
  view_used TEXT NOT NULL,
  playback_speed TEXT NOT NULL,
  skipped INTEGER NOT NULL,
  skip_reason TEXT NOT NULL,
  notes TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (coach_id, session_pitch, item_name),
  FOREIGN KEY (coach_id) REFERENCES coaches(id),
  FOREIGN KEY (session_pitch) REFERENCES label_tasks(session_pitch)
);

CREATE TABLE sessions (
  token TEXT PRIMARY KEY,
  coach_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  FOREIGN KEY (coach_id) REFERENCES coaches(id)
);
