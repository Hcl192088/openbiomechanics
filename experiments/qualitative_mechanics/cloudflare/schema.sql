DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS label_tasks;
DROP TABLE IF EXISTS coaches;

CREATE TABLE coaches (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  is_admin INTEGER NOT NULL DEFAULT 0,
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
  FOREIGN KEY (coach_id) REFERENCES coaches(id)
);
