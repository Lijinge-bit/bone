import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    db = sqlite3.connect(str(path), timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=15000")
    return db


@contextmanager
def transaction(path):
    db = connect(path)
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def initialize(path):
    with transaction(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            source_image_id TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL,
            image_sha256 TEXT NOT NULL UNIQUE,
            sex TEXT CHECK(sex IN ('M','F') OR sex IS NULL),
            sex_source TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            qc_status TEXT NOT NULL,
            qc_flags TEXT NOT NULL,
            qc_metrics TEXT NOT NULL,
            qc_note TEXT NOT NULL DEFAULT '',
            qc_reviewer TEXT NOT NULL DEFAULT '',
            revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL REFERENCES cases(id),
            input_revision INTEGER NOT NULL,
            input_sha256 TEXT NOT NULL,
            input_sex TEXT NOT NULL,
            model_id TEXT NOT NULL,
            preprocess_version TEXT NOT NULL,
            status TEXT NOT NULL,
            prediction_months REAL,
            explanation_status TEXT NOT NULL DEFAULT 'pending',
            explanation_file TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_case
            ON evaluations(case_id) WHERE status IN ('queued','running');
        CREATE INDEX IF NOT EXISTS evaluations_by_case ON evaluations(case_id,created_at);
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
            decision TEXT NOT NULL,
            note TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            confirmed_months REAL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT REFERENCES cases(id),
            action TEXT NOT NULL,
            detail TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        db.execute("""UPDATE evaluations SET status='failed',
                   explanation_status='unavailable', error='服务重启，本轮任务已中断，请重新评估',
                   finished_at=? WHERE status IN ('queued','running')""", (now(),))


def event(db, case_id, action, detail):
    db.execute("INSERT INTO events(case_id,action,detail,created_at) VALUES(?,?,?,?)",
               (case_id, action, json.dumps(detail, ensure_ascii=False), now()))


def case_dict(row):
    obj = dict(row)
    obj["qc_flags"] = json.loads(obj["qc_flags"])
    obj["qc_metrics"] = json.loads(obj["qc_metrics"])
    obj["image_url"] = f"/api/cases/{obj['id']}/image"
    return obj
