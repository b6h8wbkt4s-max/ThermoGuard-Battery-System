import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sensor.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建表（幂等），并给 timestamp 建索引以支持范围查询。"""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS environment_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                voltage REAL NOT NULL,
                source_type TEXT NOT NULL
            )
            """
        )
        # 迁移：旧版 schema 的 light（光照）列重命名为 voltage（电压）。
        # 只改列名不动数据——旧库中的历史数值仍是光照时代的刻度，需重跑 simulate_data.py 覆盖。
        cols = {row[1] for row in conn.execute("PRAGMA table_info(environment_samples)")}
        if "light" in cols and "voltage" not in cols:
            conn.execute("ALTER TABLE environment_samples RENAME COLUMN light TO voltage")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_timestamp ON environment_samples(timestamp)"
        )
        # B07 统一分析流水线：每次 /analyze 调用的完整记录（evidence/judgment 存 JSON 文本）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                analysis_time TEXT NOT NULL,
                location TEXT NOT NULL,
                evidence TEXT NOT NULL,
                judgment TEXT NOT NULL,
                explanation TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def replace_simulated(samples):
    """清空 SIMULATED 数据后批量写入，脚本可重复执行而不产生重复数据。

    samples: [(timestamp, temperature, humidity, voltage, source_type), ...]
    """
    with get_connection() as conn:
        conn.execute("DELETE FROM environment_samples WHERE source_type = 'SIMULATED'")
        conn.executemany(
            "INSERT INTO environment_samples (timestamp, temperature, humidity, voltage, source_type) "
            "VALUES (?, ?, ?, ?, ?)",
            samples,
        )


def insert_sample(timestamp, temperature, humidity, voltage, source_type="ONENET"):
    """插入单条环境样本（如 OneNET 硬件推送），返回自增 id。"""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO environment_samples (timestamp, temperature, humidity, voltage, source_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, temperature, humidity, voltage, source_type),
        )
    return cur.lastrowid


def insert_analysis_run(run_id, analysis_time, location, evidence, judgment, explanation, created_at):
    """保存一次 /analyze 的完整记录（evidence/judgment 为 JSON 字符串）。"""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO analysis_runs (run_id, analysis_time, location, evidence, judgment, explanation, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, analysis_time, location, evidence, judgment, explanation, created_at),
        )


def latest_analysis_runs(limit=10):
    """按时间倒序取最近的分析记录。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT run_id, analysis_time, location, evidence, judgment, explanation, created_at "
            "FROM analysis_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_analysis_run(run_id):
    """按 run_id 精确查询一条分析记录（evidence/judgment 为 JSON 字符串），不存在返回 None。"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT run_id, analysis_time, location, evidence, judgment, explanation, created_at "
            "FROM analysis_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def latest_samples(n=6):
    """按时间倒序取最近 n 条，返回升序列表（旧→新）。"""
    sql = (
        "SELECT timestamp, temperature, humidity, voltage, source_type "
        "FROM environment_samples ORDER BY timestamp DESC LIMIT ?"
    )
    with get_connection() as conn:
        rows = conn.execute(sql, (n,)).fetchall()
    return list(reversed([dict(row) for row in rows]))


def query_history(start=None, end=None, limit=5000):
    """按时间范围查询（含边界）。timestamp 为 'YYYY-MM-DD HH:MM:SS' 字符串，字典序即时间序。"""
    sql = "SELECT timestamp, temperature, humidity, voltage, source_type FROM environment_samples WHERE 1=1"
    params = []
    if start:
        sql += " AND timestamp >= ?"
        params.append(start)
    if end:
        sql += " AND timestamp <= ?"
        params.append(end)
    sql += " ORDER BY timestamp ASC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]
