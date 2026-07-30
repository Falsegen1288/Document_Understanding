import sqlite3
import datetime
from pathlib import Path

class CostLedger:
    def __init__(self, db_path: str = "outputs/cost_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                n_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                run_id TEXT NOT NULL,
                call_type TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def log_call(self, model_key: str, n_tokens: int, cost_usd: float, run_id: str, call_type: str) -> None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cost_log (model_key, timestamp, n_tokens, cost_usd, run_id, call_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_key, timestamp, n_tokens, cost_usd, run_id, call_type))
        conn.commit()
        conn.close()

    def get_total(self, model_key: str | None = None) -> float:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if model_key:
            cursor.execute("SELECT SUM(cost_usd) FROM cost_log WHERE model_key = ?", (model_key,))
        else:
            cursor.execute("SELECT SUM(cost_usd) FROM cost_log")
        row = cursor.fetchone()
        conn.close()
        return float(row[0]) if row and row[0] is not None else 0.0

    def get_run_summary(self, run_id: str) -> dict[str, float]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_key, SUM(cost_usd) FROM cost_log
            WHERE run_id = ?
            GROUP BY model_key
        """, (run_id,))
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: float(row[1]) for row in rows}
