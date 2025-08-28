import sqlite3, os
from datetime import datetime
DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")
def ensure():
    con = sqlite3.connect(DB_PATH)
    con.execute('''CREATE TABLE IF NOT EXISTS telemetry(
        ts TEXT, session_id TEXT, variant TEXT, task TEXT,
        latency_ms REAL, tokens_user INT, tokens_total INT, cost_usd REAL)''')
    con.commit(); con.close()
def log(session_id, variant, task, latency_ms, tokens_user, tokens_total, cost_usd=0.0):
    ensure(); con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO telemetry VALUES(?,?,?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(), session_id, variant, task, latency_ms, tokens_user, tokens_total, cost_usd))
    con.commit(); con.close()
