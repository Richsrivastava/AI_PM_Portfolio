import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths (align with telemetry.py usage)
# ─────────────────────────────────────────────────────────────────────────────
SUCCESS_DB_PATH = os.path.join(os.path.dirname(__file__), "success.db")


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers / migrations
# ─────────────────────────────────────────────────────────────────────────────
def _get_con(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)

def _table_columns(con: sqlite3.Connection, table: str):
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()

def _add_col_if_missing(con: sqlite3.Connection, table: str, name: str, decl: str):
    cols = _table_columns(con, table)
    if name not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def ensure_success_db():
    """
    Ensure the success metrics database exists with a superset schema and
    migrate missing columns for backwards compatibility.
    """
    con = _get_con(SUCCESS_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS success_metrics(
            ts TEXT,
            session_id TEXT,
            variant TEXT,
            task TEXT,
            response_text TEXT,
            user_satisfied BOOLEAN,
            auto_eval_score REAL,
            keyphrases TEXT
        )
        """
    )
    # Migrate if table exists with older/different ordering
    _add_col_if_missing(con, "success_metrics", "response_text", "TEXT")
    _add_col_if_missing(con, "success_metrics", "user_satisfied", "BOOLEAN")
    _add_col_if_missing(con, "success_metrics", "auto_eval_score", "REAL")
    _add_col_if_missing(con, "success_metrics", "keyphrases", "TEXT")
    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-eval utility
# ─────────────────────────────────────────────────────────────────────────────
def score_example(response_text: str, expected_keyphrases: List[str]) -> float:
    """
    Score an LLM response based on keyphrase coverage. Returns a float in [0,1].
    """
    if not expected_keyphrases:
        return 1.0
    text_low = (response_text or "").lower()
    hits = sum(1 for phrase in expected_keyphrases if phrase.lower() in text_low)
    return hits / max(1, len(expected_keyphrases))


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_bool(val: Optional[object]) -> Optional[int]:
    """
    Normalize assorted truthy/falsey representations to 1/0 for storage.
    Returns None if val is None.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    s = str(val).strip().lower()
    truthy = {"1", "true", "yes", "y", "👍", "👍 yes", "yes 👍"}
    falsy  = {"0", "false", "no", "n", "👎", "👎 no", "no 👎"}
    if s in truthy:
        return 1
    if s in falsy:
        return 0
    # Unknown → treat as None so it doesn't skew stats
    return None


def log_success_metrics(
    session_id: str,
    variant: str,
    task: str,
    response_text: str,
    user_satisfied: Optional[bool] = None,
    keyphrases: Optional[List[str]] = None
):
    """
    Log success metrics including user satisfaction and auto-evaluation.
    """
    ensure_success_db()

    # Auto-evaluate using keyphrase coverage if provided
    auto_eval_score = None
    if keyphrases:
        auto_eval_score = score_example(response_text or "", keyphrases)

    # Store keyphrases as JSON string
    keyphrases_json = None
    if keyphrases:
        import json
        keyphrases_json = json.dumps(keyphrases)

    norm_sat = _normalize_bool(user_satisfied)

    con = _get_con(SUCCESS_DB_PATH)
    con.execute(
        """
        INSERT INTO success_metrics
            (ts, session_id, variant, task, response_text, user_satisfied, auto_eval_score, keyphrases)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            session_id,
            variant,
            task,
            response_text,
            norm_sat,
            auto_eval_score,
            keyphrases_json,
        ),
    )
    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Task keyphrases (used by apps that want built-ins)
# ─────────────────────────────────────────────────────────────────────────────
def get_task_keyphrases(task: str) -> List[str]:
    mapping = {
        "wizard_holiday_planning": [
            "itinerary", "day", "hotel", "restaurant", "activity",
            "transportation", "schedule", "morning", "afternoon", "evening"
        ],
        "direct_chat_planning": [
            "destination", "travel", "accommodation", "activities",
            "budget", "duration", "attractions", "food", "culture"
        ],
        "sample_itinerary_planning": [
            "itinerary", "schedule", "attractions", "accommodation",
            "dining", "activities", "transportation", "recommendations"
        ],
        # Legacy names
        "Wizard Planner": [
            "itinerary", "day", "hotel", "restaurant", "activity",
            "transportation", "schedule", "morning", "afternoon", "evening"
        ],
        "Direct Chat": [
            "destination", "travel", "accommodation", "activities",
            "budget", "duration", "attractions", "food", "culture"
        ],
        "Sample Itinerary": [
            "itinerary", "schedule", "attractions", "accommodation",
            "dining", "activities", "transportation", "recommendations"
        ],
    }
    return mapping.get(task, [])


# ─────────────────────────────────────────────────────────────────────────────
# Aggregates (NULL-safe)
# ─────────────────────────────────────────────────────────────────────────────
def get_success_stats(db_path: Optional[str] = None) -> Dict:
    """
    Returns aggregate success metrics with NULL-safe math.
    Percentages are already scaled to 0–100.
    """
    path = db_path or SUCCESS_DB_PATH

    if not os.path.exists(path):
        return {
            "total_responses": 0,
            "responses_with_feedback": 0,
            "user_satisfaction_rate": 0.0,
            "avg_auto_eval_score": 0.0,  # percentage
        }

    con = _get_con(path)
    cur = con.cursor()

    # Total responses
    cur.execute("SELECT COUNT(*) FROM success_metrics")
    total_responses = int(cur.fetchone()[0] or 0)

    # Feedback counts — COALESCE SUM() so 0 rows ⇒ 0, not NULL
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_feedback,
            COALESCE(
                SUM(
                    CASE
                      WHEN user_satisfied IN (1,'1','true','TRUE','True','yes','YES','Yes','y','Y','👍','👍 Yes','Yes 👍')
                      THEN 1 ELSE 0
                    END
                ), 0
            ) AS satisfied_count
        FROM success_metrics
        WHERE user_satisfied IS NOT NULL
        """
    )
    row = cur.fetchone() or (0, 0)
    responses_with_feedback = int(row[0] or 0)
    satisfied_count = int(row[1] or 0)

    # Average auto-eval score (0..1) → percentage
    cur.execute("SELECT AVG(auto_eval_score) FROM success_metrics WHERE auto_eval_score IS NOT NULL")
    avg_row = cur.fetchone()
    avg_auto_eval_pct = float((avg_row[0] or 0.0) * 100.0)

    con.close()

    # NULL/zero-safe division for satisfaction rate
    satisfaction_rate = (satisfied_count / max(1, responses_with_feedback)) * 100.0

    return {
        "total_responses": total_responses,
        "responses_with_feedback": responses_with_feedback,
        "user_satisfaction_rate": satisfaction_rate,   # %
        "avg_auto_eval_score": avg_auto_eval_pct,      # %
    }
