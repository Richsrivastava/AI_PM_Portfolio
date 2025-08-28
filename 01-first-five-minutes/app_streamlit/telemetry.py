# telemetry.py
# Consolidated telemetry, journey tracking, and auto-eval hypothesis testing utilities.

import os
import json
import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")
JOURNEY_DB_PATH = os.path.join(os.path.dirname(__file__), "user_journey.db")
SUCCESS_DB_PATH = os.path.join(os.path.dirname(__file__), "success.db")


# ---------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------
class _MedianAggregator:
    """(Optional) SQLite aggregate to support MEDIAN() if ever needed."""
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            try:
                self.values.append(float(value))
            except (TypeError, ValueError):
                pass

    def finalize(self):
        if not self.values:
            return None
        s = sorted(self.values)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        else:
            return (s[mid - 1] + s[mid]) / 2.0


def _get_connection(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    # Register MEDIAN aggregate for this connection (kept for compatibility)
    con.create_aggregate("MEDIAN", 1, _MedianAggregator)
    return con


# ---- New migration helpers & Python-side median ----
def _table_columns(con, table):
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}

def _add_col_if_missing(con, table, name, decl):
    cols = _table_columns(con, table)
    if name not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

def _median(seq):
    s = [x for x in seq if x is not None]
    if not s:
        return None
    s.sort()
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else float((s[mid-1] + s[mid]) / 2.0)


# ---------------------------------------------------------------------
# Schema ensures (+ migrations)
# ---------------------------------------------------------------------
def ensure():
    """Ensure telemetry/user_journey DBs exist and migrate missing columns."""
    # Telemetry DB (create + migrate)
    con = _get_connection(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry(
            ts TEXT,
            session_id TEXT,
            variant TEXT,
            task TEXT,
            latency_ms REAL,
            tokens_user INT,
            tokens_total INT,
            cost_usd REAL,
            session_start_ts TEXT,
            task_completed BOOLEAN,
            steps_to_completion INTEGER,
            abandonment_point TEXT,
            time_to_value_sec INTEGER
        )
        """
    )
    # ---- migrate older DBs that don't have new columns ----
    _add_col_if_missing(con, "telemetry", "session_start_ts", "TEXT")
    _add_col_if_missing(con, "telemetry", "task_completed", "INTEGER")
    _add_col_if_missing(con, "telemetry", "steps_to_completion", "INTEGER")
    _add_col_if_missing(con, "telemetry", "abandonment_point", "TEXT")
    _add_col_if_missing(con, "telemetry", "time_to_value_sec", "INTEGER")
    con.commit()
    con.close()

    # Journey DB
    con = _get_connection(JOURNEY_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS user_journey(
            ts TEXT,
            session_id TEXT,
            variant TEXT,
            event_type TEXT,
            event_data TEXT,
            cumulative_time_sec INTEGER,
            step_number INTEGER
        )
        """
    )
    con.commit()
    con.close()

    # Success DB (for auto-eval)
    ensure_success_db()


def ensure_success_db():
    """Ensure success database (auto-eval storage) exists and migrate columns."""
    con = _get_connection(SUCCESS_DB_PATH)
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
    # Migrate any missing columns for older tables
    _add_col_if_missing(con, "success_metrics", "response_text", "TEXT")
    _add_col_if_missing(con, "success_metrics", "user_satisfied", "INTEGER")
    _add_col_if_missing(con, "success_metrics", "auto_eval_score", "REAL")
    _add_col_if_missing(con, "success_metrics", "keyphrases", "TEXT")
    con.commit()
    con.close()


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------
def calculate_claude_cost(input_tokens, output_tokens):
    """Calculate cost for Claude Haiku model."""
    input_cost = (input_tokens / 1_000_000) * 0.80
    output_cost = (output_tokens / 1_000_000) * 4.00
    return input_cost + output_cost


def log_journey_event(
    session_id,
    variant,
    event_type,
    event_data=None,
    step_number=None,
):
    """
    Log user journey events for detailed analysis.

    Args:
        session_id: Session identifier
        variant: App variant (A/B/C)
        event_type: 'session_start', 'step_start', 'step_complete', 'task_complete', 'abandon', 'retry'
        event_data: dict with additional event data
        step_number: step number for wizard flow (1, 2, 3, ...)
    """
    ensure()

    # Calculate cumulative time from session start
    con = _get_connection(JOURNEY_DB_PATH)

    session_start = con.execute(
        "SELECT MIN(ts) FROM user_journey WHERE session_id = ? AND event_type = 'session_start'",
        (session_id,),
    ).fetchone()[0]

    cumulative_time_sec = 0
    if session_start:
        start_dt = datetime.fromisoformat(session_start)
        current_dt = datetime.utcnow()
        cumulative_time_sec = int((current_dt - start_dt).total_seconds())

    event_data_json = json.dumps(event_data) if event_data else None

    con.execute(
        """
        INSERT INTO user_journey
            (ts, session_id, variant, event_type, event_data, cumulative_time_sec, step_number)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            session_id,
            variant,
            event_type,
            event_data_json,
            cumulative_time_sec,
            step_number,
        ),
    )
    con.commit()
    con.close()


def calculate_session_metrics(session_id):
    """Calculate session-level metrics from journey events."""
    con = _get_connection(JOURNEY_DB_PATH)
    events = con.execute(
        """
        SELECT event_type, cumulative_time_sec, step_number, event_data
        FROM user_journey
        WHERE session_id = ?
        ORDER BY ts
        """,
        (session_id,),
    ).fetchall()
    con.close()

    if not events:
        return None

    session_start_time = 0
    task_completion_time = None
    steps_to_completion = 0
    abandonment_point = None
    task_completed = False

    for event_type, cum_time, step_num, _event_data in events:
        if event_type == "session_start":
            session_start_time = cum_time
        elif event_type == "task_complete":
            task_completion_time = cum_time
            task_completed = True
        elif event_type == "step_complete":
            steps_to_completion += 1
        elif event_type == "abandon":
            abandonment_point = f"step_{step_num}" if step_num else "unknown"
            break

    time_to_value_sec = None
    if task_completion_time is not None:
        time_to_value_sec = task_completion_time - session_start_time

    return {
        "task_completed": task_completed,
        "steps_to_completion": steps_to_completion,
        "abandonment_point": abandonment_point,
        "time_to_value_sec": time_to_value_sec,
    }


def log(
    session_id,
    variant,
    latency_ms,
    tokens_user,
    tokens_assistant,
    task=None,
    cost_usd=None,
    session_start_ts=None,
):
    """
    Enhanced logging with session tracking.

    Args:
        session_id: Unique session identifier
        variant: App variant (A, B, C)
        latency_ms: Response time in milliseconds
        tokens_user: Input tokens from user
        tokens_assistant: Output tokens from assistant
        task: Task description or identifier
        cost_usd: Cost in USD (auto-calculated if None)
        session_start_ts: When the session started (ISO format)
    """
    ensure()

    if cost_usd is None:
        cost_usd = calculate_claude_cost(tokens_user, tokens_assistant)

    tokens_total = (tokens_user or 0) + (tokens_assistant or 0)

    session_metrics = calculate_session_metrics(session_id)
    task_completed = session_metrics["task_completed"] if session_metrics else None
    steps_to_completion = (
        session_metrics["steps_to_completion"] if session_metrics else None
    )
    abandonment_point = (
        session_metrics["abandonment_point"] if session_metrics else None
    )
    time_to_value_sec = (
        session_metrics["time_to_value_sec"] if session_metrics else None
    )

    con = _get_connection(DB_PATH)
    con.execute(
        """
        INSERT INTO telemetry
            (ts, session_id, variant, task, latency_ms, tokens_user, tokens_total, cost_usd,
             session_start_ts, task_completed, steps_to_completion, abandonment_point, time_to_value_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            session_id,
            variant,
            task,
            latency_ms,
            tokens_user,
            tokens_total,
            cost_usd,
            session_start_ts,
            task_completed,
            steps_to_completion,
            abandonment_point,
            time_to_value_sec,
        ),
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------
# Aggregate stats (telemetry / journey)
# ---------------------------------------------------------------------
def get_hypothesis_stats():
    """Get statistics for hypothesis testing from telemetry.db."""
    if not os.path.exists(DB_PATH):
        return None

    con = _get_connection(DB_PATH)

    # Task success rate per variant (SQL)
    success_rates = con.execute(
        """
        SELECT
            variant,
            COUNT(*) as total_sessions,
            SUM(CASE WHEN task_completed = 1 THEN 1 ELSE 0 END) as successful_sessions,
            AVG(CASE WHEN task_completed = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate
        FROM telemetry
        WHERE task_completed IS NOT NULL
        GROUP BY variant
        ORDER BY variant
        """
    ).fetchall()

    # Time-to-value stats: compute median in Python (avoid SQL MEDIAN)
    rows = con.execute(
        """
        SELECT variant, time_to_value_sec
        FROM telemetry
        WHERE time_to_value_sec IS NOT NULL
        ORDER BY variant
        """
    ).fetchall()
    con.close()

    from collections import defaultdict
    by_variant = defaultdict(list)
    for v, t in rows:
        try:
            by_variant[v].append(int(t))
        except (TypeError, ValueError):
            pass

    ttv_stats = []
    for v in sorted(by_variant.keys()):
        vals = by_variant[v]
        if not vals:
            continue
        avg = float(sum(vals) / len(vals))
        med = _median(vals)
        ttv_stats.append(
            (v, len(vals), avg, med, float(min(vals)), float(max(vals)))
        )

    # User journey metrics (SQL)
    con = _get_connection(DB_PATH)
    journey_stats = con.execute(
        """
        SELECT
            variant,
            AVG(steps_to_completion) as avg_steps,
            AVG(tokens_user) as avg_input_complexity,
            COUNT(*) as sample_size
        FROM telemetry
        WHERE steps_to_completion IS NOT NULL
        GROUP BY variant
        ORDER BY variant
        """
    ).fetchall()
    con.close()

    return {
        "success_rates": success_rates,
        "ttv_stats": ttv_stats,
        "journey_stats": journey_stats,
    }


def get_abandonment_analysis():
    """Analyze where users abandon in each variant from user_journey.db."""
    if not os.path.exists(JOURNEY_DB_PATH):
        return None

    con = _get_connection(JOURNEY_DB_PATH)
    abandonment_points = con.execute(
        """
        SELECT
            variant,
            event_data,
            step_number,
            COUNT(*) as abandon_count
        FROM user_journey
        WHERE event_type = 'abandon'
        GROUP BY variant, event_data, step_number
        ORDER BY variant, step_number
        """
    ).fetchall()
    con.close()
    return abandonment_points


# ---------------------------------------------------------------------
# Auto-eval hypothesis testing utilities (success.db)
# ---------------------------------------------------------------------
def get_auto_eval_hypothesis_stats(success_threshold=0.7, start_date=None, end_date=None):
    """Get statistics for hypothesis testing based on auto-evaluation scores."""
    if not os.path.exists(SUCCESS_DB_PATH):
        return None

    con = _get_connection(SUCCESS_DB_PATH)

    date_filter = ""
    params = [success_threshold, success_threshold]
    if start_date and end_date:
        date_filter = "AND DATE(ts) BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    # No SQL MEDIAN: compute min/max/avg in SQL, median in Python later
    success_rates_rows = con.execute(
        f"""
        SELECT 
            variant,
            COUNT(*) as total_sessions,
            SUM(CASE WHEN auto_eval_score >= ? THEN 1 ELSE 0 END) as successful_sessions,
            AVG(CASE WHEN auto_eval_score >= ? THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
            AVG(auto_eval_score) as avg_auto_eval_score,
            MIN(auto_eval_score) as min_auto_eval_score,
            MAX(auto_eval_score) as max_auto_eval_score
        FROM success_metrics 
        WHERE auto_eval_score IS NOT NULL {date_filter}
        GROUP BY variant
        ORDER BY variant
        """,
        params
    ).fetchall()

    # Pull raw per-variant scores to compute medians & distributions
    raw_rows = con.execute(
        f"""
        SELECT variant, auto_eval_score, task
        FROM success_metrics
        WHERE auto_eval_score IS NOT NULL {date_filter}
        ORDER BY variant
        """,
        params[2:] if (start_date and end_date) else []
    ).fetchall()
    con.close()

    from collections import defaultdict, Counter
    scores_by_variant = defaultdict(list)
    scores_by_variant_task = defaultdict(list)
    for v, s, task in raw_rows:
        try:
            sf = float(s)
            scores_by_variant[v].append(sf)
            scores_by_variant_task[(v, task)].append(sf)
        except (TypeError, ValueError):
            pass

    # Add Python-side medians back into success_rates output
    success_rates = []
    for row in success_rates_rows:
        v = row[0]
        success_rates.append((
            row[0],  # variant
            row[1],  # total_sessions
            row[2],  # successful_sessions
            row[3],  # success_rate
            row[4],  # avg_auto_eval_score
            _median(scores_by_variant.get(v, [])),  # median_auto_eval_score (python)
            row[5],  # min_auto_eval_score
            row[6],  # max_auto_eval_score
        ))

    # Score distributions (variant, score, frequency)
    score_distributions = []
    for v, lst in scores_by_variant.items():
        freq = Counter(lst)
        for score, count in sorted(freq.items()):
            score_distributions.append((v, score, count))

    # Task performance by variant
    task_performance = []
    for (v, task), lst in sorted(scores_by_variant_task.items()):
        n = len(lst)
        if n == 0:
            continue
        avg = sum(lst) / n
        succ = sum(1 for x in lst if x >= success_threshold)
        rate = succ / n * 100.0
        task_performance.append((v, task, n, avg, succ, rate))

    return {
        "success_rates": success_rates,
        "score_distributions": score_distributions,
        "task_performance": task_performance,
        "success_threshold": success_threshold,
    }


def get_statistical_test_data(success_threshold=0.7):
    """Prepare data for statistical tests comparing variants using auto-eval scores."""
    if not os.path.exists(SUCCESS_DB_PATH):
        return None

    con = _get_connection(SUCCESS_DB_PATH)
    raw_scores = con.execute(
        """
        SELECT variant, auto_eval_score, session_id
        FROM success_metrics
        WHERE auto_eval_score IS NOT NULL
        ORDER BY variant, ts
        """
    ).fetchall()
    con.close()

    if not raw_scores:
        return None

    variant_data = {}
    for variant, score, session_id in raw_scores:
        if variant not in variant_data:
            variant_data[variant] = {"scores": [], "success_binary": [], "session_ids": []}
        score_f = float(score)
        variant_data[variant]["scores"].append(score_f)
        variant_data[variant]["success_binary"].append(1 if score_f >= success_threshold else 0)
        variant_data[variant]["session_ids"].append(session_id)

    for variant in variant_data:
        scores = variant_data[variant]["scores"]
        successes = variant_data[variant]["success_binary"]
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        median = sorted(scores)[n // 2] if n else None  # simple/upper median
        var = (sum((x - mean) ** 2 for x in scores) / n) if n else 0.0
        std = var ** 0.5
        variant_data[variant]["summary"] = {
            "count": n,
            "mean_score": mean,
            "median_score": median,
            "std_score": std,
            "success_count": sum(successes),
            "success_rate": (sum(successes) / n) if n else 0.0,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
        }

    return variant_data


def calculate_power_analysis(variant_data, effect_size=0.3, alpha=0.05):
    """
    Calculate statistical power and required sample sizes for detecting effect sizes.

    Args:
        variant_data: Data from get_statistical_test_data()
        effect_size: Minimum effect size to detect (Cohen's d)
        alpha: Significance level (default 0.05)
    """
    try:
        from scipy import stats  # noqa: F401
        import numpy as np
    except ImportError:
        return {"error": "scipy and numpy required for power analysis"}

    if not variant_data or len(variant_data) < 2:
        return {"error": "Need at least 2 variants for power analysis"}

    # Choose baseline (Variant A if present)
    baseline_variant = "A" if "A" in variant_data else list(variant_data.keys())[0]
    baseline_data = variant_data[baseline_variant]

    results = {}
    for variant, test_data in variant_data.items():
        if variant == baseline_variant:
            continue

        b = baseline_data["summary"]
        t = test_data["summary"]
        denom = (b["count"] + t["count"] - 2)
        if denom <= 0 or b["std_score"] is None or t["std_score"] is None:
            observed_effect = 0.0
        else:
            pooled_std = (
                (((b["count"] - 1) * (b["std_score"] ** 2)) +
                 ((t["count"] - 1) * (t["std_score"] ** 2))) / denom
            ) ** 0.5
            observed_effect = (
                abs(t["mean_score"] - b["mean_score"]) / pooled_std if pooled_std > 0 else 0.0
            )

        # Rough sample size for 0.8 power two-sample t-test
        required_n_per_group = max(30, int(16 / (effect_size ** 2)))

        results[f"{baseline_variant}_vs_{variant}"] = {
            "observed_effect_size": observed_effect,
            "current_sample_sizes": [b["count"], t["count"]],
            "required_sample_size": required_n_per_group,
            "adequately_powered": min(b["count"], t["count"]) >= required_n_per_group,
            "effect_interpretation": (
                "Large" if observed_effect >= 0.8 else
                "Medium" if observed_effect >= 0.5 else
                "Small" if observed_effect >= 0.2 else "Negligible"
            ),
        }

    return results


def generate_hypothesis_report(success_threshold=0.7, include_raw_data=False):
    """Generate a comprehensive hypothesis testing report."""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "success_threshold": success_threshold,
        "methodology": "auto_eval_based",
    }

    auto_eval_stats = get_auto_eval_hypothesis_stats(success_threshold)
    if not auto_eval_stats:
        report["error"] = "No auto-evaluation data available"
        return report

    test_data = get_statistical_test_data(success_threshold)
    if not test_data:
        report["error"] = "Insufficient data for statistical testing"
        return report

    report["descriptive_stats"] = {v: d["summary"] for v, d in test_data.items()}

    # Statistical tests
    try:
        from scipy import stats
        import numpy as np

        score_groups = [data["scores"] for data in test_data.values()]
        if all(len(g) > 1 for g in score_groups) and len(score_groups) >= 2:
            f_stat, anova_p = stats.f_oneway(*score_groups)
            report["statistical_tests"] = {
                "anova": {
                    "f_statistic": float(f_stat),
                    "p_value": float(anova_p),
                    "significant": bool(anova_p < 0.05),
                    "interpretation": (
                        "Significant difference in auto-eval scores between variants"
                        if anova_p < 0.05
                        else "No significant difference in auto-eval scores"
                    ),
                }
            }
        else:
            report["statistical_tests"] = {
                "anova": {"error": "Not enough data for ANOVA (need ≥2 groups with >1 sample each)"}
            }

        # Chi-square for successes
        if len(test_data) >= 2:
            observed_success = []
            observed_failure = []
            for vd in test_data.values():
                sc = vd["summary"]["success_count"]
                tc = vd["summary"]["count"]
                observed_success.append(sc)
                observed_failure.append(tc - sc)

            if sum(observed_success) > 0:
                observed = [observed_success, observed_failure]
                chi2, chi2_p, dof, expected = stats.chi2_contingency(observed)
                report["statistical_tests"]["chi_square"] = {
                    "chi2_statistic": float(chi2),
                    "p_value": float(chi2_p),
                    "degrees_of_freedom": int(dof),
                    "significant": bool(chi2_p < 0.05),
                    "interpretation": (
                        "Significant difference in success rates between variants"
                        if chi2_p < 0.05
                        else "No significant difference in success rates"
                    ),
                }

        # Pairwise t-tests
        variants = list(test_data.keys())
        if len(variants) >= 2:
            report["pairwise_comparisons"] = {}
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    v1, v2 = variants[i], variants[j]
                    s1, s2 = test_data[v1]["scores"], test_data[v2]["scores"]
                    if len(s1) > 1 and len(s2) > 1:
                        t_stat, t_p = stats.ttest_ind(s1, s2, equal_var=False)
                        # Cohen's d
                        v1_var = np.var(s1, ddof=1)
                        v2_var = np.var(s2, ddof=1)
                        pooled_std = np.sqrt(((len(s1) - 1) * v1_var + (len(s2) - 1) * v2_var) / (len(s1) + len(s2) - 2)) if (len(s1) + len(s2) - 2) > 0 else 0.0
                        cohens_d = (np.mean(s1) - np.mean(s2)) / pooled_std if pooled_std > 0 else 0.0

                        key = f"{v1}_vs_{v2}"
                        report["pairwise_comparisons"][key] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(t_p),
                            "cohens_d": float(cohens_d),
                            "significant": bool(t_p < 0.05),
                            "effect_size": (
                                "Large" if abs(cohens_d) >= 0.8 else
                                "Medium" if abs(cohens_d) >= 0.5 else
                                "Small" if abs(cohens_d) >= 0.2 else "Negligible"
                            ),
                            "mean_difference": float(np.mean(s1) - np.mean(s2)),
                        }

    except ImportError:
        report["statistical_tests"] = {"error": "scipy not available for statistical testing"}

    power_analysis = calculate_power_analysis(test_data)
    if isinstance(power_analysis, dict) and "error" not in power_analysis:
        report["power_analysis"] = power_analysis

    report["conclusions"] = generate_conclusions(report)

    if include_raw_data:
        report["raw_data"] = test_data

    return report


def generate_conclusions(report):
    """Generate human-readable conclusions from statistical analysis."""
    conclusions = {
        "hypothesis_supported": False,
        "best_variant": None,
        "confidence_level": "low",
        "recommendations": [],
    }

    if "statistical_tests" not in report or "error" in report["statistical_tests"]:
        conclusions["recommendations"].append("Statistical testing unavailable - collect more data")
        return conclusions

    significant_tests = []
    anova = report["statistical_tests"].get("anova")
    if anova and anova.get("significant"):
        significant_tests.append("auto_eval_scores")

    chi = report["statistical_tests"].get("chi_square")
    if chi and chi.get("significant"):
        significant_tests.append("success_rates")

    if significant_tests:
        conclusions["hypothesis_supported"] = True
        conclusions["significant_metrics"] = significant_tests

        if "descriptive_stats" in report and report["descriptive_stats"]:
            best_variant = max(
                report["descriptive_stats"].keys(),
                key=lambda x: report["descriptive_stats"][x]["mean_score"],
            )
            conclusions["best_variant"] = best_variant
            conclusions["best_variant_score"] = report["descriptive_stats"][best_variant]["mean_score"]
            conclusions["best_variant_success_rate"] = report["descriptive_stats"][best_variant]["success_rate"]

    sample_sizes = [s["count"] for s in report.get("descriptive_stats", {}).values()] or [0]
    min_sample = min(sample_sizes)

    if min_sample >= 100:
        conclusions["confidence_level"] = "high"
    elif min_sample >= 50:
        conclusions["confidence_level"] = "medium"
    else:
        conclusions["confidence_level"] = "low"

    if conclusions["hypothesis_supported"]:
        if conclusions["confidence_level"] == "high":
            conclusions["recommendations"].extend(
                [
                    f"Deploy Variant {conclusions['best_variant']} as primary experience",
                    "Continue monitoring performance with larger sample sizes",
                    f"Investigate success factors of Variant {conclusions['best_variant']}",
                    "Apply learnings to improve other variants",
                ]
            )
        else:
            conclusions["recommendations"].extend(
                [
                    "Promising results detected - collect more data to confirm",
                    f"Monitor Variant {conclusions['best_variant']} closely",
                    "Increase sample sizes before making deployment decisions",
                ]
            )
    else:
        if conclusions["confidence_level"] == "low":
            conclusions["recommendations"].extend(
                [
                    "Collect more data - sample sizes too small for reliable conclusions",
                    "Continue running all variants until reaching 100+ samples each",
                ]
            )
        else:
            conclusions["recommendations"].extend(
                [
                    "No significant differences detected between variants",
                    "Consider testing more extreme variations",
                    "Review variant implementations for meaningful differences",
                    "Analyze qualitative feedback for insights",
                ]
            )

    return conclusions


def log_with_auto_eval(
    session_id,
    variant,
    latency_ms,
    tokens_user,
    tokens_assistant,
    auto_eval_score,
    task=None,
    cost_usd=None,
    session_start_ts=None,
):
    """
    Enhanced logging that includes auto-eval score for immediate hypothesis testing.
    Bridges telemetry.db and success.db for better analysis.
    """
    # Standard telemetry logging
    log(session_id, variant, latency_ms, tokens_user, tokens_assistant, task, cost_usd, session_start_ts)

    # Success DB logging
    ensure_success_db()
    con = _get_connection(SUCCESS_DB_PATH)
    con.execute(
        """
        INSERT INTO success_metrics
            (ts, session_id, variant, task, auto_eval_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            session_id,
            variant,
            task,
            auto_eval_score,
        ),
    )
    con.commit()
    con.close()


def get_real_time_hypothesis_status():
    """Get real-time status of hypothesis testing progress (auto-eval)."""
    test_data = get_statistical_test_data()
    if not test_data:
        return {
            "status": "no_data",
            "message": "No auto-evaluation data available yet",
        }

    sample_sizes = {variant: data["summary"]["count"] for variant, data in test_data.items()}
    min_sample = min(sample_sizes.values())
    total_samples = sum(sample_sizes.values())

    if min_sample < 30:
        return {
            "status": "collecting",
            "message": f"Collecting data... ({total_samples} total samples, need 30+ per variant)",
            "sample_sizes": sample_sizes,
            "progress": min_sample / 30.0,
        }
    elif min_sample < 100:
        return {
            "status": "preliminary",
            "message": f"Preliminary analysis ready ({total_samples} total samples)",
            "sample_sizes": sample_sizes,
            "progress": min_sample / 100.0,
        }
    else:
        return {
            "status": "ready",
            "message": f"Full statistical analysis ready ({total_samples} total samples)",
            "sample_sizes": sample_sizes,
            "progress": 1.0,
        }


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------
__all__ = [
    # Core & schema
    "ensure",
    "ensure_success_db",
    "calculate_claude_cost",
    "log_journey_event",
    "calculate_session_metrics",
    "log",
    "get_hypothesis_stats",
    "get_abandonment_analysis",
    # Auto-eval
    "get_auto_eval_hypothesis_stats",
    "get_statistical_test_data",
    "calculate_power_analysis",
    "generate_hypothesis_report",
    "log_with_auto_eval",
    "get_real_time_hypothesis_status",
]
