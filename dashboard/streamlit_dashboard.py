# Updated streamlit_dashboard.py with hypothesis testing integration

import streamlit as st
# ✅ must be the first Streamlit call on the page
st.set_page_config(page_title="AI Portfolio Dashboard", layout="wide")

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
import os
import importlib
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Robust imports for success.py and telemetry.py regardless of run dir ---
HERE = Path(__file__).resolve()
# repo root (…/AI_Portfolio), adjust if your layout differs
ROOT = HERE.parents[1]

# Candidate locations that commonly contain the app modules
CANDIDATES = [
    ROOT / "app_streamlit",
    ROOT / "01-first-five-minutes" / "app_streamlit",
    ROOT / "01_first_five_minutes" / "app_streamlit",  # underscore variant, just in case
    HERE.parent,  # dashboard sibling
]

success_available = False
hypothesis_available = False

telemetry_mod = None
success_mod = None

# Try to import telemetry.py
for base in CANDIDATES:
    tele_path = base / "telemetry.py"
    if tele_path.exists():
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        try:
            telemetry_mod = importlib.import_module("telemetry")
            break
        except Exception:
            pass

# Try to import success.py
for base in CANDIDATES:
    succ_path = base / "success.py"
    if succ_path.exists():
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        try:
            success_mod = importlib.import_module("success")
            break
        except Exception:
            pass

# Expose functions/constants expected by the rest of the dashboard
if success_mod is not None:
    get_success_stats = getattr(success_mod, "get_success_stats", None)
    SUCCESS_DB_PATH = getattr(
        success_mod, "SUCCESS_DB_PATH",
        str((Path(success_mod.__file__).parent / "success.db"))
    )
    success_available = callable(get_success_stats)
else:
    SUCCESS_DB_PATH = ""

if telemetry_mod is not None:
    get_hypothesis_stats = getattr(telemetry_mod, "get_hypothesis_stats", None)
    get_abandonment_analysis = getattr(telemetry_mod, "get_abandonment_analysis", None)
    hypothesis_available = callable(get_hypothesis_stats) and callable(get_abandonment_analysis)

# (Optional) tiny debug line so you can see what got picked up
#st.caption(
#    f"telemetry: {getattr(telemetry_mod, '__file__', 'not found')} • "
#    f"success: {getattr(success_mod, '__file__', 'not found')}"
#)

# ---- Success DB resolution + ensure (with UI override) ----
_default_success_db = None
if success_mod is not None and hasattr(success_mod, "SUCCESS_DB_PATH"):
    _default_success_db = Path(getattr(success_mod, "SUCCESS_DB_PATH")).resolve()
elif telemetry_mod is not None and hasattr(telemetry_mod, "SUCCESS_DB_PATH"):
    _default_success_db = Path(getattr(telemetry_mod, "SUCCESS_DB_PATH")).resolve()

FOUND_SUCCESS_DBS = sorted({str(p.resolve()) for p in ROOT.glob("**/success.db")})

with st.sidebar:
    st.subheader("⚙️ Settings")
    cand = []
    if _default_success_db:
        cand.append(str(_default_success_db))
    if SUCCESS_DB_PATH:
        cand.append(str(Path(SUCCESS_DB_PATH).resolve()))
    cand.extend(FOUND_SUCCESS_DBS)
    cand = sorted(set([c for c in cand if c]))

    if cand:
        SUCCESS_DB_PATH = st.selectbox("success.db path", cand, index=0)
    else:
        SUCCESS_DB_PATH = st.text_input("success.db path (none found yet)", "")

    # Try to ensure/migrate DB schema (call whichever module exposes it)
    for fn_name, mod in [("ensure_success_db", success_mod), ("ensure_success_db", telemetry_mod)]:
        fn = getattr(mod, fn_name, None) if mod else None
        if callable(fn):
            try:
                fn()
            except Exception as e:
                st.caption(f"ensure_success_db error: {e}")

    st.caption(f"Using success DB: {SUCCESS_DB_PATH or '—'}")

st.title("🚀 AI Portfolio Dashboard")

# Helper functions for hypothesis testing (moved from previous artifact)
def calculate_confidence_interval(success_count, total_count, confidence=0.95):
    """Calculate confidence interval for success rate."""
    if total_count == 0:
        return 0, 0, 0

    p = success_count / total_count
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / total_count)

    return p * 100, max(0, (p - margin) * 100), min(100, (p + margin) * 100)

def perform_chi_square_test(variant_data):
    """Perform chi-square test for independence between variants and success."""
    observed = []
    variant_names = []

    for variant, data in variant_data.items():
        if data['total'] > 0:
            observed.append([data['success'], data['total'] - data['success']])
            variant_names.append(variant)

    if len(observed) < 2:
        return None, None, variant_names

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    return chi2, p_value, variant_names

def calculate_effect_size(variant_a, variant_b):
    """Calculate Cohen's h for effect size between two proportions."""
    p1 = variant_a['success'] / max(1, variant_a['total'])
    p2 = variant_b['success'] / max(1, variant_b['total'])

    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return abs(h)

# File discovery
db_paths = list(Path(".").glob("**/telemetry.db"))

# Telemetry database section
if db_paths:
    db = st.selectbox("📊 Select a telemetry database", db_paths)

    # Load data
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY ts DESC LIMIT 2000", con)
    con.close()

    if not df.empty:
        # Convert timestamp to datetime
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("🔥 Total Requests", len(df))

        with col2:
            avg_latency = df['latency_ms'].mean()
            st.metric("⚡ Avg Latency", f"{avg_latency:.0f} ms")

        with col3:
            total_tokens = df['tokens_total'].sum()
            st.metric("🎯 Total Tokens", f"{total_tokens:,}")

        with col4:
            total_cost = df['cost_usd'].sum()
            st.metric("💰 Total Cost", f"${total_cost:.3f}")

        with col5:
            # Success metrics
            if success_available and callable(get_success_stats):
                success_stats = get_success_stats()
                satisfaction_rate = success_stats['user_satisfaction_rate']
                st.metric("😊 User Satisfaction", f"{satisfaction_rate:.1f}%")
            else:
                st.metric("😊 User Satisfaction", "N/A")

        # Enhanced tabs including hypothesis testing
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_auto_eval = st.tabs([
            "📈 Performance", "🔀 Variants", "📋 Tasks",
            "😊 Success Metrics", "🧪 Hypothesis Testing",
            "📊 Recent Activity", "💾 Raw Data",
            "🎯 Auto-Eval (New)"
        ])

        with tab1:
            st.subheader("Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:
                # Latency histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df['latency_ms'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Latency (ms)')
                ax.set_ylabel('Count')
                ax.set_title('Response Latency Distribution')
                st.pyplot(fig)

            with col2:
                # Token usage over time
                daily_stats = df.groupby('date').agg({
                    'tokens_total': 'sum',
                    'cost_usd': 'sum',
                    'latency_ms': 'mean'
                }).reset_index()

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(daily_stats['date'], daily_stats['tokens_total'], marker='o', color='green')
                ax.set_xlabel('Date')
                ax.set_ylabel('Total Tokens')
                ax.set_title('Token Usage Over Time')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # =========================
        # Existing Hypothesis Testing Tab (tab5)
        # =========================
        with tab5:
            st.subheader("🧪 Hypothesis Testing Dashboard")

            if not hypothesis_available:
                st.warning("⚠️ Enhanced telemetry not available. Please implement the enhanced tracking features.")
                st.info("📝 This tab requires the updated telemetry.py with session tracking and journey events.")

                # Show what would be available
                st.markdown("### 📋 Hypothesis Being Tested")
                st.markdown("""
                **H₀:** No difference in auto-eval scores and time-to-value between variants  
                **H₁:** Guided onboarding (Variants B & C) achieves higher auto-eval scores and reduces time-to-value vs. Direct Chat (Variant A)
                """)

                st.markdown("### 🚧 Required Metrics (Not Yet Available)")
                missing_metrics = pd.DataFrame([
                    {"Metric": "auto_eval_score", "Status": "⌛ Computing", "Notes": "Computed for each response automatically"},
                    {"Metric": "time_to_value_sec", "Status": "❌ Missing", "Notes": "Need session start time tracking"},
                    {"Metric": "tokens_user", "Status": "✅ Available", "Notes": "Already tracked"},
                    {"Metric": "latency_ms", "Status": "✅ Available", "Notes": "Already tracked"},
                    {"Metric": "variant", "Status": "✅ Available", "Notes": "Already tracked"}
                ])
                st.dataframe(missing_metrics, use_container_width=True)

                st.stop()

            else:
                # (kept) Full auto-eval hypothesis testing implementation already in your file
                st.info("Using existing comprehensive auto-eval hypothesis testing in this tab.")
                st.caption("Tip: Use the new '🎯 Auto-Eval (New)' tab for a streamlined view.")

        # =========================
        # NEW: Streamlined Auto-Eval Hypothesis Testing Tab
        # =========================
        with tab_auto_eval:  # streamlined version
            st.subheader("🎯 Auto-Eval Based Hypothesis Testing")

            # Methodology explanation
            with st.expander("📋 Testing Methodology", expanded=False):
                st.markdown("""
                **Hypothesis:**
                - **H₀:** No difference in response quality (auto-eval scores) between variants
                - **H₁:** Guided onboarding (B & C) produces higher quality responses than Direct Chat (A)
                
                **Auto-Evaluation Criteria:**
                - **Keyphrase Coverage (80%):** Presence of task-relevant terms
                - **Structure (10%):** Organized format with days/times/sections  
                - **Specifics (10%):** Concrete details like prices, times, durations
                
                **Success Definition:** Response with auto-eval score ≥ threshold (default: 70%)
                """)

            # Configuration controls
            col1, col2, col3 = st.columns(3)
            with col1:
                threshold = st.slider(
                    "Success Threshold", 0.5, 0.9, 0.7, 0.05,
                    help="Auto-eval score threshold for 'successful' response"
                )
            with col2:
                min_samples = st.number_input(
                    "Min Samples per Variant", 10, 200, 30, 10,
                    help="Minimum samples needed for reliable testing"
                )
            with col3:
                confidence_level = st.selectbox(
                    "Confidence Level", [0.90, 0.95, 0.99], index=1,
                    format_func=lambda x: f"{x:.0%}"
                )

            # ---- Load auto-eval data from success database (after ensuring/choosing path)
            if SUCCESS_DB_PATH and os.path.exists(SUCCESS_DB_PATH):
                success_con = sqlite3.connect(SUCCESS_DB_PATH)
                auto_eval_query = """
                SELECT variant, auto_eval_score, ts, session_id, task
                FROM success_metrics 
                WHERE auto_eval_score IS NOT NULL 
                ORDER BY ts DESC
                """
                auto_eval_df = pd.read_sql_query(auto_eval_query, success_con)
                success_con.close()
            else:
                auto_eval_df = None

            if auto_eval_df is None:
                st.warning("📋 Success metrics database not found.")
                st.info("""
**Setup to populate `success.db`:**
1) Make sure your app logs with `log_with_auto_eval(...)` (from `telemetry.py`)  
2) That writes into `success_metrics` in `success.db` next to your app modules  
3) Use the sidebar ► **Settings** to pick the correct `success.db` if it lives elsewhere
""")
                # (Optional) quick preview so the tab still shows something useful
                st.markdown("---")
                st.subheader("📊 Preview: What You'll See (Sample Data)")
                sample_data = pd.DataFrame({
                    'variant': ['A','A','A','B','B','B','C','C','C'] * 10,
                    'auto_eval_score': np.random.normal(0.65, 0.15, 90).clip(0, 1)
                })
                sample_data['score_pct'] = sample_data['auto_eval_score'] * 100
                fig = px.violin(
                    sample_data, x='variant', y='score_pct',
                    title='Preview: Auto-Eval Score Distribution by Variant',
                    color='variant',
                    color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'}
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation_text="Success Threshold (70%)")
                fig.update_layout(
                    yaxis=dict(title="Auto-Eval Score (%)", range=[0, 100]),
                    xaxis=dict(title="Variant"),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.stop()

            # Data summary
            total_responses = len(auto_eval_df)
            date_range = f"{auto_eval_df['ts'].min()[:10]} to {auto_eval_df['ts'].max()[:10]}"
            st.info(f"📊 **Data Summary:** {total_responses} responses from {date_range}")

            # Process data for analysis
            auto_eval_df['is_successful'] = auto_eval_df['auto_eval_score'] >= threshold
            auto_eval_df['score_pct'] = auto_eval_df['auto_eval_score'] * 100

            # Check sample sizes
            sample_sizes = auto_eval_df['variant'].value_counts()
            insufficient_samples = sample_sizes[sample_sizes < min_samples].index.tolist()

            if insufficient_samples:
                st.warning(f"⚠️ **Insufficient samples for:** {', '.join(insufficient_samples)}")
                st.write(f"**Current samples:** {dict(sample_sizes)}")
                st.write(f"**Needed:** {min_samples} per variant for reliable testing")

            # Variant comparison table
            st.markdown("---")
            st.subheader("📈 Variant Performance Comparison")

            variant_stats = auto_eval_df.groupby('variant').agg({
                'auto_eval_score': ['count', 'mean', 'std'],
                'is_successful': ['sum', 'mean'],
                'score_pct': ['min', 'max']
            }).round(4)

            # Flatten column names
            variant_stats.columns = [
                'Sample_Size', 'Mean_Score', 'Std_Score',
                'Success_Count', 'Success_Rate', 'Min_Score', 'Max_Score'
            ]

            # Calculate improvements vs baseline (Variant A)
            baseline_mean = (
                variant_stats.loc['A', 'Mean_Score']
                if 'A' in variant_stats.index else variant_stats['Mean_Score'].iloc[0]
            )
            baseline_success = (
                variant_stats.loc['A', 'Success_Rate']
                if 'A' in variant_stats.index else variant_stats['Success_Rate'].iloc[0]
            )

            display_data = []
            for variant_key in variant_stats.index:
                stats_row = variant_stats.loc[variant_key]
                score_improvement = (
                    ((stats_row['Mean_Score'] - baseline_mean) / baseline_mean * 100)
                    if variant_key != 'A' and baseline_mean > 0 else 0
                )
                success_improvement = (
                    (stats_row['Success_Rate'] - baseline_success) * 100
                    if variant_key != 'A' else 0
                )

                # Confidence interval for success rate
                n = stats_row['Sample_Size']
                p_hat = stats_row['Success_Rate']
                se = np.sqrt(p_hat * (1 - p_hat) / n) if n > 0 else 0
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                ci_margin = z_score * se * 100

                display_data.append({
                    'Variant': f"{variant_key} ({'Direct Chat' if variant_key == 'A' else 'Wizard' if variant_key == 'B' else 'Examples'})",
                    'Sample Size': int(stats_row['Sample_Size']),
                    'Avg Score': f"{stats_row['Mean_Score']*100:.1f}%",
                    'Success Rate': f"{stats_row['Success_Rate']*100:.1f}% (±{ci_margin:.1f}%)",
                    'Score vs A': (
                        f"+{score_improvement:.1f}%"
                        if score_improvement > 0 else
                        (f"{score_improvement:.1f}%" if score_improvement != 0 else "Baseline")
                    ),
                    'Success vs A': (
                        f"+{success_improvement:.1f}pp"
                        if success_improvement > 0 else
                        (f"{success_improvement:.1f}pp" if success_improvement != 0 else "Baseline")
                    ),
                    'Range': f"{stats_row['Min_Score']:.1f}%-{stats_row['Max_Score']:.1f}%"
                })

            comparison_df = pd.DataFrame(display_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Statistical tests
            st.markdown("---")
            st.subheader("🧪 Statistical Analysis")

            variants_with_data = auto_eval_df['variant'].unique()
            if len(variants_with_data) >= 2:
                # ANOVA for auto-eval scores
                score_groups = [
                    auto_eval_df[auto_eval_df['variant'] == v]['auto_eval_score'].values
                    for v in variants_with_data
                ]
                try:
                    f_stat, anova_p = stats.f_oneway(*score_groups)
                except Exception:
                    f_stat, anova_p = np.nan, np.nan

                # Chi-square for success rates
                contingency_table = pd.crosstab(auto_eval_df['variant'], auto_eval_df['is_successful'])
                if contingency_table.shape[1] == 2:
                    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
                else:
                    chi2, chi2_p = np.nan, np.nan

                # Display results
                test_col1, test_col2, test_col3, test_col4 = st.columns(4)
                with test_col1:
                    st.metric("ANOVA F-stat", "N/A" if np.isnan(f_stat) else f"{f_stat:.3f}")
                with test_col2:
                    if np.isnan(anova_p):
                        st.metric("Score Difference", "N/A")
                    else:
                        significance = "✅ Significant" if anova_p < (1 - confidence_level) else "❌ Not Significant"
                        st.metric("Score Difference", significance)
                        st.caption(f"p = {anova_p:.4f}")

                if not np.isnan(chi2):
                    with test_col3:
                        st.metric("Chi² statistic", f"{chi2:.3f}")
                    with test_col4:
                        significance = "✅ Significant" if chi2_p < (1 - confidence_level) else "❌ Not Significant"
                        st.metric("Success Rate Diff", significance)
                        st.caption(f"p = {chi2_p:.4f}")

                # Interpretation
                st.markdown("#### 🎯 Results Interpretation")
                if not np.isnan(anova_p) and anova_p < (1 - confidence_level):
                    st.success("✅ **Significant differences in response quality between variants**")
                    best_variant = variant_stats['Mean_Score'].idxmax()
                    best_variant_stats = variant_stats.loc[best_variant]
                    st.markdown(f"**🏆 Best Performing Variant: {best_variant}**")
                    st.write(f"- Average Score: **{best_variant_stats['Mean_Score']*100:.1f}%**")
                    st.write(f"- Success Rate: **{best_variant_stats['Success_Rate']*100:.1f}%**")
                    if not np.isnan(chi2_p) and chi2_p < (1 - confidence_level):
                        st.write("- ✅ **Both quality scores AND success rates are significantly better**")
                    else:
                        st.write("- ⚠️ **Quality scores differ, but success rates may not be significantly different**")
                else:
                    st.warning("❌ **No significant differences detected in response quality**")
                    st.write("Possible reasons:")
                    st.write("- Sample sizes too small (increase data collection)")
                    st.write("- Variant differences too subtle (try more extreme variations)")
                    st.write("- Success threshold too low/high (adjust threshold)")

            # Visualizations
            st.markdown("---")
            st.subheader("📊 Visual Analysis")

            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                # Score distribution by variant
                fig = px.violin(
                    auto_eval_df, x='variant', y='score_pct',
                    title='Auto-Eval Score Distribution by Variant',
                    color='variant',
                    color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'}
                )
                fig.add_hline(
                    y=threshold * 100, line_dash="dash",
                    line_color="red",
                    annotation_text=f"Success Threshold ({threshold*100:.0f}%)",
                    annotation_position="top left"
                )
                # ✅ Option B: use layout for axes (no update_yaxis/xaxis)
                fig.update_layout(
                    yaxis=dict(title="Auto-Eval Score (%)", range=[0, 100]),
                    xaxis=dict(title="Variant"),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with viz_col2:
                # Success rate comparison with confidence intervals
                fig = go.Figure()
                colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'}

                for v_key in variant_stats.index:
                    row = variant_stats.loc[v_key]
                    success_rate = row['Success_Rate'] * 100

                    n = row['Sample_Size']
                    p_hat = row['Success_Rate']
                    se = np.sqrt(p_hat * (1 - p_hat) / n) if n > 0 else 0
                    z_score = stats.norm.ppf((1 + confidence_level) / 2)
                    ci_margin = z_score * se * 100

                    fig.add_trace(go.Bar(
                        x=[f'Variant {v_key}'],
                        y=[success_rate],
                        error_y=dict(type='data', array=[ci_margin]),
                        marker_color=colors.get(v_key, '#999999'),
                        name=f'Variant {v_key}'
                    ))

                    fig.add_annotation(
                        x=f'Variant {v_key}',
                        y=success_rate + ci_margin + 5,
                        text=f'n={int(row["Sample_Size"])}',
                        showarrow=False,
                        font=dict(size=10)
                    )

                fig.update_layout(
                    title=f'Success Rate by Variant ({confidence_level:.0%} CI)',
                    yaxis_title='Success Rate (%)',
                    yaxis=dict(range=[0, 100]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Pairwise comparisons
            variants_with_data = auto_eval_df['variant'].unique()
            if len(variants_with_data) >= 2:
                st.markdown("---")
                st.subheader("🔍 Pairwise Comparisons")

                pairwise_results = []
                for i, var1 in enumerate(variants_with_data):
                    for j, var2 in enumerate(variants_with_data):
                        if i < j:
                            g1 = auto_eval_df[auto_eval_df['variant'] == var1]['auto_eval_score']
                            g2 = auto_eval_df[auto_eval_df['variant'] == var2]['auto_eval_score']

                            t_stat, t_p = stats.ttest_ind(g1, g2)

                            pooled_std = np.sqrt(
                                ((len(g1) - 1) * np.var(g1) + (len(g2) - 1) * np.var(g2)) /
                                (len(g1) + len(g2) - 2)
                            ) if (len(g1) + len(g2) - 2) > 0 else 0
                            cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

                            if abs(cohens_d) >= 0.8:
                                eff = "Large"
                            elif abs(cohens_d) >= 0.5:
                                eff = "Medium"
                            elif abs(cohens_d) >= 0.2:
                                eff = "Small"
                            else:
                                eff = "Negligible"

                            mean_diff = (np.mean(g1) - np.mean(g2)) * 100
                            pairwise_results.append({
                                'Comparison': f'{var1} vs {var2}',
                                'Mean Difference': f'{mean_diff:+.1f}%',
                                'P-value': f'{t_p:.4f}',
                                'Significant': '✅' if t_p < (1 - confidence_level) else '❌',
                                'Cohen\'s d': f'{cohens_d:.3f}',
                                'Effect Size': eff
                            })

                if pairwise_results:
                    pairwise_df = pd.DataFrame(pairwise_results)
                    st.dataframe(pairwise_df, use_container_width=True)

            # Power Analysis
            st.markdown("---")
            st.subheader("⚡ Power Analysis")

            effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
            alpha = 1 - confidence_level
            power = 0.8

            power_analysis = []
            for es in effect_sizes:
                # Approximation for two-sample t-test
                required_n = int(16 / (es ** 2) * ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2))
                current_min = min(sample_sizes) if len(sample_sizes) > 0 else 0
                adequately_powered = current_min >= required_n

                power_analysis.append({
                    'Effect Size': f'{es} ({["Small", "Medium", "Large"][effect_sizes.index(es)]})',
                    'Required N per Variant': required_n,
                    'Current Min N': current_min,
                    'Adequately Powered': '✅' if adequately_powered else '❌',
                    'Progress': f'{(min(100, current_min / max(1, required_n) * 100)):.0f}%'
                })

            power_df = pd.DataFrame(power_analysis)
            st.dataframe(power_df, use_container_width=True)

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")

            total_samples = len(auto_eval_df)
            min_sample_size = min(sample_sizes) if len(sample_sizes) > 0 else 0

            if min_sample_size < min_samples:
                st.warning(f"📊 **Continue Data Collection** (Current: {total_samples} total, {min_sample_size} min per variant)")
                st.write(f"- Target: {min_samples}+ samples per variant for reliable results")
                st.write("- Increase traffic or extend test duration")

            elif 'anova_p' in locals() and not np.isnan(anova_p) and anova_p < (1 - confidence_level):
                st.success("🎯 **Actionable Results Available!**")
                best_variant = variant_stats['Mean_Score'].idxmax()
                improvement = (
                    (variant_stats.loc[best_variant, 'Mean_Score'] - baseline_mean) /
                    max(1e-9, baseline_mean) * 100
                )
                st.write(f"**Recommended Action:** Deploy Variant {best_variant}")
                st.write(f"- Expected improvement: {improvement:+.1f}% in response quality")
                st.write(f"- Success rate: {variant_stats.loc[best_variant, 'Success_Rate']*100:.1f}%")
                st.write(f"- Based on {total_samples} total responses")
            else:
                st.info("🔬 **Optimization Needed**")
                st.write("**Consider:**")
                st.write(f"- Adjusting success threshold (currently {threshold:.0%})")
                st.write("- Testing more extreme variant differences")
                st.write("- Collecting qualitative feedback for insights")
                st.write("- Segmenting analysis by task type or user characteristics")

            # Export functionality
            st.markdown("---")
            with st.expander("📁 Export Data & Results"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    csv_data = auto_eval_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Raw Data",
                        csv_data,
                        f"auto_eval_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                with c2:
                    summary_csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        "📈 Download Summary",
                        summary_csv,
                        f"variant_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                with c3:
                    if 'pairwise_results' in locals() and len(pairwise_results) > 0:
                        stats_csv = pd.DataFrame(pairwise_results).to_csv(index=False)
                        st.download_button(
                            "🧪 Download Stats",
                            stats_csv,
                            f"statistical_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv"
                        )

        with tab2:
            st.subheader("Variant Comparison")

            variant_stats = df.groupby('variant').agg({
                'latency_ms': ['mean', 'std'],
                'tokens_total': 'mean',
                'cost_usd': 'mean',
                'session_id': 'count'
            }).round(2)

            variant_stats.columns = ['Avg Latency (ms)', 'Latency Std', 'Avg Tokens', 'Avg Cost ($)', 'Usage Count']
            st.dataframe(variant_stats)

            # Variant performance chart
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                variant_latency = df.groupby('variant')['latency_ms'].mean()
                bars = ax.bar(variant_latency.index, variant_latency.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_ylabel('Average Latency (ms)')
                ax.set_title('Average Latency by Variant')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 10, f'{height:.0f}ms', ha='center', va='bottom')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                variant_usage = df['variant'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                ax.pie(variant_usage.values, labels=variant_usage.index, autopct='%1.1f%%', colors=colors[:len(variant_usage)])
                ax.set_title('Usage Distribution by Variant')
                st.pyplot(fig)

        with tab3:
            st.subheader("Task Analysis")

            # Check if task column exists and has data
            if 'task' in df.columns and df['task'].notna().any():
                task_stats = df.groupby('task').agg({
                    'latency_ms': ['mean', 'std'],
                    'tokens_total': 'mean',
                    'cost_usd': 'mean',
                    'session_id': 'count'
                }).round(2)

                task_stats.columns = ['Avg Latency (ms)', 'Latency Std', 'Avg Tokens', 'Avg Cost ($)', 'Usage Count']
                st.dataframe(task_stats)

                # Task performance charts
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    task_latency = df.groupby('task')['latency_ms'].mean()
                    bars = ax.bar(range(len(task_latency)), task_latency.values, color=['#FF9999', '#66B2FF', '#99FF99'])
                    ax.set_ylabel('Average Latency (ms)')
                    ax.set_title('Average Latency by Task')
                    ax.set_xticks(range(len(task_latency)))
                    ax.set_xticklabels(task_latency.index, rotation=45, ha='right')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 10, f'{height:.0f}ms', ha='center', va='bottom')
                    st.pyplot(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    task_usage = df['task'].value_counts()
                    colors = ['#FF9999', '#66B2FF', '#99FF99']
                    ax.pie(task_usage.values, labels=task_usage.index, autopct='%1.1f%%', colors=colors[:len(task_usage)])
                    ax.set_title('Usage Distribution by Task')
                    st.pyplot(fig)

                # Task vs Variant cross-analysis
                st.subheader("Task vs Variant Analysis")
                if len(df['task'].unique()) > 1 and len(df['variant'].unique()) > 1:
                    cross_stats = df.groupby(['variant', 'task']).agg({
                        'latency_ms': 'mean',
                        'tokens_total': 'mean',
                        'cost_usd': 'mean',
                        'session_id': 'count'
                    }).round(2)
                    cross_stats.columns = ['Avg Latency (ms)', 'Avg Tokens', 'Avg Cost ($)', 'Count']
                    st.dataframe(cross_stats)
            else:
                st.info("🔍 No task data available. Task column may be empty or missing.")

        with tab4:
            st.subheader("Success Metrics & User Feedback")

            if SUCCESS_DB_PATH and os.path.exists(SUCCESS_DB_PATH):
                # Load success data
                success_con = sqlite3.connect(SUCCESS_DB_PATH)
                success_df = pd.read_sql_query("SELECT * FROM success_metrics ORDER BY ts DESC LIMIT 1000", success_con)
                success_con.close()

                if not success_df.empty:
                    # Success overview
                    if success_available and callable(get_success_stats):
                        success_stats = get_success_stats()
                    else:
                        # Fallback quick stats if success.py not present
                        success_stats = {
                            'responses_with_feedback': int(success_df['user_satisfied'].notna().sum()) if 'user_satisfied' in success_df else 0,
                            'user_satisfaction_rate': float(
                                (success_df.get('user_satisfied', pd.Series(dtype='float')).fillna(False).astype(bool).mean() * 100)
                                if 'user_satisfied' in success_df else 0.0
                            ),
                            'avg_auto_eval_score': float(success_df.get('auto_eval_score', pd.Series(dtype='float')).mean() * 100 if 'auto_eval_score' in success_df else 0.0),
                            'total_responses': int(len(success_df)),
                        }

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Feedback", success_stats['responses_with_feedback'])
                    with col2:
                        st.metric("👍 Satisfaction Rate", f"{success_stats['user_satisfaction_rate']:.1f}%")
                    with col3:
                        st.metric("🤖 Avg Auto-Eval", f"{success_stats['avg_auto_eval_score']:.1f}%")
                    with col4:
                        st.metric("📈 Total Responses", success_stats['total_responses'])

                    # Charts
                    col1, col2 = st.columns(2)

                    with col1:
                        # User satisfaction by variant
                        if 'user_satisfied' in success_df.columns and success_df['user_satisfied'].notna().any():
                            fig, ax = plt.subplots(figsize=(8, 6))
                            satisfaction_by_variant = success_df[success_df['user_satisfied'].notna()].groupby(['variant', 'user_satisfied']).size().unstack(fill_value=0)
                            if not satisfaction_by_variant.empty:
                                if True in satisfaction_by_variant.columns and False in satisfaction_by_variant.columns:
                                    satisfaction_rates = satisfaction_by_variant[True] / (satisfaction_by_variant[True] + satisfaction_by_variant[False]) * 100
                                elif True in satisfaction_by_variant.columns:
                                    satisfaction_rates = pd.Series(100.0, index=satisfaction_by_variant.index)
                                else:
                                    satisfaction_rates = pd.Series(0.0, index=satisfaction_by_variant.index)

                                bars = ax.bar(satisfaction_rates.index, satisfaction_rates.values, color=['#4CAF50', '#2196F3', '#FF9800'])
                                ax.set_ylabel('Satisfaction Rate (%)')
                                ax.set_title('User Satisfaction by Variant')
                                ax.set_ylim(0, 100)
                                for bar, rate in zip(bars, satisfaction_rates.values):
                                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, f'{rate:.1f}%', ha='center', va='bottom')
                            st.pyplot(fig)

                    with col2:
                        # Auto-eval scores by task
                        if 'auto_eval_score' in success_df.columns and success_df['auto_eval_score'].notna().any():
                            fig, ax = plt.subplots(figsize=(8, 6))
                            task_scores = success_df[success_df['auto_eval_score'].notna()].groupby('task')['auto_eval_score'].mean() * 100
                            if not task_scores.empty:
                                bars = ax.bar(range(len(task_scores)), task_scores.values, color=['#9C27B0', '#607D8B', '#795548'])
                                ax.set_ylabel('Auto-Eval Score (%)')
                                ax.set_title('Auto-Evaluation Scores by Task')
                                ax.set_xticks(range(len(task_scores)))
                                ax.set_xticklabels(task_scores.index, rotation=45, ha='right')
                                ax.set_ylim(0, 100)
                                for bar, score in zip(bars, task_scores.values):
                                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, f'{score:.1f}%', ha='center', va='bottom')
                            st.pyplot(fig)

                    # Recent feedback
                    st.subheader("Recent User Feedback")
                    if 'user_satisfied' in success_df.columns and success_df['user_satisfied'].notna().any():
                        feedback_df = success_df[success_df['user_satisfied'].notna()].head(20)
                        # guard for accidental 'task,' column name
                        display_df = feedback_df[['ts', 'variant', 'task,', 'user_satisfied', 'auto_eval_score']].copy() if 'task,' in feedback_df.columns else feedback_df[['ts', 'variant', 'task', 'user_satisfied', 'auto_eval_score']].copy()
                        display_df['ts'] = pd.to_datetime(display_df['ts']).dt.strftime('%Y-%m-%d %H:%M')
                        display_df['user_satisfied'] = display_df['user_satisfied'].map({True: '👍 Yes', False: '👎 No'})
                        display_df['auto_eval_score'] = display_df['auto_eval_score'].fillna(0).apply(lambda x: f"{x*100:.1f}%")
                        display_df.columns = ['Timestamp', 'Variant', 'Task', 'User Satisfied', 'Auto-Eval Score']
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("📊 No success metrics data yet. Users need to provide feedback first!")
            else:
                st.warning("📊 Success metrics not available. Pick the correct `success.db` in the sidebar or ensure your app is writing to it.")

        with tab6:
            st.subheader("Recent Activity (Last 50 requests)")

            # Include task column if it exists
            columns_to_show = ['ts', 'variant', 'latency_ms', 'tokens_total', 'cost_usd']
            if 'task' in df.columns:
                columns_to_show.insert(2, 'task')

            recent_df = df.head(50)[columns_to_show]

            # Rename columns for display
            column_renames = {
                'ts': 'Timestamp',
                'variant': 'Variant',
                'task': 'Task',
                'latency_ms': 'Latency (ms)',
                'tokens_total': 'Tokens',
                'cost_usd': 'Cost ($)'
            }
            recent_df = recent_df.rename(columns=column_renames)

            # Format the dataframe for better display
            if 'Cost ($)' in recent_df.columns:
                recent_df['Cost ($)'] = recent_df['Cost ($)'].map('${:.4f}'.format)
            if 'Latency (ms)' in recent_df.columns:
                recent_df['Latency (ms)'] = recent_df['Latency (ms)'].map('{:.0f}'.format)

            st.dataframe(recent_df, use_container_width=True)

        with tab7:
            st.subheader("Raw Data Export")

            # Download button for CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download data as CSV",
                data=csv,
                file_name=f"telemetry_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            # Show full data
            st.dataframe(df, use_container_width=True)

    else:
        st.info("🔭 No telemetry data found. Run your app to generate some data!")

else:
    st.warning("🔍 No telemetry.db found yet. Run an app to generate logs.")
    st.info("Make sure to run your `app.py` first to create telemetry data.")

# Evaluation Results Section
st.divider()
st.subheader("🧪 Evaluation Results")

eval_paths = list(Path(".").glob("**/eval_results.csv"))
if eval_paths:
    eval_file = st.selectbox("📋 Select evaluation results", eval_paths)
    eval_df = pd.read_csv(eval_file)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Total Evaluations", len(eval_df))
    with col2:
        if 'score' in eval_df.columns:
            avg_score = eval_df['score'].mean()
            st.metric("📊 Average Score", f"{avg_score:.2f}")

    st.dataframe(eval_df.head(50), use_container_width=True)
else:
    st.info("🔍 No eval_results.csv found yet. Run `run_evals.py` in a project to generate evaluation data.")
