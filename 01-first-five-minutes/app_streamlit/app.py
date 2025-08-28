import os, time, uuid, re
import streamlit as st
from telemetry import log, log_journey_event, log_with_auto_eval  # includes log_with_auto_eval
import sys
import importlib
from datetime import datetime
import sqlite3
from telemetry import log, log_journey_event, log_with_auto_eval # includes log_with_auto_eval

# Try to import metrics from either `evals.metrics` (package style) or `metrics` (flat file)
METRICS = None
for modname in ("evals.metrics", "metrics"):
    try:
        METRICS = importlib.import_module(modname)
        break
    except Exception:
        pass

st.set_page_config(page_title="Plan A Rocking Holiday!", layout="wide")
st.title("Plan A Rocking Holiday!")
st.caption(f"Auto-eval module loaded: {bool(METRICS)}")

from dotenv import load_dotenv
load_dotenv()

# -----------------------
# Project paths & DBs
# -----------------------
# Get absolute path to the 'common' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
common_path = os.path.join(project_root, "common")
if common_path not in sys.path:
    sys.path.append(common_path)

# DB paths (stored alongside app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
success_db = os.path.join(current_dir, "success_metrics.db")
telemetry_db = os.path.join(current_dir, "telemetry.db")
user_journey_db = os.path.join(current_dir, "user_journey.db")

# Import from common
from model_clients import llm_complete  # type: ignore
from success import log_success_metrics, get_task_keyphrases, score_example

# -----------------------
# Session bootstrap
# -----------------------
if "sid" not in st.session_state:
    st.session_state["sid"] = str(uuid.uuid4())
    st.session_state["session_start_time"] = datetime.utcnow().isoformat()
    st.session_state["variant_selected"] = False

# persistent plan + refinement state
for k, v in {
    "current_plan": None,          # latest plan text to revise
    "current_details": "",         # original request / assembled details
    "current_task": None,          # variant code (A/B/C)
    "change_round": 0,             # increments after every refine
    "active_variant": None,        # track switching A/B/C
    "just_generated": False,       # 🔸 skip double-render on the run we just generated/refined
}.items():
    st.session_state.setdefault(k, v)

sid = st.session_state["sid"]
session_start_time = st.session_state["session_start_time"]

variant = st.selectbox("Try out a holiday planner", ["A - Chat Directly","B - Wizard Planner","C - Sample Itinerary"])
variant_code = variant.split(" ")[0]  # "A", "B", or "C"

# If variant changed, clear current plan context so we can start fresh
if st.session_state.get("active_variant") != variant_code:
    st.session_state["active_variant"] = variant_code
    st.session_state["current_plan"] = None
    st.session_state["current_details"] = ""
    st.session_state["current_task"] = variant_code
    st.session_state["change_round"] = 0
    st.session_state["just_generated"] = False

# Log session start and variant selection once
if not st.session_state.get("variant_selected", False):
    log_journey_event(sid, variant_code, "session_start", {
        "selected_variant": variant,
        "timestamp": session_start_time
    })
    st.session_state["variant_selected"] = True

# -----------------------
# Helpers
# -----------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 word ≈ 1.33 tokens."""
    words = len((text or "").split())
    return int(words * 1.33)

def compute_auto_eval(task: str, response_text: str, keyphrases: list[str]) -> dict:
    """Auto-eval with fallback."""
    # Best path: use custom metrics if provided
    if METRICS and hasattr(METRICS, "evaluate_response"):
        try:
            res = METRICS.evaluate_response(task, response_text)
            required = {"keyphrase_coverage", "response_length", "has_structure", "has_specifics", "overall_score"}
            if isinstance(res, dict) and required.issubset(res.keys()):
                return res
        except Exception:
            pass

    # Fallback: simple components
    try:
        if METRICS and hasattr(METRICS, "contains_keyphrases"):
            coverage = float(METRICS.contains_keyphrases(response_text, keyphrases))
        else:
            coverage = float(score_example(response_text, keyphrases))
    except Exception:
        coverage = float(score_example(response_text, keyphrases))

    response_length = len((response_text or "").split())
    has_structure = bool(re.search(r'(Day\s+\d+|Morning|Afternoon|Evening|\n\d+\.)', response_text or ""))
    has_specifics = bool(re.search(r'(\$|€|£)\s?\d+|\d{1,2}:\d{2}|\b\d+\s+(hours?|days?)\b', response_text or ""))
    overall = (coverage + (0.1 if has_structure else 0) + (0.1 if has_specifics else 0)) / 1.2
    overall = max(0.0, min(1.0, overall))
    return {
        "keyphrase_coverage": coverage,
        "response_length": response_length,
        "has_structure": has_structure,
        "has_specifics": has_specifics,
        "overall_score": overall,
    }

def show_feedback_section(session_id: str, variant_str: str, task: str, response_text: str):
    """Feedback widget + immediate auto-eval logging for hypothesis testing."""
    st.markdown("---")
    st.subheader("📝 Help us improve!")
    keyphrases = get_task_keyphrases(task) or []
    eval_res = compute_auto_eval(task, response_text, keyphrases)

    # log auto-eval snapshot (0ms latency here; main call logs real latency too)
    log_with_auto_eval(
        session_id=session_id,
        variant=variant_str,
        latency_ms=0,
        tokens_user=estimate_tokens(task),
        tokens_assistant=estimate_tokens(response_text),
        auto_eval_score=eval_res['overall_score'],
        task=task,
        session_start_ts=st.session_state.get("session_start_time")
    )

    auto_eval_success = eval_res['overall_score'] >= st.session_state.get("auto_eval_threshold", 0.7)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    user_satisfied = None
    task_completed = None

    with col4:
        auto_score_pct = eval_res['overall_score'] * 100
        if auto_score_pct >= 80:
            st.metric("🎯 Auto-Eval", f"{auto_score_pct:.0f}%", delta="Excellent")
        elif auto_score_pct >= 70:
            st.metric("🎯 Auto-Eval", f"{auto_score_pct:.0f}%", delta="Good")
        elif auto_score_pct >= 50:
            st.metric("🎯 Auto-Eval", f"{auto_score_pct:.0f}%", delta="Fair") 
        else:
            st.metric("🎯 Auto-Eval", f"{auto_score_pct:.0f}%", delta="Needs Work")
        with st.expander("📊 Auto-Eval Details"):
            st.write(f"**Keyphrase coverage:** {eval_res['keyphrase_coverage']*100:.0f}%")
            st.write(f"**Response length:** {eval_res['response_length']} words")
            st.write(f"**Has structure:** {'✅' if eval_res['has_structure'] else '❌'}")
            st.write(f"**Has specifics:** {'✅' if eval_res['has_specifics'] else '❌'}")
            st.write(f"**Success threshold ({int(st.session_state.get('auto_eval_threshold',0.7)*100)}%):** {'✅ Met' if auto_eval_success else '❌ Not met'}")

    st.write("**Did this help you finish your task?**")
    fb_key = f"feedback_{session_id}_{int(time.time())}"

    with col1:
        if st.button("🎉 Perfect! Task Complete", key=f"{fb_key}_complete"):
            user_satisfied = True
            task_completed = True
            log_journey_event(session_id, variant_str, "task_complete", {
                "feedback": "complete",
                "auto_eval_score": eval_res["overall_score"],
                "auto_eval_above_threshold": auto_eval_success
            })
    with col2:
        if st.button("👍 Helpful, but need more", key=f"{fb_key}_helpful"):
            user_satisfied = True
            task_completed = False
            log_journey_event(session_id, variant_str, "partial_success", {
                "feedback": "helpful_incomplete",
                "auto_eval_score": eval_res["overall_score"]
            })
    with col3:
        if st.button("👎 Not helpful", key=f"{fb_key}_no"):
            user_satisfied = False
            task_completed = False
            log_journey_event(session_id, variant_str, "task_failure", {
                "feedback": "not_helpful",
                "auto_eval_score": eval_res["overall_score"]
            })

    if user_satisfied is None:
        if auto_eval_success:
            st.success(f"🎯 Auto-eval suggests this is high quality ({auto_score_pct:.0f}%).")
        else:
            st.warning(f"⚠️ Auto-eval suggests this could be improved ({auto_score_pct:.0f}%).")

    if user_satisfied is not None:
        log_success_metrics(
            session_id=session_id,
            variant=variant_str,
            task=task,
            response_text=response_text,
            user_satisfied=user_satisfied,
            keyphrases=keyphrases
        )
        st.session_state[f"task_completed_{session_id}"] = task_completed
        if task_completed:
            st.success("✨ Excellent! Thanks for confirming your task is complete.")
        elif user_satisfied:
            st.info("👍 Thanks! Feel free to ask follow-ups below.")
        else:
            st.info("💡 Thanks for the feedback! Try refining below.")

    return user_satisfied, auto_eval_success

# ---------- NEW: unified refinement UI  ----------
def build_refinement_prompt(original_details: str, current_plan: str, change_request: str) -> str:
    return (
        "You previously drafted a holiday plan.\n\n"
        "Original request/details:\n"
        f"{original_details}\n\n"
        "Current plan to revise:\n"
        f"{current_plan}\n\n"
        "Change request from user:\n"
        f"{change_request}\n\n"
        "Please produce an UPDATED plan:\n"
        "• Keep a clear day-by-day structure (Morning/Afternoon/Evening where relevant)\n"
        "• Preserve good parts that still fit; only change what’s requested\n"
        "• If you add costs/times, keep them realistic and consistent\n"
        "• At the top, include a short bullet list: “Changes applied”\n"
    )

def show_change_requests(session_id: str, variant_str: str, task: str):
    """Shown under any generated plan. Supports multiple refinement rounds."""
    if not st.session_state.get("current_plan"):
        return

    st.markdown("### ✏️ Ask for changes")
    with st.expander("Describe what you want to change", expanded=True):
        cr_key = f"change_req_{st.session_state['change_round']}"
        change_req = st.text_area(
            "What should we change?",
            key=cr_key,
            placeholder="e.g., Make Day 2 more kid-friendly, add budget estimates, and swap the museum for a boat tour.",
            height=120,
        )
        c1, c2 = st.columns([1,1])
        with c1:
            apply_clicked = st.button("✅ Apply changes", type="primary", key=f"apply_{cr_key}")
        with c2:
            restart_clicked = st.button("🗑️ Start a new plan", key=f"restart_{cr_key}")

    if restart_clicked:
        # Clear plan/refinement state; keep session + variant
        st.session_state["current_plan"] = None
        st.session_state["current_details"] = ""
        st.session_state["change_round"] = 0
        st.session_state["just_generated"] = False
        st.info("Start a new plan by generating again.")
        st.stop()

    if apply_clicked and change_req.strip():
        # Log refine request
        log_journey_event(session_id, variant_str, "refine_request", {
            "round": st.session_state["change_round"] + 1,
            "chars": len(change_req)
        })

        # Build prompt to refine existing plan
        prompt = build_refinement_prompt(
            st.session_state.get("current_details", ""),
            st.session_state.get("current_plan", ""),
            change_req
        )

        t0 = time.time()
        try:
            new_out = llm_complete(prompt)
        except Exception as e:
            st.error(f"Refinement failed: {e}")
            return
        dt = int((time.time() - t0) * 1000)

        # Show updated plan
        st.subheader("✅ Updated Plan")
        st.code(new_out)

        # Log tokens + auto-eval for this refinement
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(new_out)

        keyphrases = get_task_keyphrases(task) or []
        eval_res = compute_auto_eval(task, new_out, keyphrases)

        log_with_auto_eval(
            session_id=session_id,
            variant=variant_str,
            latency_ms=dt,
            tokens_user=input_tokens,
            tokens_assistant=output_tokens,
            auto_eval_score=eval_res['overall_score'],
            task=task,
            session_start_ts=st.session_state.get("session_start_time")
        )

        log_journey_event(session_id, variant_str, "refine_complete", {
            "round": st.session_state["change_round"] + 1,
            "latency_ms": dt,
            "auto_eval_score": eval_res['overall_score']
        })

        # Update current plan + round and offer another refinement (loop)
        st.session_state["current_plan"] = new_out
        st.session_state["change_round"] += 1

        # 🔸 Mark that we just generated to prevent double-render in persistent zone this run
        st.session_state["just_generated"] = True

        # Optional: ask feedback again on the refined plan
        show_feedback_section(session_id, variant_str, task, new_out)

# -----------------------
# Variants
# -----------------------

# --- Option B: Wizard Planner 3-Step Workflow ---
if variant.startswith("B"):
    if "step" not in st.session_state:
        st.session_state.step = 1
        st.session_state.step_data = {}
        log_journey_event(sid, variant_code, "wizard_start")

    if st.session_state.step == 1:
        st.subheader("🗺️ Step 1 of 3: Choose Your Destination")
        destination = st.text_input("Enter your dream destination:")
        if st.button("Next →", key="wizard_step1"):
            if destination:
                st.session_state.step_data["destination"] = destination
                st.session_state.step = 2
                log_journey_event(sid, variant_code, "step_complete",
                                  {"step": 1, "destination": destination}, step_number=1)
                st.rerun()
            else:
                st.warning("Please enter a destination to continue.")

    elif st.session_state.step == 2:
        st.subheader("🎯 Step 2 of 3: Plan Your Activities")
        st.info(f"Destination: **{st.session_state.step_data.get('destination', '')}**")
        activities = st.text_area("List activities you want to do:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="wizard_back_2"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Next →", key="wizard_step2"):
                if activities:
                    st.session_state.step_data["activities"] = activities
                    st.session_state.step = 3
                    log_journey_event(sid, variant_code, "step_complete",
                                      {"step": 2, "activities": activities}, step_number=2)
                    st.rerun()
                else:
                    st.warning("Please describe your desired activities.")

    elif st.session_state.step == 3:
        st.subheader("⏰ Step 3 of 3: Set Your Duration")
        st.info(f"**Destination:** {st.session_state.step_data.get('destination', '')}")
        st.info(f"**Activities:** {st.session_state.step_data.get('activities', '')}")
        duration = st.text_input("How many days will your holiday be?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="wizard_back_3"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("🎉 Generate My Itinerary!", key="wizard_step3", type="primary"):
                if duration:
                    st.session_state.step_data["duration"] = duration
                    log_journey_event(sid, variant_code, "step_complete",
                                      {"step": 3, "duration": duration}, step_number=3)

                    # Compose details for LLM
                    details = (
                        f"Destination: {st.session_state.step_data['destination']}\n"
                        f"Activities: {st.session_state.step_data['activities']}\n"
                        f"Duration: {st.session_state.step_data['duration']} days"
                    )
                    task = variant_code
                    log_journey_event(sid, variant_code, "generation_start",
                                      {"input_complexity": len(details.split())})

                    t0 = time.time()
                    out = llm_complete(details)
                    dt = int((time.time() - t0) * 1000)

                    st.subheader("🎉 Your Personalized Itinerary")
                    st.code(out)

                    # Tokens + logging
                    input_tokens = estimate_tokens(details)
                    output_tokens = estimate_tokens(out)
                    keyphrases = get_task_keyphrases(task) or []
                    eval_res = compute_auto_eval(task, out, keyphrases)

                    log_with_auto_eval(
                        session_id=sid,
                        variant=variant_code,
                        latency_ms=dt,
                        tokens_user=input_tokens,
                        tokens_assistant=output_tokens,
                        auto_eval_score=eval_res['overall_score'],
                        task=task,
                        session_start_ts=session_start_time
                    )

                    # Save plan for iterative refinements
                    st.session_state["current_plan"] = out
                    st.session_state["current_details"] = details
                    st.session_state["current_task"] = task
                    st.session_state["change_round"] = 0
                    st.session_state["just_generated"] = True  # 🔸

                    # Feedback + Changes
                    show_feedback_section(sid, variant_code, task, out)
                    show_change_requests(sid, variant_code, task)

                    # 🔁 Do NOT auto-reset wizard here; user can click this to start fresh
                    st.markdown("---")
                    if st.button("🗑️ Start a new wizard plan"):
                        st.session_state.step = 1
                        st.session_state.step_data = {}
                        st.session_state.current_plan = None
                        st.session_state.current_details = ""
                        st.session_state.change_round = 0
                        st.session_state.just_generated = False
                        st.rerun()
                else:
                    st.warning("Please specify the duration of your trip.")

# --- Option A: Direct Chat ---
elif variant.startswith("A"):
    st.subheader("💬 Direct Chat Planning")
    details = st.text_area("Details", "Paste text or describe holiday of your dreams...", key="text_plain_chat")

    if st.button("🚀 Plan My Holiday", key="run_a", type="primary"):
        if details.strip():
            task = variant_code
            log_journey_event(sid, variant_code, "generation_start",
                              {"input_complexity": len(details.split())})

            t0 = time.time()
            out = llm_complete(details)
            dt = int((time.time() - t0) * 1000)

            st.subheader("🎯 Your Holiday Plan")
            st.code(out)

            input_tokens = estimate_tokens(details)
            output_tokens = estimate_tokens(out)
            keyphrases = get_task_keyphrases(task) or []
            eval_res = compute_auto_eval(task, out, keyphrases)

            log_with_auto_eval(
                session_id=sid,
                variant=variant_code,
                latency_ms=dt,
                tokens_user=input_tokens,
                tokens_assistant=output_tokens,
                auto_eval_score=eval_res['overall_score'],
                task=task,
                session_start_ts=session_start_time
            )

            # Save plan for refinements
            st.session_state["current_plan"] = out
            st.session_state["current_details"] = details
            st.session_state["current_task"] = task
            st.session_state["change_round"] = 0
            st.session_state["just_generated"] = True  # 🔸

            # Feedback + Changes
            show_feedback_section(sid, variant_code, task, out)
            show_change_requests(sid, variant_code, task)
        else:
            st.warning("Please describe your holiday plans first!")

# --- Option C: Example Gallery ---
elif variant.startswith("C"):
    # Track example interactions once
    if "c_interaction_started" not in st.session_state:
        log_journey_event(sid, variant_code, "gallery_view_start")
        st.session_state["c_interaction_started"] = True

    main_col, sidebar_col = st.columns([2, 1])

    examples = [
        {"title": "Maldives", "description": "Luxury overwater villas",
         "prompt": "Plan a 7-day luxury getaway to the Maldives. Include overwater villa stays, snorkeling excursions, spa treatments, and romantic sunset dining experiences.", "emoji": "🏖️"},
        {"title": "Japan", "description": "Cultural temples & cuisine",
         "prompt": "Create a 10-day cultural tour of Japan covering Tokyo, Kyoto, and Osaka. Include temple visits, traditional ryokan stays, sushi experiences, and cherry blossom viewing.", "emoji": "🗾"},
        {"title": "Swiss Alps", "description": "Mountain adventures",
         "prompt": "Design an 8-day adventure trip to the Swiss Alps. Include hiking trails, mountain railway rides, stays in alpine chalets, and outdoor activities like paragliding.", "emoji": "🏔️"},
        {"title": "Kenya Safari", "description": "Wildlife & culture",
         "prompt": "Plan a 9-day safari adventure in Kenya. Include Masai Mara game drives, cultural village visits, hot air balloon rides, and luxury safari lodge accommodations.", "emoji": "🦁"},
        {"title": "Greece", "description": "History & islands",
         "prompt": "Create a 12-day historical and island-hopping tour of Greece. Include Athens ancient sites, Santorini sunsets, Mykonos beaches, and traditional Greek taverna experiences.", "emoji": "🏛️"},
    ]

    with sidebar_col:
        st.subheader("🌍 Examples")
        for i, example in enumerate(examples):
            button_key = f"example_{i}"
            button_content = f"{example['emoji']} **{example['title']}**\n\n{example['description']}"
            if st.button(button_content, key=button_key, use_container_width=True):
                st.session_state.selected_example_prompt = example['prompt']
                st.session_state.custom_mode = False
                log_journey_event(sid, variant_code, "example_selected",
                                  {"example_title": example['title'], "example_index": i})
                st.rerun()

    with main_col:
        st.subheader("✨ Create Your Holiday Plan")
        if st.session_state.get("selected_example_prompt") and not st.session_state.get("custom_mode", False):
            st.success("📋 Example selected! You can modify the details below or generate directly.")
            details = st.text_area("Trip Details:", value=st.session_state.selected_example_prompt,
                                   height=150, key="selected_prompt")
            if details != st.session_state.selected_example_prompt:
                if not st.session_state.get("example_modified", False):
                    log_journey_event(sid, variant_code, "example_modified")
                    st.session_state["example_modified"] = True
            if st.button("🔄 Switch to Custom Mode"):
                st.session_state.custom_mode = True
                st.session_state.selected_example_prompt = None
                log_journey_event(sid, variant_code, "switch_to_custom")
                st.rerun()
        else:
            details = st.text_area(
                "Describe your dream destination:",
                placeholder="Tell us about your perfect holiday...",
                height=150, key="custom_prompt"
            )
            if st.button("📋 Use Example Instead") and st.session_state.get("selected_example_prompt"):
                st.session_state.custom_mode = False
                log_journey_event(sid, variant_code, "switch_to_example")
                st.rerun()

        if st.button("🎉 Generate Itinerary", key="run_c", type="primary", use_container_width=True):
            if details.strip():
                task = variant_code
                log_journey_event(sid, variant_code, "generation_start", {
                    "input_complexity": len(details.split()),
                    "used_example": bool(st.session_state.get("selected_example_prompt")),
                    "modified_example": st.session_state.get("example_modified", False)
                })

                t0 = time.time()
                out = llm_complete(details)
                dt = int((time.time() - t0) * 1000)

                st.subheader("🎉 Your Personalized Itinerary")
                st.code(out)

                input_tokens = estimate_tokens(details)
                output_tokens = estimate_tokens(out)
                keyphrases = get_task_keyphrases(task) or []
                eval_res = compute_auto_eval(task, out, keyphrases)

                log_with_auto_eval(
                    session_id=sid,
                    variant=variant_code,
                    latency_ms=dt,
                    tokens_user=input_tokens,
                    tokens_assistant=output_tokens,
                    auto_eval_score=eval_res['overall_score'],
                    task=task,
                    session_start_ts=session_start_time
                )

                # Save plan for refinements
                st.session_state["current_plan"] = out
                st.session_state["current_details"] = details
                st.session_state["current_task"] = task
                st.session_state["change_round"] = 0
                st.session_state["just_generated"] = True  # 🔸

                # Feedback + Changes
                show_feedback_section(sid, variant_code, task, out)
                show_change_requests(sid, variant_code, task)
            else:
                st.warning("Please select an example or describe your own destination!")

# -----------------------
# 🔁 Persistent refinement zone (all variants)
# -----------------------
if st.session_state.get("current_plan"):
    # If we *just* generated/refined, the variant block already showed the plan + change box.
    # Skip once to avoid double-render, then show persistently on subsequent reruns.
    if not st.session_state.pop("just_generated", False):
        st.markdown("## 📝 Current Plan")
        st.code(st.session_state["current_plan"])
        show_change_requests(
            st.session_state["sid"],
            st.session_state.get("active_variant") or variant_code,
            st.session_state.get("current_task") or variant_code,
        )

# -----------------------
# Sidebar: Session / hypothesis / advanced
# -----------------------
with st.sidebar:
    st.subheader("📊 Session Info")
    st.text(f"Session ID: {sid[:8]}...")
    st.text(f"Variant: {variant_code}")
    st.text(f"Start Time: {session_start_time[:19]}")

    try:
        from telemetry import get_real_time_hypothesis_status
        status = get_real_time_hypothesis_status()
        st.markdown("---")
        st.subheader("🧪 Hypothesis Testing")
        if status['status'] == 'no_data':
            st.info("📊 No data yet")
        elif status['status'] == 'collecting':
            st.info(f"📈 Collecting: {status['progress']:.1%}")
            if 'sample_sizes' in status:
                for v, size in status['sample_sizes'].items():
                    st.text(f"Variant {v}: {size} samples")
        elif status['status'] == 'preliminary':
            st.warning(f"⚡ Preliminary: {status['progress']:.1%}")
            st.text("Ready for initial analysis")
        elif status['status'] == 'ready':
            st.success("✅ Ready for full analysis")
        st.text(status.get('message', ''))
    except ImportError:
        st.text("Hypothesis tracking: Not available")

    if st.button("🔄 Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Advanced settings (auto-eval threshold)
if st.sidebar.checkbox("⚙️ Advanced Settings"):
    st.sidebar.subheader("Auto-Eval Configuration")
    st.session_state.setdefault("auto_eval_threshold", 0.7)
    new_thr = st.sidebar.slider(
        "Success Threshold",
        min_value=0.5, max_value=0.9,
        value=st.session_state["auto_eval_threshold"],
        step=0.05,
        help="Responses above this auto-eval score are considered successful"
    )
    if new_thr != st.session_state["auto_eval_threshold"]:
        st.session_state["auto_eval_threshold"] = new_thr
        st.sidebar.success(f"Threshold updated to {new_thr:.0%}")

    st.sidebar.markdown("**Auto-Eval Components:**")
    st.sidebar.text("• Keyphrase coverage (80%)")
    st.sidebar.text("• Has structure (10%)")
    st.sidebar.text("• Has specifics (10%)")

    if st.sidebar.button("📋 Export Session Data"):
        try:
            from telemetry import get_statistical_test_data
            data = get_statistical_test_data(st.session_state["auto_eval_threshold"])
            if data:
                st.sidebar.success("✅ Data exported to logs")
            else:
                st.sidebar.info("📊 No data to export yet")
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")
