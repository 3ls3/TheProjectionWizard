# app/wizardry/step6_training.py
import streamlit as st

def run():
    st.header("🏋️ Step 6: Model Training & Tuning")

    tool = st.session_state.get("automl_tool")
    models = st.session_state.get("candidate_models")
    task = st.session_state.get("task_type")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")

    if None in (tool, models, task, df, target_col):
        st.warning("Missing configuration. Complete previous steps first.")
        return

    st.write(f"Training models for task: `{task}` using `{tool}`...")
    st.write("Selected candidate models:", models)

    # Placeholder for actual training logic
    st.info("Training logic not implemented yet — here you will train models with CV, tune, and rank them.")

    # Simulate leaderboard
    leaderboard = [
        {"model": model, "score": round(0.8 - i*0.02, 4)} for i, model in enumerate(models)
    ]
    st.session_state["leaderboard"] = leaderboard
    st.session_state["best_model"] = leaderboard[0]["model"]

    st.success(f"Best model: {leaderboard[0]['model']} (score: {leaderboard[0]['score']})")
