# app/wizardry/step6_training.py
import streamlit as st

def run():
    st.header("🏋️ Step 6: Model Training & Tuning")

    # Get required data from session state
    tool = st.session_state.get("automl_tool")
    models = st.session_state.get("candidate_models")
    task = st.session_state.get("task_type")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")

    # Check if any required variables are missing
    required_vars = {
        "AutoML Tool": tool,
        "Candidate Models": models,
        "Task Type": task,
        "Dataset": df,
        "Target Column": target_col
    }
    
    missing_vars = [name for name, value in required_vars.items() if value is None]
    if missing_vars:
        st.warning(f"Missing configuration: {', '.join(missing_vars)}. Complete previous steps first.")
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

    # Add navigation buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Back: Model Selection", use_container_width=True):
            st.session_state.current_step = "Step 5: Model Selection"
            st.rerun()
    
    with col3:
        if st.button("Next: Inference ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = "Step 7: Inference"
            st.rerun()
