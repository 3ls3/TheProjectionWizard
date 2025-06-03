# app/wizardry/step8_explainability.py
import streamlit as st

def run():
    st.header("🧠 Step 8: Model Explainability")

    model = st.session_state.get("best_model")
    if model is None:
        st.warning("No model available for explanation.")
        return

    st.write(f"Explainability for model: `{model}`")
    st.write("Using SHAP and LIME (placeholders for now)...")

    # Placeholder explanation visuals
    st.bar_chart([0.2, 0.15, 0.1, 0.05, 0.03], use_container_width=True)
    st.info("SHAP force plots and LIME graphs will go here.")

    st.session_state["explanations"] = "Explanation artifacts (placeholder)"

    # Add navigation buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Back: Inference", use_container_width=True):
            st.session_state.current_step = "Step 7: Inference"
            st.rerun()
    
    with col3:
        if st.button("Next: Final Outputs ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = "Step 9: Outputs"
            st.rerun()
