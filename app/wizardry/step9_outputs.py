# app/wizardry/step9_outputs.py
import streamlit as st

def run():
    st.header("📦 Step 9: Final Outputs to User")

    st.write("🎁 Final Deliverables:")
    
    if "predictions" in st.session_state:
        st.download_button("📄 Download Predictions CSV",
                           data=st.session_state["predictions"].to_csv(index=False),
                           file_name="final_predictions.csv")

    if "clean_data" in st.session_state:
        st.download_button("🧼 Download Cleaned Data CSV",
                           data=st.session_state["clean_data"].to_csv(index=False),
                           file_name="cleaned_data.csv")

    st.write("📁 Model File: (simulated)")
    st.download_button("💾 Download Model File", data="BinaryDataPlaceholder", file_name="model.pkl")

    st.write("📊 Explainability Artifacts:")
    st.text(st.session_state.get("explanations", "None available"))

    # Add back button at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("⬅️ Back: Explainability", use_container_width=True):
            st.session_state.current_step = "Step 8: Explainability"
            st.rerun()
