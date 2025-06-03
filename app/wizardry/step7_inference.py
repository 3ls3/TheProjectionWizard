# app/wizardry/step7_inference.py
import streamlit as st
import pandas as pd

def run():
    st.header("📈 Step 7: Inference (Prediction)")

    best_model = st.session_state.get("best_model")
    if best_model is None:
        st.warning("Train and select a model before running inference.")
        return

    uploaded_file = st.file_uploader("Upload new data for prediction", type=["csv"])
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
        st.write("Uploaded test data:")
        st.dataframe(test_df.head())

        # Placeholder prediction
        preds = [1 if i % 2 == 0 else 0 for i in range(len(test_df))]  # Dummy output
        test_df["prediction"] = preds
        st.session_state["predictions"] = test_df

        st.success(f"Predictions made using {best_model}.")
        st.download_button("Download Predictions", test_df.to_csv(index=False), file_name="predictions.csv")
