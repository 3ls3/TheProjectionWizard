# app/wizardry/step7_inference.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

def run():
    st.header("📈 Step 7: Inference (Prediction)")

    # Check prerequisites
    best_model = st.session_state.get("best_model")
    clean_data = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")
    
    if any(x is None for x in [best_model, clean_data, target_col]):
        st.warning("Please complete previous steps (training and data preparation) before running inference.")
        return

    st.write(f"Using model: `{best_model}`")
    
    # Add option to use existing data or upload new data
    data_source = st.radio(
        "Choose data source for predictions:",
        ["Use existing dataset (test split)", "Upload new data"]
    )
    
    if data_source == "Use existing dataset (test split)":
        # If we haven't split the data yet, do it now
        if "test_data" not in st.session_state:
            test_size = st.slider("Test set size (%)", 10, 40, 20)
            test_size = test_size / 100
            
            # Split the data
            train_data, test_data = train_test_split(
                clean_data, 
                test_size=test_size,
                random_state=42
            )
            
            # Store both splits
            st.session_state["train_data"] = train_data
            st.session_state["test_data"] = test_data
            
            st.success(f"Data split into {len(train_data)} training and {len(test_data)} test samples")
        
        test_df = st.session_state["test_data"]
    else:
        # Handle new data upload
        uploaded_file = st.file_uploader("Upload new data for prediction", type=["csv"])
        if not uploaded_file:
            st.info("Please upload a CSV file to make predictions")
            return
        test_df = pd.read_csv(uploaded_file)
    
    # Display test data preview
    st.write("Data for predictions:")
    st.dataframe(test_df.head())
    
    # Make predictions (placeholder for now)
    # TODO: Replace with actual model predictions
    preds = [1 if i % 2 == 0 else 0 for i in range(len(test_df))]
    test_df["prediction"] = preds
    
    # Store predictions
    st.session_state["predictions"] = test_df
    
    # Display results
    st.success(f"Predictions completed using {best_model}")
    
    # Add download button
    st.download_button(
        "Download Predictions",
        test_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
    
    # Show prediction distribution
    st.write("Prediction Distribution:")
    st.bar_chart(test_df["prediction"].value_counts())

    # Add navigation buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Back: Training", use_container_width=True):
            st.session_state.current_step = "Step 6: Training"
            st.rerun()
    
    with col3:
        if st.button("Next: Explainability ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = "Step 8: Explainability"
            st.rerun()
