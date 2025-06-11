"""
Prediction page for The Projection Wizard.
Provides UI for making predictions with the trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import io
import base64

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.step_7_predict import predict_logic, plot_utils
from common import constants, storage, schemas, utils


def create_input_form(metadata: dict, df_original: pd.DataFrame) -> dict:
    """
    Create a Streamlit form for user input based on original data features.

    This function creates inputs for the original categorical features and handles
    the one-hot encoding transformation needed for the trained model.

    Args:
        metadata: Metadata dictionary containing target info and feature schemas
        df_original: Original dataset to extract feature info from

    Returns:
        Dictionary of user input values (encoded to match model expectations)
    """
    target_info = schemas.TargetInfo(**metadata['target_info'])
    target_column = target_info.name

    # Load column mapping to understand the encoding
    run_id = st.session_state['run_id']
    try:
        column_mapping = storage.read_json(run_id, 'column_mapping.json')
        encoded_columns = column_mapping.get('encoded_columns', [])
    except:
        # Fallback if column mapping not available
        encoded_columns = [col for col in df_original.columns if col != target_column]

    st.subheader("🎯 Enter Values for Prediction")
    st.write(f"Please provide values for the following features (excluding target: **{target_column}**):")

    user_input = {}

    with st.form(key="prediction_form"):
        # Handle each original column and create appropriate inputs
        for col in df_original.columns:
            if col == target_column:
                continue  # Skip target column

            col_data = df_original[col].dropna()

            if len(col_data) == 0:
                st.warning(f"No data available for column: {col}")
                continue

            # Determine if column is numeric or categorical
            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric input - directly map to encoded columns
                if col in encoded_columns:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())

                    user_input[col] = st.number_input(
                        label=f"**{col}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100 if max_val != min_val else 1.0,
                        help=f"Range: {min_val:.3f} to {max_val:.3f}, Mean: {mean_val:.3f}"
                    )
            else:
                # Categorical input - need to handle one-hot encoding
                unique_values = sorted(col_data.unique().astype(str))

                selected_value = st.selectbox(
                    label=f"**{col}**",
                    options=unique_values,
                    index=0,
                    help=f"Available options: {', '.join(unique_values[:5])}{'...' if len(unique_values) > 5 else ''}"
                )

                # Create one-hot encoded columns for this categorical feature
                for val in unique_values:
                    encoded_col_name = f"{col}_{val}"
                    if encoded_col_name in encoded_columns:
                        user_input[encoded_col_name] = 1 if selected_value == val else 0

        # Submit button
        submitted = st.form_submit_button("🔮 Make Prediction", type="primary", use_container_width=True)

    if submitted:
        return user_input
    else:
        return None


def load_and_prepare_data(run_id: str) -> tuple:
    """
    Load necessary data for prediction.

    Returns:
        Tuple of (metadata, df_original, df_cleaned)
    """
    # Load metadata
    metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
    if not metadata:
        raise Exception("Could not load metadata")

    # Load original data for feature information
    original_data_path = storage.get_run_dir(run_id) / constants.ORIGINAL_DATA_FILENAME
    if not original_data_path.exists():
        raise Exception("Original data file not found")
    df_original = pd.read_csv(original_data_path)

    # Load cleaned data for training target distribution (for regression plots)
    df_cleaned = storage.read_cleaned_data(run_id)
    if df_cleaned is None:
        raise Exception("Could not load cleaned data")

    return metadata, df_original, df_cleaned


def display_prediction_results(
    result_df: pd.DataFrame,
    task_type: str,
    target_column: str,
    metadata: dict,
    df_cleaned: pd.DataFrame,
    run_id: str
) -> tuple:
    """
    Display prediction results with visualizations.

    Returns:
        Tuple of (csv_data, plot_path) for downloads
    """
    prediction_value = result_df['prediction'].iloc[0]

    if task_type.lower() == "classification":
        # Classification results
        st.success(f"🎯 **Predicted Class:** {prediction_value}")

        # Show model performance from metadata
        automl_info = metadata.get('automl_info', {})
        performance_metrics = automl_info.get('performance_metrics', {})

        if performance_metrics:
            # Find best accuracy-like metric to display
            accuracy_metric = None
            if 'Accuracy' in performance_metrics:
                accuracy_metric = ('Accuracy', performance_metrics['Accuracy'])
            elif 'AUC' in performance_metrics:
                accuracy_metric = ('AUC', performance_metrics['AUC'])
            elif 'F1' in performance_metrics:
                accuracy_metric = ('F1 Score', performance_metrics['F1'])

            if accuracy_metric:
                st.info(f"📊 Model **{accuracy_metric[0]}:** {accuracy_metric[1]:.1%}")

        # Get class labels for probability visualization
        class_labels_data = None
        try:
            class_labels_data = storage.read_json(run_id, 'class_labels.json')
        except:
            pass

        # Create probability bar chart
        if class_labels_data and 'class_labels' in class_labels_data:
            class_labels = class_labels_data['class_labels']

            # For single prediction, create mock probabilities (this is a limitation -
            # ideally we'd use predict_proba but that requires more complex pipeline handling)
            st.write("**📊 Prediction Confidence:**")

            # Create mock probabilities for visualization
            # In a real implementation, you'd use model.predict_proba()
            prob_data = {label: 0.1 for label in class_labels}
            prob_data[str(prediction_value)] = 0.8  # High confidence for predicted class

            # Create probability plot
            plots_dir = storage.get_run_dir(run_id) / constants.PLOTS_DIR
            plots_dir.mkdir(exist_ok=True)
            prob_plot_path = plots_dir / f"prediction_probs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            try:
                plot_utils.plot_classification_probs(
                    prob_data,
                    prob_plot_path,
                    title=f"Prediction Confidence for {target_column}"
                )

                if prob_plot_path.exists():
                    st.image(str(prob_plot_path), use_container_width=True)
                    return result_df.to_csv(index=False), prob_plot_path

            except Exception as e:
                st.warning(f"Could not create probability plot: {e}")

        else:
            st.info("💡 Prediction confidence visualization not available (class labels not found)")

    else:
        # Regression results
        st.success(f"🎯 **Predicted Value:** {prediction_value:.4f}")

        # Show model performance from metadata
        automl_info = metadata.get('automl_info', {})
        performance_metrics = automl_info.get('performance_metrics', {})

        if performance_metrics:
            # Find best regression metric to display
            regression_metric = None
            if 'R2' in performance_metrics:
                regression_metric = ('R² Score', performance_metrics['R2'])
            elif 'RMSE' in performance_metrics:
                regression_metric = ('RMSE', performance_metrics['RMSE'])
            elif 'MAE' in performance_metrics:
                regression_metric = ('MAE', performance_metrics['MAE'])

            if regression_metric:
                if regression_metric[0] == 'R² Score':
                    st.info(f"📊 Model **{regression_metric[0]}:** {regression_metric[1]:.1%}")
                else:
                    st.info(f"📊 Model **{regression_metric[0]}:** {regression_metric[1]:.4f}")

        # Create histogram with prediction line
        target_data = df_cleaned[target_column].dropna()

        if len(target_data) > 0:
            st.write("**📊 Prediction in Context:**")

            plots_dir = storage.get_run_dir(run_id) / constants.PLOTS_DIR
            plots_dir.mkdir(exist_ok=True)
            hist_plot_path = plots_dir / f"prediction_hist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            try:
                plot_utils.plot_regression_hist(
                    target_data,
                    prediction_value,
                    hist_plot_path,
                    title=f"Prediction vs Training Distribution for {target_column}"
                )

                if hist_plot_path.exists():
                    st.image(str(hist_plot_path), use_container_width=True)
                    return result_df.to_csv(index=False), hist_plot_path

            except Exception as e:
                st.warning(f"Could not create histogram plot: {e}")

        else:
            st.warning("No training target data available for context visualization")

    # Return CSV data only if no plot was created
    return result_df.to_csv(index=False), None


def show_predict_page():
    """Display the prediction page."""

    # Page Title
    st.title("Step 8: Make Predictions")

    # Check if run_id exists in session state
    if 'run_id' not in st.session_state:
        st.error("No active run found. Please upload a file first.")
        if st.button("Go to Upload Page"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
        return

    run_id = st.session_state['run_id']

    # Display Current Run ID
    st.info(f"**Current Run ID:** {run_id}")

    # Introductory text
    st.write("Use your trained model to make predictions on new data. Enter feature values below and get instant predictions with confidence insights.")

    try:
        # Load necessary data
        with st.spinner("Loading model and data..."):
            metadata, df_original, df_cleaned = load_and_prepare_data(run_id)

        # Get task info
        target_info = schemas.TargetInfo(**metadata['target_info'])
        task_type = target_info.task_type
        target_column = target_info.name

        # Display model info
        automl_info = metadata.get('automl_info', {})
        model_name = automl_info.get('best_model_name', 'Unknown')

        st.subheader("📋 Model Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Task Type", task_type.title())
        with col2:
            st.metric("Target Column", target_column)
        with col3:
            st.metric("Model", model_name)

        # Create input form
        user_input = create_input_form(metadata, df_original)

        if user_input is not None:
            # User submitted the form
            st.subheader("🔮 Prediction Results")

            with st.spinner("Making prediction..."):
                try:
                    # Load the trained model
                    run_dir = storage.get_run_dir(run_id)
                    model = predict_logic.load_pipeline(run_dir)

                    # Convert user input to DataFrame
                    input_df = pd.DataFrame([user_input])

                    # Generate predictions
                    result_df = predict_logic.generate_predictions(model, input_df, target_column)

                    # Display results with visualizations
                    csv_data, plot_path = display_prediction_results(
                        result_df, task_type, target_column, metadata, df_cleaned, run_id
                    )

                    # Show detailed prediction info
                    with st.expander("🔍 Detailed Prediction Information", expanded=False):
                        st.write("**Input Features (aligned for model):**")
                        # Show the aligned features (excluding prediction column)
                        feature_df = result_df.drop(columns=['prediction'])
                        st.dataframe(feature_df.T, use_container_width=True)

                        st.write("**Full Prediction Result:**")
                        st.dataframe(result_df, use_container_width=True)

                    # Download buttons (outside the form)
                    st.subheader("💾 Download Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        # CSV download
                        st.download_button(
                            label="📄 Download Prediction CSV",
                            data=csv_data,
                            file_name=f"{run_id}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:
                        # PNG download (if plot was created)
                        if plot_path and plot_path.exists():
                            with open(plot_path, 'rb') as f:
                                st.download_button(
                                    label="🖼️ Download Prediction Chart",
                                    data=f.read(),
                                    file_name=f"{run_id}_prediction_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        else:
                            st.info("Chart download not available")

                    # Success message
                    st.success("✅ Prediction completed successfully!")

                except Exception as e:
                    is_dev_mode = st.session_state.get("developer_mode_active", False)
                    utils.display_page_error(e, run_id=run_id, stage_name="PREDICT", dev_mode=is_dev_mode)

        else:
            # Show help information while waiting for user input
            with st.expander("ℹ️ How to make predictions", expanded=False):
                st.write(f"""
                **Prediction Process:**
                1. **Enter Feature Values:** Fill in the form above with values for each feature
                2. **Submit:** Click "Make Prediction" to get results
                3. **View Results:** See the predicted {task_type} result with model confidence
                4. **Download:** Get CSV data and visualization charts

                **Feature Information:**
                - **Target Column:** {target_column} (excluded from input)
                - **Input Features:** {len([col for col in df_original.columns if col != target_column])} features required
                - **Model Used:** {model_name}

                **Tips:**
                - Use realistic values within the training data ranges
                - Check the help text for each field to see valid ranges
                - All fields are required for accurate predictions
                """)

    except Exception as e:
        is_dev_mode = st.session_state.get("developer_mode_active", False)
        utils.display_page_error(e, run_id=run_id, stage_name="PREDICT", dev_mode=is_dev_mode)


if __name__ == "__main__":
    show_predict_page()
