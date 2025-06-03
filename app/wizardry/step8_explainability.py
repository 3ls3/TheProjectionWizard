# app/wizardry/step8_explainability.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from .step6_training import preprocess_data
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def get_explainer(model, X, task_type):
    """Get the appropriate explainer based on the model type"""
    if isinstance(model, (KMeans, DBSCAN, AgglomerativeClustering)):
        # For clustering models, use a custom explainer
        return None
    elif task_type in ['regression', 'time_series_forecasting']:
        return shap.Explainer(model, X)
    else:
        return shap.Explainer(model.predict_proba, X)

def analyze_cluster_centers(model, X):
    """Analyze and visualize cluster centers for clustering models"""
    if isinstance(model, KMeans):
        # Get cluster centers
        centers = model.cluster_centers_
        
        # Create a DataFrame for better visualization
        centers_df = pd.DataFrame(centers, columns=X.columns)
        
        # Plot cluster centers
        st.write("Cluster Centers:")
        st.dataframe(centers_df)
        
        # Plot feature importance based on variance of cluster centers
        feature_importance = np.std(centers, axis=0)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        st.write("Feature Importance (based on cluster center variance):")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance for Clustering')
        st.pyplot(fig)
        plt.close()
        
        # Plot cluster sizes
        cluster_sizes = pd.Series(model.labels_).value_counts()
        st.write("Cluster Sizes:")
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_sizes.plot(kind='bar')
        plt.title('Number of Samples per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        st.pyplot(fig)
        plt.close()

def run():
    st.header("🧠 Step 8: Model Explainability")

    # Get required data from session state
    task = st.session_state.get("task_type")
    best_model_name = st.session_state.get("best_model")
    trained_models = st.session_state.get("trained_models")
    scaler = st.session_state.get("scaler")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")

    # Check if any required variables are missing
    required_vars = {
        "Task Type": task,
        "Best Model": best_model_name,
        "Trained Models": trained_models,
        "Scaler": scaler,
        "Dataset": df
    }
    
    missing_vars = [name for name, value in required_vars.items() if value is None]
    if missing_vars:
        st.warning(f"Missing configuration: {', '.join(missing_vars)}. Complete previous steps first.")
        return

    st.write(f"Explainability for model: `{best_model_name}`")
    model = trained_models[best_model_name]

    # Preprocess the data for explanation
    processed_df, _ = preprocess_data(df, target_col)
    X = processed_df

<<<<<<< Updated upstream
    st.session_state["explanations"] = "Explanation artifacts (placeholder)"
=======
    # Handle different model types
    if isinstance(model, (KMeans, DBSCAN, AgglomerativeClustering)):
        st.subheader("Clustering Analysis")
        try:
            analyze_cluster_centers(model, X)
        except Exception as e:
            st.error(f"Error in clustering analysis: {str(e)}")
    else:
        # SHAP Analysis for non-clustering models
        st.subheader("SHAP Analysis")
        try:
            with st.spinner("Calculating SHAP values..."):
                explainer = get_explainer(model, X, task)
                if explainer is not None:
                    shap_values = explainer(X.iloc[:100])  # Use first 100 samples for performance

                    # Summary Plot
                    st.write("Feature Importance Summary")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X.iloc[:100], show=False)
                    st.pyplot(fig)
                    plt.close()

                    # Dependence Plot for top feature
                    if task in ['regression', 'time_series_forecasting']:
                        st.write("Feature Dependence Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.dependence_plot(0, shap_values.values, X.iloc[:100], show=False)
                        st.pyplot(fig)
                        plt.close()

        except Exception as e:
            st.error(f"Error in SHAP analysis: {str(e)}")

        # LIME Analysis for non-clustering models
        st.subheader("LIME Analysis")
        try:
            with st.spinner("Calculating LIME explanations..."):
                # Create LIME explainer
                categorical_features = np.where(X.dtypes == 'object')[0]
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=X.columns,
                    class_names=['Class 0', 'Class 1'] if task == 'binary_classification' else None,
                    categorical_features=categorical_features,
                    mode='regression' if task in ['regression', 'time_series_forecasting'] else 'classification'
                )

                # Explain a few examples
                for i in range(min(3, len(X))):
                    st.write(f"Example {i+1} Explanation:")
                    exp = explainer.explain_instance(
                        X.iloc[i].values,
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        num_features=5
                    )
                    
                    # Display explanation
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)
                    plt.close()

        except Exception as e:
            st.error(f"Error in LIME analysis: {str(e)}")

    # Store explanations in session state
    st.session_state["explanations"] = {
        "shap_values": shap_values if 'shap_values' in locals() else None,
        "lime_explainer": explainer if 'explainer' in locals() else None
    }

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
>>>>>>> Stashed changes
