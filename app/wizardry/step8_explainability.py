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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, target_col):
    """Preprocess data for model explanation"""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Separate features and target
    y = processed_df[target_col] if target_col in processed_df.columns else None
    X = processed_df.drop(columns=[target_col]) if target_col in processed_df.columns else processed_df
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

def get_explainer(model, X, task_type):
    """Get the appropriate explainer based on the model type"""
    try:
        if isinstance(model, (KMeans, DBSCAN, AgglomerativeClustering)):
            return None
        elif task_type in ['regression', 'time_series_forecasting']:
            return shap.Explainer(model, X)
        else:
            return shap.Explainer(model.predict_proba, X)
    except Exception as e:
        st.warning(f"Could not create SHAP explainer: {str(e)}")
        return None

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

def get_feature_importance(model, X, y, task_type):
    """Get feature importance using permutation importance"""
    try:
        if task_type in ['regression', 'time_series_forecasting']:
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'accuracy'
            
        result = permutation_importance(
            model, X, y if y is not None else np.zeros(len(X)),
            n_repeats=10,
            random_state=42,
            scoring=scoring
        )
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        st.warning(f"Could not calculate feature importance: {str(e)}")
        return None

def run():
    st.header("🧠 Step 8: Model Explainability")

    # Get required data from session state
    model = st.session_state.get("best_model")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")
    task = st.session_state.get("task_type")

    if model is None:
        st.warning("No model available for explanation.")
        return

    st.write(f"Explainability for model: `{model}`")

    # Preprocess the data for explanation
    processed_df, y = preprocess_data(df, target_col)
    X = processed_df

    # Create tabs for different types of explanations
    tab1, tab2, tab3 = st.tabs(["Global Analysis", "Local Analysis", "Model-Specific Analysis"])

    with tab1:
        st.subheader("Global Model Analysis")
        
        # Feature Importance
        st.write("Feature Importance Analysis")
        importance_df = get_feature_importance(model, X, y, task)
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance (Permutation Importance)')
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("Local Model Analysis")
        
        # SHAP Analysis
        st.write("SHAP Analysis")
        try:
            with st.spinner("Calculating SHAP values..."):
                explainer = get_explainer(model, X, task)
                if explainer is not None:
                    # Use a smaller sample for SHAP calculations
                    sample_size = min(100, len(X))
                    X_sample = X.iloc[:sample_size]
                    
                    shap_values = explainer(X_sample)

                    # Summary Plot
                    st.write("Feature Importance Summary")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(fig)
                    plt.close()

                    # Dependence Plot for top feature
                    if task in ['regression', 'time_series_forecasting']:
                        st.write("Feature Dependence Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.dependence_plot(0, shap_values.values, X_sample, show=False)
                        st.pyplot(fig)
                        plt.close()

        except Exception as e:
            st.error(f"Error in SHAP analysis: {str(e)}")

        # LIME Analysis
        st.write("LIME Analysis")
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

                # Let user select example to explain
                example_idx = st.selectbox(
                    "Select example to explain",
                    range(min(5, len(X))),
                    format_func=lambda x: f"Example {x+1}"
                )

                exp = explainer.explain_instance(
                    X.iloc[example_idx].values,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=10
                )
                
                # Display explanation
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(f"Error in LIME analysis: {str(e)}")

    with tab3:
        st.subheader("Model-Specific Analysis")
        
        if isinstance(model, (KMeans, DBSCAN, AgglomerativeClustering)):
            try:
                analyze_cluster_centers(model, X)
            except Exception as e:
                st.error(f"Error in clustering analysis: {str(e)}")
        else:
            # Add model-specific visualizations here
            st.write("Model-specific analysis not available for this model type.")

    # Store explanations in session state
    st.session_state["explanations"] = {
        "shap_values": shap_values if 'shap_values' in locals() else None,
        "lime_explainer": explainer if 'explainer' in locals() else None,
        "feature_importance": importance_df if 'importance_df' in locals() else None
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
