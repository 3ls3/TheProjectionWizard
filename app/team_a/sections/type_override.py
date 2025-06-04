"""
Type override section for the main pipeline flow.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

def type_override_pipeline_section(is_current: bool = True):
    """Type override section for the main pipeline flow with continuous validation and cleaning."""
    st.header("🎯 Type Override & Target Selection")

    # If not current stage, show summary
    if not is_current and st.session_state.get('types_confirmed', False):
        target_col = st.session_state.get('target_column', 'None')
        type_changes = 0
        if 'uploaded_df' in st.session_state and 'type_overrides' in st.session_state:
            df = st.session_state['uploaded_df']
            def pandas_to_simple_type(dtype_str):
                dtype_str = str(dtype_str).lower()
                if 'int' in dtype_str:
                    return 'integer'
                elif 'float' in dtype_str:
                    return 'float'
                elif 'bool' in dtype_str:
                    return 'boolean'
                elif 'datetime' in dtype_str:
                    return 'datetime'
                elif 'category' in dtype_str:
                    return 'category'
                else:
                    return 'string/object'

            type_changes = sum(1 for col in df.columns
                              if st.session_state['type_overrides'].get(col) != pandas_to_simple_type(df[col].dtype))

        with st.expander(f"✅ **Types Configured** | Target: `{target_col}` | Changes: {type_changes}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🎯 **Target Variable:** `{target_col}`")
            with col2:
                if type_changes > 0:
                    st.info(f"📝 **Type changes applied:** {type_changes} columns")
                else:
                    st.info("📝 **No type changes** were needed")
        return

    # If current stage or not confirmed, show full UI
    if 'uploaded_df' not in st.session_state:
        st.error("❌ No data found. Please start from the beginning.")
        if is_current and st.button("🔙 Back to Upload"):
            st.session_state.stage = "upload"
            st.rerun()
        return

    df = st.session_state['uploaded_df']

    # Show type override section
    continuous_type_override_section(df, is_current)

def continuous_type_override_section(df, is_current: bool = True):
    """Continuous type override section with validation, cleaning, and final results on same page."""
    # Initialize session state for type overrides if not exists
    if 'type_overrides' not in st.session_state:
        st.session_state['type_overrides'] = {}
    if 'target_column' not in st.session_state:
        st.session_state['target_column'] = None
    if 'types_confirmed' not in st.session_state:
        st.session_state['types_confirmed'] = False

    # Define available data types for dropdown
    available_types = [
        'string/object',
        'integer',
        'float',
        'boolean',
        'category',
        'datetime'
    ]

    # Map pandas dtypes to our simplified types
    def pandas_to_simple_type(dtype_str):
        dtype_str = str(dtype_str).lower()
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'category' in dtype_str:
            return 'category'
        else:
            return 'string/object'

    # Always show the type configuration section (even if completed)
    st.markdown("---")
    st.subheader("🎯 Configure Data Types & Target")

    # Show completion status if types are confirmed
    if st.session_state.get('types_confirmed', False):
        st.success("✅ **Types and target already confirmed!**")

        # Show summary of confirmed settings
        target_col = st.session_state.get('target_column')
        st.info(f"🎯 **Target Variable:** `{target_col}`")

        # Show any type changes made
        type_changes = sum(1 for col in df.columns
                          if st.session_state['type_overrides'].get(col) != pandas_to_simple_type(df[col].dtype))
        if type_changes > 0:
            st.info(f"📝 **Type changes applied:** {type_changes} columns")

            # Show the changes in an expander
            with st.expander("📋 View Applied Type Changes", expanded=False):
                changes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Original Type': [pandas_to_simple_type(df[col].dtype) for col in df.columns],
                    'Applied Type': [st.session_state['type_overrides'].get(col, pandas_to_simple_type(df[col].dtype)) for col in df.columns],
                    'Is Target': ['✅' if col == target_col else '' for col in df.columns]
                })
                st.dataframe(changes_df, use_container_width=True)
        else:
            st.info("📝 **No type changes** were needed")

    else:
        # Show type override UI if not confirmed yet
        st.write("Select your target variable and review data types for optimal ML performance.")

        # Suggest target variable if not already set
        if st.session_state['target_column'] is None:
            # Smart target suggestion: look for likely target columns
            suggested_target = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'outcome', 'result', 'prediction', 'survived', 'price', 'salary', 'score']):
                    suggested_target = col
                    break

            # If no keyword match, suggest the last column
            if suggested_target is None:
                suggested_target = df.columns[-1]

            st.session_state['target_column'] = suggested_target

        # Target variable selection at the top
        st.write("**🎯 Target Variable Selection:**")
        target_options = list(df.columns)
        current_target_idx = target_options.index(st.session_state['target_column']) if st.session_state['target_column'] in target_options else 0

        selected_target = st.selectbox(
            "Choose the column you want to predict (target variable):",
            target_options,
            index=current_target_idx,
            help="This is the variable your ML model will learn to predict"
        )
        st.session_state['target_column'] = selected_target

        # Show target column info
        target_col = df[selected_target]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Type", pandas_to_simple_type(target_col.dtype))
        with col2:
            st.metric("Unique Values", target_col.nunique())
        with col3:
            st.metric("Missing Values", target_col.isnull().sum())

        st.markdown("---")

        # Data types configuration table (simplified)
        st.write("**📝 Data Types Configuration:**")
        st.write("Review and adjust data types if needed. Most columns should be correctly detected.")

        # Show only columns that might need type changes or are important
        columns_to_show = []
        for col in df.columns:
            if col == selected_target:
                continue  # Skip target column in type override

            inferred_type = pandas_to_simple_type(df[col].dtype)
            # Show columns that might need attention
            if (inferred_type == 'string/object' and df[col].nunique() < 20) or \
               (inferred_type == 'float' and df[col].nunique() < 10):
                columns_to_show.append(col)
            elif df[col].isnull().sum() > 0:  # Show columns with missing values
                columns_to_show.append(col)

        # If no special columns, show first few columns
        if not columns_to_show:
            columns_to_show = [col for col in df.columns[:5] if col != selected_target]

        # Display type override table
        if columns_to_show:
            # Create columns for the table header
            header_cols = st.columns([3, 2, 3])
            with header_cols[0]:
                st.write("**Column Name**")
            with header_cols[1]:
                st.write("**Detected Type**")
            with header_cols[2]:
                st.write("**Override Type**")

            st.markdown("---")

            # Create rows for important columns
            for idx, column in enumerate(columns_to_show):
                col_row = st.columns([3, 2, 3])

                with col_row[0]:
                    st.write(f"`{column}`")
                    # Show some sample values
                    sample_vals = df[column].dropna().iloc[:3].astype(str).tolist()
                    if sample_vals:
                        st.caption(f"Sample: {', '.join(sample_vals[:2])}...")

                with col_row[1]:
                    inferred_type = pandas_to_simple_type(df[column].dtype)
                    st.write(inferred_type)

                with col_row[2]:
                    # Get current override type or default to inferred type
                    current_type = st.session_state['type_overrides'].get(column, inferred_type)
                    new_type = st.selectbox(
                        "Select type",
                        available_types,
                        index=available_types.index(current_type) if current_type in available_types else 0,
                        key=f"type_{column}_{idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state['type_overrides'][column] = new_type

            # Show expandable section for all other columns
            with st.expander("⚙️ Advanced: Override types for other columns", expanded=False):
                other_columns = [col for col in df.columns if col not in columns_to_show and col != selected_target]
                if other_columns:
                    st.write("**All other columns (usually don't need changes):**")
                    for idx, column in enumerate(other_columns):
                        col_row = st.columns([3, 2, 3])
                        with col_row[0]:
                            st.write(f"`{column}`")
                        with col_row[1]:
                            inferred_type = pandas_to_simple_type(df[column].dtype)
                            st.write(inferred_type)
                        with col_row[2]:
                            current_type = st.session_state['type_overrides'].get(column, inferred_type)
                            new_type = st.selectbox(
                                "Select type",
                                available_types,
                                index=available_types.index(current_type) if current_type in available_types else 0,
                                key=f"type_other_{column}_{idx}",
                                label_visibility="collapsed"
                            )
                            st.session_state['type_overrides'][column] = new_type

        st.markdown("---")

        # Display current selections summary
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🎯 **Target Column:** `{st.session_state['target_column']}`")

        with col2:
            type_changes = sum(1 for col in df.columns
                              if st.session_state['type_overrides'].get(col) != pandas_to_simple_type(df[col].dtype))
            if type_changes > 0:
                st.info(f"📝 **Type changes:** {type_changes} columns")
            else:
                st.info("📝 **No type changes**")

        # Confirmation button (only show if current)
        if is_current:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                confirm_clicked = st.button("✅ Confirm Types & Target", type="primary")
        else:
            confirm_clicked = False

        # Handle confirmation outside of column layout to avoid width constraints
        if confirm_clicked and not st.session_state.get('types_confirmed', False):
            if st.session_state['target_column'] is None:
                st.error("❌ Please select a target column before confirming.")
            else:
                # Apply type conversions to the dataframe
                success, updated_df = apply_type_conversions(df, st.session_state['type_overrides'])

                if success:
                    st.session_state['types_confirmed'] = True
                    st.session_state['processed_df'] = updated_df

                    # 2️⃣ Transition to validation stage
                    st.session_state["stage"] = "validation"
                    st.rerun()
                else:
                    st.error("❌ Error applying type conversions. Please check your type selections.")

def apply_type_conversions(df, type_overrides):
    """Apply type conversions to the dataframe with error handling."""
    try:
        updated_df = df.copy()

        for column, new_type in type_overrides.items():
            if column not in df.columns:
                continue

            try:
                if new_type == 'integer':
                    # Handle potential NaN values for integer conversion
                    if updated_df[column].isnull().any():
                        # Convert to nullable integer type
                        updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce').astype('Int64')
                    else:
                        updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce').astype('int64')

                elif new_type == 'float':
                    updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce').astype('float64')

                elif new_type == 'boolean':
                    # Convert to boolean, handling common string representations
                    updated_df[column] = updated_df[column].astype(str).str.lower()
                    updated_df[column] = updated_df[column].replace({
                        'true': True, 'false': False, '1': True, '0': False,
                        'yes': True, 'no': False, 't': True, 'f': False
                    }).astype('boolean')

                elif new_type == 'category':
                    updated_df[column] = updated_df[column].astype('category')

                elif new_type == 'datetime':
                    updated_df[column] = pd.to_datetime(updated_df[column], errors='coerce')

                elif new_type == 'string/object':
                    updated_df[column] = updated_df[column].astype('object')

            except Exception as e:
                st.warning(f"⚠️ Could not convert column '{column}' to {new_type}: {str(e)}")
                # Keep original type if conversion fails
                continue

        return True, updated_df

    except Exception as e:
        st.error(f"❌ Error during type conversion: {str(e)}")
        return False, df
