"""
Plotting utilities for predictions in The Projection Wizard.
Handles visualization of classification probabilities and regression predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, List
import warnings

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

# Configure plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_classification_probs(
    prob_series_or_df: Union[pd.Series, pd.DataFrame, dict, list],
    out_path: Union[str, Path],
    title: Optional[str] = None,
    class_labels: Optional[List[str]] = None,
    figsize: tuple = (10, 6),
    color_palette: str = "viridis"
) -> bool:
    """
    Create and save a horizontal bar chart of classification probabilities.

    Args:
        prob_series_or_df: Probabilities data - can be:
            - pd.Series with class names as index and probabilities as values
            - pd.DataFrame with one row of probabilities
            - dict with class_name: probability pairs
            - list of probabilities (requires class_labels)
        out_path: Path where to save the plot
        title: Custom title for the plot
        class_labels: List of class labels (required if prob_series_or_df is a list)
        figsize: Figure size as (width, height)
        color_palette: Color palette name for the bars

    Returns:
        True if plot was created successfully, False otherwise
    """
    try:
        # Convert input to pandas Series for consistent handling
        if isinstance(prob_series_or_df, pd.Series):
            prob_series = prob_series_or_df.copy()
        elif isinstance(prob_series_or_df, pd.DataFrame):
            if len(prob_series_or_df) == 1:
                prob_series = prob_series_or_df.iloc[0]
            else:
                # Take the first row if multiple rows
                prob_series = prob_series_or_df.iloc[0]
                warnings.warn("DataFrame has multiple rows, using first row for visualization")
        elif isinstance(prob_series_or_df, dict):
            prob_series = pd.Series(prob_series_or_df)
        elif isinstance(prob_series_or_df, list):
            if class_labels is None:
                raise ValueError("class_labels must be provided when prob_series_or_df is a list")
            if len(prob_series_or_df) != len(class_labels):
                raise ValueError("Length of probabilities list must match length of class_labels")
            prob_series = pd.Series(prob_series_or_df, index=class_labels)
        else:
            raise ValueError(f"Unsupported type for prob_series_or_df: {type(prob_series_or_df)}")

        # Validate probabilities
        if prob_series.empty:
            raise ValueError("Probability data is empty")

        # Handle negative values
        if (prob_series < 0).any():
            warnings.warn("Found negative probability values, setting them to 0")
            prob_series = prob_series.clip(lower=0)

        # Sort by probability values for better visualization
        prob_series = prob_series.sort_values(ascending=True)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar chart
        colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(prob_series)))
        bars = ax.barh(range(len(prob_series)), prob_series.values, color=colors)

        # Customize the plot
        ax.set_yticks(range(len(prob_series)))
        ax.set_yticklabels(prob_series.index)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)

        # Set title
        if title is None:
            title = f'Classification Probabilities'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add probability values as text on bars
        for i, (bar, value) in enumerate(zip(bars, prob_series.values)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=10)

        # Set x-axis limits
        ax.set_xlim(0, max(1.1, prob_series.max() * 1.1))

        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3)

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        # Clean up any open figures
        plt.close('all')
        raise Exception(f"Failed to create classification probability plot: {str(e)}")


def plot_regression_hist(
    train_target_series: Union[pd.Series, np.ndarray, list],
    pred_value: float,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    bins: int = 30,
    figsize: tuple = (10, 6),
    alpha: float = 0.7,
    show_kde: bool = True
) -> bool:
    """
    Create and save a histogram/KDE of training target values with a vertical line for prediction.

    Args:
        train_target_series: Training target values for histogram
        pred_value: Predicted value to show as vertical line
        out_path: Path where to save the plot
        title: Custom title for the plot
        bins: Number of bins for histogram
        figsize: Figure size as (width, height)
        alpha: Transparency of histogram bars
        show_kde: Whether to overlay a KDE curve

    Returns:
        True if plot was created successfully, False otherwise
    """
    try:
        # Convert input to numpy array for consistent handling
        if isinstance(train_target_series, pd.Series):
            train_values = train_target_series.dropna().values
        elif isinstance(train_target_series, np.ndarray):
            train_values = train_target_series[~np.isnan(train_target_series)]
        elif isinstance(train_target_series, list):
            train_values = np.array([x for x in train_target_series if pd.notna(x)])
        else:
            raise ValueError(f"Unsupported type for train_target_series: {type(train_target_series)}")

        # Validate inputs
        if len(train_values) == 0:
            raise ValueError("Training target values are empty after removing NaN")

        if pd.isna(pred_value):
            raise ValueError("Prediction value cannot be NaN")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create histogram
        n, bins_edges, patches = ax.hist(
            train_values,
            bins=bins,
            alpha=alpha,
            color='skyblue',
            edgecolor='black',
            linewidth=0.5,
            density=True,
            label='Training Target Distribution'
        )

        # Add KDE if requested
        if show_kde and len(train_values) > 1:
            try:
                # Create KDE
                from scipy import stats
                kde = stats.gaussian_kde(train_values)
                x_range = np.linspace(train_values.min(), train_values.max(), 200)
                kde_values = kde(x_range)
                ax.plot(x_range, kde_values, 'r-', linewidth=2, alpha=0.8, label='KDE')
            except ImportError:
                # Fallback if scipy is not available
                warnings.warn("scipy not available, skipping KDE overlay")
            except Exception as kde_error:
                warnings.warn(f"Could not create KDE: {str(kde_error)}")

        # Add vertical line for prediction
        ax.axvline(
            pred_value,
            color='red',
            linestyle='--',
            linewidth=3,
            alpha=0.8,
            label=f'Prediction: {pred_value:.3f}'
        )

        # Customize the plot
        ax.set_xlabel('Target Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

        # Set title
        if title is None:
            title = 'Training Target Distribution vs Prediction'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add legend
        ax.legend(fontsize=10)

        # Add grid for better readability
        ax.grid(alpha=0.3)

        # Add statistics text box
        stats_text = f'Training Stats:\n'
        stats_text += f'Mean: {np.mean(train_values):.3f}\n'
        stats_text += f'Std: {np.std(train_values):.3f}\n'
        stats_text += f'Min: {np.min(train_values):.3f}\n'
        stats_text += f'Max: {np.max(train_values):.3f}'

        # Position the text box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        # Clean up any open figures
        plt.close('all')
        raise Exception(f"Failed to create regression histogram plot: {str(e)}")


def plot_prediction_confidence(
    predictions: Union[pd.Series, np.ndarray, list],
    confidences: Union[pd.Series, np.ndarray, list],
    out_path: Union[str, Path],
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> bool:
    """
    Create a scatter plot of predictions vs confidence scores.

    Args:
        predictions: Prediction values
        confidences: Confidence scores for each prediction
        out_path: Path where to save the plot
        title: Custom title for the plot
        figsize: Figure size as (width, height)

    Returns:
        True if plot was created successfully, False otherwise
    """
    try:
        # Convert inputs to arrays
        pred_array = np.array(predictions)
        conf_array = np.array(confidences)

        if len(pred_array) != len(conf_array):
            raise ValueError("Predictions and confidences must have the same length")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create scatter plot
        scatter = ax.scatter(pred_array, conf_array, alpha=0.6, s=50)

        # Customize the plot
        ax.set_xlabel('Prediction Value', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)

        if title is None:
            title = 'Prediction vs Confidence'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save the plot
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        plt.close('all')
        raise Exception(f"Failed to create prediction confidence plot: {str(e)}")


def create_prediction_summary_plot(
    result_df: pd.DataFrame,
    task_type: str,
    out_path: Union[str, Path],
    title: Optional[str] = None
) -> bool:
    """
    Create a summary visualization based on task type.

    Args:
        result_df: DataFrame with prediction results
        task_type: "classification" or "regression"
        out_path: Path where to save the plot
        title: Custom title for the plot

    Returns:
        True if plot was created successfully, False otherwise
    """
    try:
        if 'prediction' not in result_df.columns:
            raise ValueError("result_df must contain a 'prediction' column")

        if task_type.lower() == "classification":
            # For classification, show prediction distribution
            pred_counts = result_df['prediction'].value_counts()

            fig, ax = plt.subplots(figsize=(10, 6))
            pred_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_xlabel('Predicted Class', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(title or 'Prediction Distribution', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)

        elif task_type.lower() == "regression":
            # For regression, show prediction histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(result_df['prediction'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Predicted Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(title or 'Prediction Distribution', fontsize=14, fontweight='bold')

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        # Add grid
        ax.grid(alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save the plot
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        plt.close('all')
        raise Exception(f"Failed to create prediction summary plot: {str(e)}")
