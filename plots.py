from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator
from sklearn.tree import plot_tree
from wordcloud import WordCloud
from models import ClassificationMetrics, RegressionMetrics


# Common plotting configurations
DEFAULT_FIGSIZE = (8, 6)
DEFAULT_PALETTE = "viridis"
DEFAULT_GRID_ALPHA = 0.3


def _setup_plot_style(figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> None:
    """Setup common plot styling."""
    plt.figure(figsize=figsize)
    plt.grid(alpha=DEFAULT_GRID_ALPHA)


def _finalize_plot(title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply final styling and show plot."""
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_hist_numeric(
    data: pd.DataFrame, 
    feature: str, 
    figsize: Tuple[int, int] = (8, 4), 
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None
) -> None:
    """Plot histogram of numeric feature with optional filtering."""
    filtered_data = data.copy()
    if x_min is not None:
        filtered_data = filtered_data[filtered_data[feature] >= x_min]
    if x_max is not None:
        filtered_data = filtered_data[filtered_data[feature] <= x_max]

    _setup_plot_style(figsize)
    sns.histplot(filtered_data[feature], kde=True)
    _finalize_plot(f'Distribution of {feature}', feature, 'Frequency')

def barplot(
    category_counts: pd.Series, 
    title: str, 
    ylabel: str, 
    figsize: Tuple[int, int] = (4, 6), 
    top_n: Optional[int] = None, 
    color_palette: str = DEFAULT_PALETTE
) -> None:
    """Create a horizontal bar plot of category counts."""
    # Limit to top n values if specified
    if top_n is not None and len(category_counts) > top_n:
        plot_data = category_counts.nlargest(top_n)
    else:
        plot_data = category_counts
    
    plt.figure(figsize=figsize)
    plt.grid(axis='x', alpha=DEFAULT_GRID_ALPHA)
    sns.barplot(x=plot_data.values,
                y=plot_data.index,
                hue=plot_data.index,
                palette=color_palette,
                orient='h',
                legend=False,
                dodge=False)
    _finalize_plot(title, 'Frequency', ylabel)


def plot_hist_categorical(
    data: pd.DataFrame, 
    feature: str, 
    figsize: Tuple[int, int] = (4, 4)
) -> None:
    """Plot histogram of categorical feature."""
    category_counts = data[feature].value_counts()
    category_counts = category_counts.sort_values(ascending=False)
    barplot(category_counts, f'Distribution of {feature}', feature, figsize)


def plot_categorical_relationship(
    df: pd.DataFrame, 
    col1: str, 
    col2: str
) -> None:
    """Plot relationship between two categorical variables."""
    # Абсолютные значения
    count_crosstab = pd.crosstab(df[col1], df[col2])

    # Доли по строкам (внутри col1)
    row_prop = pd.crosstab(df[col1], df[col2], normalize='index')

    # Доли по столбцам (внутри col2)
    col_prop = pd.crosstab(df[col1], df[col2], normalize='columns')

    # Фигура с 3 подграфиками по горизонтали
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Абсолютные значения
    sns.heatmap(count_crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f'Абсолютные значения\n{col1} vs {col2}')
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    # 2. Доли внутри col1 (по строкам)
    sns.heatmap(row_prop, annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title(f'Доли внутри {col1} (по строкам)')
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel(col1)

    # 3. Доли внутри col2 (по столбцам)
    sns.heatmap(col_prop, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2])
    axes[2].set_title(f'Доли внутри {col2} (по столбцам)')
    axes[2].set_xlabel(col2)
    axes[2].set_ylabel(col1)

    plt.tight_layout()
    plt.show()


def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: Optional[str] = None,
    target_colors: Optional[Dict[Any, str]] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> None:
    """Plot scatter plot between two numeric variables with optional target coloring."""
    # Проверка колонок
    for col in [x_col, y_col, target_col] if target_col else [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в DataFrame.")

    # Проверка типов
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    # Проверка бинарного таргета
    if target_col is not None:
        unique_vals = sorted(df[target_col].dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Таргет '{target_col}' должен быть бинарным (2 уникальных значения).")

        # Палитра
        if target_colors is None:
            palette = {unique_vals[0]: 'blue', unique_vals[1]: 'red'}
        else:
            if not all(val in target_colors for val in unique_vals):
                raise ValueError(
                    f"target_colors должен содержать оба значения таргета: {unique_vals}")
            palette = target_colors

    # Построение графика
    plt.figure(figsize=(8, 6))
    if target_col:
        sns.scatterplot(data=df, x=x_col, y=y_col,
                        hue=target_col, palette=palette)
        plt.legend(title=target_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, color='blue')

    # Ограничения осей
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.title(f'Зависимость {y_col} от {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_classification_results(
    metrics: ClassificationMetrics, 
    model_name: str = "Model"
) -> None:
    """Plot classification evaluation results."""
    plt.figure(figsize=(15, 6))

    # Plot 1: Confusion Matrix
    if metrics.confusion_matrix is not None:
        plt.subplot(1, 2, 1)
        sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

    # Plot 2: ROC Curve (if available)
    if metrics.roc_curve is not None and metrics.roc_auc is not None:
        plt.subplot(1, 2, 2)
        plt.plot(metrics.roc_curve.fpr, metrics.roc_curve.tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {metrics.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def print_classification_report(
    metrics: Dict[str, Any], 
    model_name: str = "Model"
) -> None:
    """Print classification evaluation report."""
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
        'Value': [
            f'{metrics["ROC AUC"]:.4f}' if metrics["ROC AUC"] is not None else 'N/A',
            f'{metrics["F1 Score"]:.4f}',
            f'{metrics["Precision"]:.4f}',
            f'{metrics["Recall"]:.4f}',
            f'{metrics["Accuracy"]:.4f}'
        ]
    })

    # Classification report dataframe
    class_report_df = pd.DataFrame(metrics['Classification Report'])

    # Display results
    print("\n" + "="*60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("="*60)

    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))

    print("\n\nCLASSIFICATION REPORT:")
    print(class_report_df.to_string(index=False))

    print("\n" + "="*60)


def plot_feature_importance(
    model: BaseEstimator, 
    feature_names: List[str], 
    top_n: Optional[int] = None, 
    figsize: Tuple[int, int] = (10, 6),
    model_type: str = 'auto'
) -> pd.DataFrame:
    """Plot feature importance for various model types."""
    # Determine model type if auto
    if model_type == 'auto':
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            raise ValueError(
                "Could not determine model type automatically. Please specify 'tree' or 'linear'")

    # Get feature importances based on model type
    if model_type == 'tree':
        importances = model.feature_importances_
        importance_label = "Feature Importance"
    elif model_type == 'linear':
        # For linear models, use absolute coefficients as importance
        if len(model.coef_.shape) > 1:  # multi-class
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:  # binary classification
            importances = np.abs(model.coef_[0])
        importance_label = "Absolute Coefficient"
    else:
        raise ValueError("model_type must be either 'tree' or 'linear'")

    # Create DataFrame
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Select top_n features if specified
    if top_n is not None:
        feature_imp = feature_imp.head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature',
                data=feature_imp, hue='Feature', palette='viridis', legend=False)
    plt.title(f'Feature Importances ({model_type} model)')
    plt.xlabel(importance_label)
    plt.tight_layout()
    plt.show()

    return feature_imp


def visualize_decision_tree(
    model: BaseEstimator, 
    feature_names: List[str], 
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 10), 
    max_depth: Optional[int] = None
) -> None:
    """Visualize the decision tree structure."""
    plt.figure(figsize=figsize)
    plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              proportion=True,
              max_depth=max_depth)
    plt.title('Decision Tree Visualization')
    plt.show()


def plot_hyperparam_search_results(
    results: Union[Dict[str, Any], pd.DataFrame],
    score_key: str = 'mean_test_score',
    title: str = 'Hyperparameter Tuning Results',
    xtick_step: int = 5
) -> pd.DataFrame:
    """Plot hyperparameter search results."""
    # Normalize input
    if isinstance(results, dict):
        params = results.get('params')
        scores = results.get(score_key)
        if params is None or scores is None:
            raise ValueError(
                f"'params' and '{score_key}' must exist in results dict.")
        df = pd.DataFrame(params)
        df[score_key] = scores
    elif isinstance(results, pd.DataFrame):
        if 'params' in results.columns:
            df = pd.DataFrame(results['params'].tolist())
            df[score_key] = results[score_key].values
        else:
            raise ValueError("DataFrame input must have a 'params' column.")
    else:
        raise TypeError(
            "results must be a dict (like cv_results_) or a DataFrame.")

    df = df.reset_index().rename(columns={'index': 'Set #'})

    # Best score
    best_idx = df[score_key].idxmax()
    best_score = df.loc[best_idx, score_key]

    # Plot
    plt.figure(figsize=(12, 6))
    x = df['Set #']
    y = df[score_key]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Hyperparameter Set #")
    plt.ylabel(score_key)
    plt.grid(True)

    # Clean x-ticks
    plt.xticks(ticks=x[::xtick_step])

    # Highlight best
    plt.plot(df.loc[best_idx, 'Set #'], best_score,
             'ro', label=f'Best: {best_score:.4f}')
    plt.annotate(f'Best\n{best_score:.4f}',
                 xy=(df.loc[best_idx, 'Set #'], best_score),
                 xytext=(df.loc[best_idx, 'Set #'], best_score + 0.02),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='center')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


def compare_metrics_heatmap(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    df1_name: str = 'DF1', 
    df2_name: str = 'DF2',
    figsize: Tuple[int, int] = (8, 4), 
    annot_fontsize: int = 10,
    title: str = 'Comparison of ML Metrics',
    lower_is_better_metrics: Optional[List[str]] = None
) -> Tuple[Any, pd.DataFrame]:
    """Compare two DataFrames of ML metrics and plot a heatmap of their differences with per-column color scaling.
    
    Args:
        df1, df2: DataFrames to compare
        df1_name, df2_name: Names for the DataFrames
        figsize: Figure size
        annot_fontsize: Font size for annotations
        title: Plot title
        lower_is_better_metrics: List of metric names where lower values are better (e.g., ['Training Time', 'Loss'])
                                If None, will auto-detect based on common patterns
    """

    # Auto-detect lower-is-better metrics if not provided
    if lower_is_better_metrics is None:
        lower_is_better_patterns = [
            'time', 'loss', 'error', 'cost', 'latency', 'duration', 
            'mse', 'mae', 'rmse', 'runtime', 'seconds', 'minutes'
        ]
        lower_is_better_metrics = []
        for col in df1.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in lower_is_better_patterns):
                lower_is_better_metrics.append(col)

    # Calculate delta (difference) between DataFrames
    delta = df2 - df1
    
    # Add percentage change columns for training time metrics
    time_patterns = ['time', 'duration', 'runtime', 'seconds', 'minutes']
    time_columns = []
    for col in df1.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in time_patterns):
            time_columns.append(col)
            # Calculate percentage change: (new - old) / old * 100
            # Handle division by zero by setting to 0 where original value is 0
            pct_col_name = f"{col} Change (%)"
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_change = np.where(df1[col] != 0, (delta[col] / df1[col]) * 100, 0)
            delta[pct_col_name] = pct_change
            
            # Add percentage change column to lower_is_better_metrics if the original column is there
            if col in lower_is_better_metrics:
                lower_is_better_metrics.append(pct_col_name)
    
    # Create semantic delta where green = improvement, red = degradation
    semantic_delta = delta.copy()
    for col in lower_is_better_metrics:
        if col in semantic_delta.columns:
            # For lower-is-better metrics, flip the sign so negative changes (improvements) become positive
            semantic_delta[col] = -semantic_delta[col]

    # Normalize each column (metric) while preserving zero as center point
    # Use semantic_delta for normalization but keep original delta for annotations
    normalized_delta = semantic_delta.copy()
    for col in semantic_delta.columns:
        col_values = semantic_delta[col]
        col_min = col_values.min()
        col_max = col_values.max()
        
        if col_max != col_min:  # Avoid division by zero
            # Normalize positive and negative values separately to preserve zero as center
            normalized_col = col_values.copy()
            
            # Handle positive values (improvements - map to 0 to 1 for green)
            positive_mask = col_values > 0
            if col_max > 0 and positive_mask.any():
                normalized_col[positive_mask] = col_values[positive_mask] / col_max
            
            # Handle negative values (degradations - map to -1 to 0 for red)
            negative_mask = col_values < 0
            if col_min < 0 and negative_mask.any():
                normalized_col[negative_mask] = col_values[negative_mask] / abs(col_min)
            
            # Zero values remain exactly 0
            normalized_delta[col] = normalized_col
        else:
            normalized_delta[col] = 0  # Set to center value if all values are the same

    # Create a custom red-white-green colormap
    colors = ["#ff2700", "#ffffff", "#00b975"]  # Red -> White -> Green
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap using normalized data for colors but original data for annotations
    sns.heatmap(
        normalized_delta,
        annot=delta,
        fmt=".3f",
        cmap=cmap,
        center=0,
        linewidths=.5,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar_kws={'label': 'Improvement (Green) ← → Degradation (Red)'}
    )

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    return fig, delta


def plot_count_based_analysis(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = 'heatmap',
    sort_by_class: Optional[Union[str, int]] = None
) -> None:
    """Visualize count-based analysis results for multi-class text classification.
    
    Args:
        df: DataFrame from count_based_analysis function
        top_n: Number of top tokens to display
        figsize: Figure size tuple
        plot_type: Type of plot ('heatmap', 'bar', 'stacked_bar')
        sort_by_class: Class label to sort tokens by frequency (None for total_count sorting)
    """
    # Validate input
    if 'token' not in df.columns:
        raise ValueError("DataFrame must contain 'token' column")
    
    # Get class columns (count_X and freq_X)
    count_cols = [col for col in df.columns if col.startswith('count_') and col != 'total_count']
    freq_cols = [col for col in df.columns if col.startswith('freq_')]
    
    if not count_cols:
        raise ValueError("No count columns found in DataFrame")
    
    # Extract class names from column names
    classes = [col.replace('count_', '') for col in count_cols]
    
    # Sort DataFrame if sort_by_class is specified
    if sort_by_class is not None:
        sort_col = f'freq_{sort_by_class}'
        if sort_col not in df.columns:
            raise ValueError(f"Class '{sort_by_class}' not found. Available classes: {classes}")
        sorted_df = df.sort_values(sort_col, ascending=False).copy()
    else:
        sorted_df = df.copy()
    
    # Select top N tokens
    top_df = sorted_df.head(top_n).copy()
    
    if plot_type == 'heatmap':
        _plot_count_heatmap(top_df, freq_cols, classes, figsize, sort_by_class)
    elif plot_type == 'bar':
        _plot_count_bars(top_df, count_cols, classes, figsize, sort_by_class)
    elif plot_type == 'stacked_bar':
        _plot_stacked_bars(top_df, count_cols, classes, figsize, sort_by_class)
    else:
        raise ValueError("plot_type must be one of: 'heatmap', 'bar', 'stacked_bar'")


def _plot_count_heatmap(
    df: pd.DataFrame, 
    freq_cols: List[str], 
    classes: List[str], 
    figsize: Tuple[int, int],
    sort_by_class: Optional[Union[str, int]] = None
) -> None:
    """Plot heatmap of token frequencies across classes."""
    # Prepare data for heatmap (transpose to have tokens on y-axis)
    heatmap_data = df[freq_cols].copy()
    heatmap_data.index = df['token']
    heatmap_data.columns = [f'Class {cls}' for cls in classes]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Frequency'},
        xticklabels=True,
        yticklabels=True
    )
    
    # Update title based on sorting
    if sort_by_class is not None:
        title = f'Token Frequencies Across Classes (sorted by Class {sort_by_class})'
    else:
        title = 'Token Frequencies Across Classes'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Tokens', fontsize=12)
    plt.tight_layout()
    plt.show()


def _plot_count_bars(
    df: pd.DataFrame, 
    count_cols: List[str], 
    classes: List[str], 
    figsize: Tuple[int, int],
    sort_by_class: Optional[Union[str, int]] = None
) -> None:
    """Plot grouped horizontal bar chart of token counts across classes."""
    # Prepare data
    tokens = df['token'].head(15)  # Limit for readability
    data_subset = df[count_cols].head(15)
    
    # Set up the plot
    y = np.arange(len(tokens))
    height = 0.8 / len(classes)
    
    plt.figure(figsize=figsize)
    
    # Create horizontal bars for each class
    for i, (col, cls) in enumerate(zip(count_cols, classes)):
        plt.barh(y + i * height, data_subset[col], height, 
                 label=f'Class {cls}', alpha=0.8)
    
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Tokens', fontsize=12)
    
    # Update title based on sorting
    if sort_by_class is not None:
        title = f'Token Counts Across Classes (sorted by Class {sort_by_class})'
    else:
        title = 'Token Counts Across Classes'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yticks(y + height * (len(classes) - 1) / 2, tokens)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_stacked_bars(
    df: pd.DataFrame, 
    count_cols: List[str], 
    classes: List[str], 
    figsize: Tuple[int, int],
    sort_by_class: Optional[Union[str, int]] = None
) -> None:
    """Plot horizontal stacked bar chart of token counts across classes."""
    # Prepare data
    tokens = df['token'].head(20)
    data_subset = df[count_cols].head(20)
    
    plt.figure(figsize=figsize)
    
    # Create horizontal stacked bars
    left = np.zeros(len(tokens))
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    for col, cls, color in zip(count_cols, classes, colors):
        plt.barh(tokens, data_subset[col], left=left, 
                 label=f'Class {cls}', color=color, alpha=0.8)
        left += data_subset[col]
    
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Tokens', fontsize=12)
    
    # Update title based on sorting
    if sort_by_class is not None:
        title = f'Stacked Token Counts Across Classes (sorted by Class {sort_by_class})'
    else:
        title = 'Stacked Token Counts Across Classes'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_wordcloud(
    token_counts: pd.Series,
    title: str = "Word Cloud",
    figsize: Tuple[int, int] = (12, 8),
    max_words: int = 100,
    background_color: str = "white",
    colormap: str = "viridis",
    width: int = 800,
    height: int = 400
) -> None:
    """Create and display a word cloud from token counts.
    
    Args:
        token_counts: Pandas Series with tokens as index and counts as values
        title: Title for the plot
        figsize: Figure size tuple
        max_words: Maximum number of words to display
        background_color: Background color of the word cloud
        colormap: Matplotlib colormap name for word colors
        width: Width of the word cloud image
        height: Height of the word cloud image
    """
    # Convert pandas Series to dictionary for WordCloud
    word_freq = token_counts.to_dict()
    
    # Create WordCloud object
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(word_freq)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_regression_results(
    metrics: RegressionMetrics, 
    model_name: str = "Model"
) -> None:
    """Plot regression evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Regression Results', fontsize=16, fontweight='bold')
    
    # Get metrics data
    metrics_data = metrics.get_numeric_metrics()
    
    # Plot 1: Metrics bar chart
    metric_names = list(metrics_data.keys())
    metric_values = list(metrics_data.values())
    
    # Remove training time from main metrics for cleaner visualization
    main_metrics = {k: v for k, v in metrics_data.items() if k != 'Training Time (s)'}
    main_names = list(main_metrics.keys())
    main_values = list(main_metrics.values())
    
    axes[0, 0].bar(main_names, main_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Regression Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(main_values):
        axes[0, 0].text(i, v + max(main_values) * 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=10)
    
    # Plot 2: R² Score (most important metric)
    r2_score = metrics_data.get('R2', 0)
    axes[0, 1].bar(['R² Score'], [r2_score], color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('R² Score')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].text(0, r2_score + 0.02, f'{r2_score:.3f}', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 3: Error metrics comparison
    error_metrics = ['MAE', 'MSE', 'RMSE']
    error_values = [metrics_data.get(metric, 0) for metric in error_metrics]
    
    axes[1, 0].bar(error_metrics, error_values, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Error Metrics')
    axes[1, 0].set_ylabel('Error Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(error_values):
        axes[1, 0].text(i, v + max(error_values) * 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Training time (if available)
    if 'Training Time (s)' in metrics_data:
        training_time = metrics_data['Training Time (s)']
        axes[1, 1].bar(['Training Time'], [training_time], color='orange', alpha=0.7)
        axes[1, 1].set_title('Training Time')
        axes[1, 1].set_ylabel('Seconds')
        axes[1, 1].text(0, training_time + max(training_time, 1) * 0.01, f'{training_time:.2f}s', 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Training Time\nNot Available', 
                       ha='center', va='center', fontsize=12, 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Time')
    
    plt.tight_layout()
    plt.show()


def print_regression_report(
    metrics: Dict[str, Any], 
    model_name: str = "Model"
) -> None:
    """Print regression evaluation report in a formatted table."""
    print(f"\n{'='*60}")
    print(f"REGRESSION EVALUATION REPORT - {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create a formatted table
    print(f"{'Metric':<20} {'Value':<15} {'Interpretation':<25}")
    print(f"{'-'*60}")
    
    # R² Score
    r2 = metrics.get('R2', 0)
    r2_interpretation = "Excellent" if r2 >= 0.9 else "Good" if r2 >= 0.7 else "Fair" if r2 >= 0.5 else "Poor"
    print(f"{'R² Score':<20} {r2:<15.4f} {r2_interpretation:<25}")
    
    # MAE
    mae = metrics.get('MAE', 0)
    print(f"{'Mean Absolute Error':<20} {mae:<15.4f} {'Lower is better':<25}")
    
    # MSE
    mse = metrics.get('MSE', 0)
    print(f"{'Mean Squared Error':<20} {mse:<15.4f} {'Lower is better':<25}")
    
    # RMSE
    rmse = metrics.get('RMSE', 0)
    print(f"{'Root Mean Squared Error':<20} {rmse:<15.4f} {'Lower is better':<25}")
    
    # Explained Variance
    if 'Explained Variance' in metrics:
        explained_var = metrics['Explained Variance']
        ev_interpretation = "Excellent" if explained_var >= 0.9 else "Good" if explained_var >= 0.7 else "Fair" if explained_var >= 0.5 else "Poor"
        print(f"{'Explained Variance':<20} {explained_var:<15.4f} {ev_interpretation:<25}")
    
    # Training Time
    if 'Training Time (s)' in metrics:
        training_time = metrics['Training Time (s)']
        print(f"{'Training Time (s)':<20} {training_time:<15.2f} {'Seconds':<25}")
    
    print(f"{'-'*60}")
    print(f"Model Performance Summary:")
    print(f"  • R² Score: {r2:.4f} ({r2_interpretation})")
    print(f"  • Average Error (MAE): {mae:.4f}")
    print(f"  • Root Mean Squared Error: {rmse:.4f}")
    if 'Explained Variance' in metrics:
        print(f"  • Explained Variance: {explained_var:.4f} ({ev_interpretation})")
    print(f"{'='*60}")