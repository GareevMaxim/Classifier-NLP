from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)
from sklearn.model_selection import cross_validate
from typing import Tuple

import plots as p
from models import ClassificationMetrics, RocCurveData, ClassificationReportRow, BaseMetrics, RegressionMetrics, MultipleModelResults

def divide_data(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split data into features and target."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def _calculate_classification_metrics(
    y_test: Any, y_pred: Any, y_probs: Optional[Any] = None
) -> ClassificationMetrics:
    """Calculate classification performance metrics into a Pydantic model."""
    roc_auc_value: Optional[float] = (
        float(roc_auc_score(y_test, y_probs)) if y_probs is not None else None
    )
    metrics_model = ClassificationMetrics(
        roc_auc=roc_auc_value,
        f1_score=float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        precision=float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        recall=float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        accuracy=float((y_pred == y_test).mean()),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )

    # Classification report rows
    class_rows: List[ClassificationReportRow] = [
        ClassificationReportRow(
            class_label='Positive',
            precision=float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        ),
        ClassificationReportRow(
            class_label='Negative',
            precision=float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        ),
    ]
    metrics_model.classification_report = class_rows

    if y_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        metrics_model.roc_curve = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)

    return metrics_model


def evaluate_classification(
    y_test: Any,
    y_pred: Any,
    y_probs: Optional[Any] = None,
    model_name: str = "Model",
    enable_plot: bool = True
) -> ClassificationMetrics:
    """Evaluate performance and optionally plot/print using the Pydantic model."""
    metrics_model = _calculate_classification_metrics(y_test, y_pred, y_probs)

    if enable_plot:
        p.plot_classification_results(metrics_model, model_name)
        p.print_classification_report(metrics_model.to_report_dict(), model_name)

    return metrics_model


def _calculate_regression_metrics(y_test: Any, y_pred: Any) -> RegressionMetrics:
    """Calculate regression performance metrics into a Pydantic model."""
    mae_value = float(mean_absolute_error(y_test, y_pred))
    mse_value = float(mean_squared_error(y_test, y_pred))
    rmse_value = float(mse_value ** 0.5)
    r2_value = float(r2_score(y_test, y_pred))
    explained_variance_value = float(explained_variance_score(y_test, y_pred))
    
    metrics_model = RegressionMetrics(
        mae=mae_value,
        mse=mse_value,
        rmse=rmse_value,
        r2=r2_value,
        explained_variance=explained_variance_value,
    )
    
    return metrics_model


def evaluate_regression(
    y_test: Any,
    y_pred: Any,
    model_name: str = "Model",
    enable_plot: bool = True
) -> RegressionMetrics:
    """Evaluate regression performance and optionally plot/print using the Pydantic model."""
    metrics_model = _calculate_regression_metrics(y_test, y_pred)

    if enable_plot:
        p.plot_regression_results(metrics_model, model_name)
        p.print_regression_report(metrics_model.get_numeric_metrics(), model_name)

    return metrics_model


def aggregate_regression_cv_metrics(
    *,
    mae: Optional[float] = None,
    mse: Optional[float] = None,
    rmse: Optional[float] = None,
    r2: Optional[float] = None,
    explained_variance: Optional[float] = None,
    training_time: Optional[float] = None,
    name: Optional[str] = None,
) -> RegressionMetrics:
    """Build a RegressionMetrics object from CV summaries, computing RMSE if needed."""
    computed_rmse = rmse
    if computed_rmse is None and mse is not None:
        try:
            computed_rmse = float(mse ** 0.5)
        except Exception:
            computed_rmse = None

    metrics = RegressionMetrics(
        mae=float(mae) if mae is not None else float('nan'),
        mse=float(mse) if mse is not None else float('nan'),
        rmse=float(computed_rmse) if computed_rmse is not None else float('nan'),
        r2=float(r2) if r2 is not None else float('nan'),
        explained_variance=float(explained_variance) if explained_variance is not None else float('nan'),
        training_time=float(training_time) if training_time is not None else None,
        name=name,
    )

    return metrics


def aggregate_classification_cv_metrics(
    *,
    accuracy: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1_score_value: Optional[float] = None,
    roc_auc: Optional[float] = None,
    training_time: Optional[float] = None,
    name: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, Any]] = None,
    y_pred: Optional[Union[np.ndarray, Any]] = None,
    y_probs: Optional[Union[np.ndarray, Any]] = None,
) -> ClassificationMetrics:
    """Build a ClassificationMetrics object from CV summaries and optional OOF data."""
    cm = None
    roc_curve_data = None

    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)

    if y_true is not None and y_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_curve_data = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)
        if roc_auc is None:
            try:
                roc_auc = float(roc_auc_score(y_true, y_probs))
            except Exception:
                roc_auc = None

    metrics = ClassificationMetrics(
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        f1_score=float(f1_score_value) if f1_score_value is not None else float('nan'),
        precision=float(precision) if precision is not None else float('nan'),
        recall=float(recall) if recall is not None else float('nan'),
        accuracy=float(accuracy) if accuracy is not None else float('nan'),
        confusion_matrix=cm,
        training_time=float(training_time) if training_time is not None else None,
        name=name,
    )

    if roc_curve_data is not None:
        metrics.roc_curve = roc_curve_data

    return metrics


def _set_model_random_state(model: BaseEstimator, seed: Optional[int]) -> None:
    """Set random state for model if supported."""
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        elif hasattr(model, 'seed'):
            model.set_params(seed=seed)


def plot_metrics_heatmap(
    metrics: List[BaseMetrics], 
    title: str = 'Model Evaluation Metrics Comparison',
    figsize: Tuple[int, int] = (8, 4)
) -> None:
    """Plot heatmap of model metrics with per-column color scaling.
    
    Args:
        metrics: List of BaseMetrics objects
        title: Title for the plot
        figsize: Figure size tuple
    """
    # Convert BaseMetrics list to DataFrame
    rows: Dict[str, Dict[str, float]] = {}
    for i, m in enumerate(metrics):
        row_name = m.name if getattr(m, 'name', None) else f'Model {i+1}'
        rows[row_name] = m.get_numeric_metrics()
    metrics_df = pd.DataFrame.from_dict(rows, orient='index')
    
    plt.figure(figsize=figsize)
    
    # Normalize each column (metric) to 0-1 scale for color mapping
    normalized_df = metrics_df.copy()
    for col in metrics_df.columns:
        col_min = metrics_df[col].min()
        col_max = metrics_df[col].max()
        if col_max != col_min:  # Avoid division by zero
            normalized_df[col] = (metrics_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0.5  # Set to middle value if all values are the same
    
    # Plot using normalized data for colors but original data for annotations
    sns.heatmap(normalized_df, cmap='RdBu_r', annot=metrics_df, fmt=".3f", 
                cbar_kws={'label': 'Normalized Score (0-1 per metric)'})
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _evaluate_multiple_models(
    models: List[Tuple[str, BaseEstimator]], 
    evaluation_func: Callable,
    *args,
    **kwargs
) -> pd.DataFrame:
    """Generic function to evaluate multiple models and return metrics DataFrame."""
    all_metrics: Dict[str, Dict[str, Any]] = {}
    metrics_objects: List[BaseMetrics] = []
    
    for model_name, model in models:
        # Work with model copy to avoid modifying original models
        current_model = clone(model)
        eval_result = evaluation_func(current_model, model_name, *args, **kwargs)
        
        # Handle different return types
        if isinstance(eval_result, BaseMetrics):
            eval_result.name = model_name
            metrics_objects.append(eval_result)
            all_metrics[model_name] = eval_result.get_numeric_metrics()
        else:
            # Convert dict to GenericMetrics for plotting
            from models import GenericMetrics
            generic_metrics = GenericMetrics(values=eval_result, name=model_name)
            metrics_objects.append(generic_metrics)
            all_metrics[model_name] = eval_result
    
    # Plot using BaseMetrics objects
    plot_metrics_heatmap(metrics_objects)
    
    # Convert metrics to DataFrame for return
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    return metrics_df


def _evaluate_multiple_models_pydantic(
    models: List[Tuple[str, BaseEstimator]], 
    evaluation_func: Callable,
    task_type: str,
    *args,
    **kwargs
) -> MultipleModelResults:
    """Generic function to evaluate multiple models and return Pydantic model."""
    metrics_objects: List[BaseMetrics] = []
    
    for model_name, model in models:
        # Work with model copy to avoid modifying original models
        current_model = clone(model)
        eval_result = evaluation_func(current_model, model_name, *args, **kwargs)
        
        # Handle different return types
        if isinstance(eval_result, BaseMetrics):
            eval_result.name = model_name
            metrics_objects.append(eval_result)
        else:
            # Convert dict to GenericMetrics for plotting
            from models import GenericMetrics
            generic_metrics = GenericMetrics(values=eval_result, name=model_name)
            metrics_objects.append(generic_metrics)
    
    # Plot using BaseMetrics objects
    plot_metrics_heatmap(metrics_objects)
    
    # Return Pydantic model
    return MultipleModelResults(results=metrics_objects, task_type=task_type)


def train_evaluate_model(
    model: BaseEstimator, 
    model_name: str, 
    X_train: Any, 
    y_train: Any, 
    X_test: Any, 
    y_test: Any, 
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Train and evaluate a single model."""
    _set_model_random_state(model, seed)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # For ROC curve

    # Evaluate
    metrics_model = evaluate_classification(
        y_test=y_test,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model_name,
        enable_plot=False
    )
    return metrics_model.to_compact_dict()


def train_evaluate_model_cv(
    model: BaseEstimator, 
    model_name: str, 
    X: Any, 
    y: Any,
    preprocessor: Optional[Any] = None, 
    cv: int = 5, 
    seed: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    plot_feature_importance: bool = True,
    task_type: str = "classification"
) -> BaseMetrics:
    """Train and evaluate a model using cross-validation for both classification and regression."""
    _set_model_random_state(model, seed)

    # Create or extend pipeline with preprocessor and model
    if isinstance(preprocessor, Pipeline):
        # If preprocessor is already a pipeline, create a new pipeline with all steps
        steps = preprocessor.steps.copy()
        steps.append(('model', model))
        pipeline = Pipeline(steps)
    elif preprocessor is not None:
        # Create new pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # No preprocessor, just use the model
        pipeline = model

    # Define scoring metrics based on task type
    if task_type == "classification":
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, average='macro', zero_division=0),
            'roc_auc': 'roc_auc'
        }
    elif task_type == "regression":
        scoring = {
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'r2': 'r2',
            'explained_variance': 'explained_variance'
        }
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Must be 'classification' or 'regression'")

    # Measure training time and perform cross-validation
    start_time = time.time()
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )
    end_time = time.time()
    training_time = end_time - start_time

    # Create metrics object based on task type
    if task_type == "classification":
        cv_metrics = aggregate_classification_cv_metrics(
            accuracy=float(cv_results['test_accuracy'].mean()),
            precision=float(cv_results['test_precision'].mean()),
            recall=float(cv_results['test_recall'].mean()),
            f1_score_value=float(cv_results['test_f1'].mean()),
            roc_auc=float(cv_results['test_roc_auc'].mean()),
            training_time=float(training_time),
            name=model_name,
        )
        p.plot_classification_results(cv_metrics, model_name)
        
    elif task_type == "regression":
        cv_metrics = aggregate_regression_cv_metrics(
            mae=float(-cv_results['test_mae'].mean()),
            mse=float(-cv_results['test_mse'].mean()),
            r2=float(cv_results['test_r2'].mean()),
            explained_variance=float(cv_results['test_explained_variance'].mean()),
            training_time=float(training_time),
            name=model_name,
        )
        p.plot_regression_results(cv_metrics, model_name)

    # Plot feature importance if requested and model supports it
    if plot_feature_importance:
        _plot_feature_importance_cv(pipeline, model_name, feature_names, X, y)

    return cv_metrics


def _plot_feature_importance_cv(
    pipeline: Any, 
    model_name: str, 
    feature_names: Optional[List[str]], 
    X: Any,
    y: Any
) -> None:
    """Plot feature importance for cross-validation trained model."""
    try:
        # Extract the final model from pipeline
        if isinstance(pipeline, Pipeline):
            final_model = pipeline.named_steps['model']
        else:
            final_model = pipeline
        
        # Train the pipeline on full data to get feature importances
        pipeline.fit(X, y)
        
        # Check if model supports feature importance after training
        if hasattr(final_model, 'feature_importances_') or hasattr(final_model, 'coef_'):
            # Get the actual number of features from the trained model
            if hasattr(final_model, 'feature_importances_'):
                n_features = len(final_model.feature_importances_)
            else:  # hasattr(final_model, 'coef_')
                if len(final_model.coef_.shape) > 1:  # multi-class
                    n_features = final_model.coef_.shape[1]
                else:  # binary classification
                    n_features = len(final_model.coef_[0])
            
            # Extract meaningful feature names from the pipeline
            actual_feature_names = _extract_feature_names_from_pipeline(pipeline, X, n_features, feature_names)
            
            p.plot_feature_importance(final_model, actual_feature_names, top_n=20)
        else:
            print(f"Warning: Model {model_name} does not support feature importance")
    except Exception as e:
        print(f"Warning: Could not plot feature importance for {model_name}: {str(e)}")


def _extract_feature_names_from_pipeline(
    pipeline: Any, 
    X: Any, 
    n_features: int, 
    provided_feature_names: Optional[List[str]] = None
) -> List[str]:
    """Extract meaningful feature names from a trained pipeline."""
    # If provided feature names match the number of features, use them
    if provided_feature_names is not None and len(provided_feature_names) == n_features:
        return provided_feature_names
    
    # If no preprocessor or simple case, use original column names
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) == 1:
        if hasattr(X, 'columns') and len(X.columns) == n_features:
            return list(X.columns)
        else:
            return [f'feature_{i}' for i in range(n_features)]
    
    # Extract feature names from preprocessor steps
    feature_names = []
    
    # Get the preprocessor (everything except the final model)
    preprocessor_steps = pipeline.steps[:-1]  # All steps except the last (model)
    
    if len(preprocessor_steps) == 1:
        # Single preprocessor step
        step_name, preprocessor = preprocessor_steps[0]
        feature_names = _extract_feature_names_from_transformer(preprocessor, step_name)
    else:
        # Multiple preprocessor steps - combine feature names
        for step_name, transformer in preprocessor_steps:
            step_feature_names = _extract_feature_names_from_transformer(transformer, step_name)
            feature_names.extend(step_feature_names)
    
    # If we couldn't extract meaningful names, fall back to generic ones
    if len(feature_names) != n_features:
        if hasattr(X, 'columns') and len(X.columns) == n_features:
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return feature_names


def _extract_feature_names_from_transformer(transformer: Any, step_name: str) -> List[str]:
    """Extract feature names from a specific transformer."""
    try:
        # Handle ColumnTransformer
        if hasattr(transformer, 'transformers_'):
            feature_names = []
            for name, trans, columns in transformer.transformers_:
                if hasattr(trans, 'get_feature_names_out'):
                    # Get feature names from this transformer
                    trans_feature_names = trans.get_feature_names_out()
                    # Add prefix to distinguish features from different columns
                    if isinstance(columns, str):
                        prefix = f"{name}_{columns}_"
                    else:
                        prefix = f"{name}_"
                    prefixed_names = [f"{prefix}{name}" for name in trans_feature_names]
                    feature_names.extend(prefixed_names)
            return feature_names
        
        # Handle individual transformers
        elif hasattr(transformer, 'get_feature_names_out'):
            return list(transformer.get_feature_names_out())
        
        # Handle vectorizers with vocabulary
        elif hasattr(transformer, 'vocabulary_'):
            # For vectorizers, return the vocabulary keys (feature names)
            return list(transformer.vocabulary_.keys())
        
        # Handle transformers with feature_names_in_
        elif hasattr(transformer, 'feature_names_in_'):
            return list(transformer.feature_names_in_)
        
        # If no feature names can be extracted, return empty list
        return []
        
    except Exception:
        # If anything goes wrong, return empty list
        return []


def train_evaluate_models_cv(
    models: List[Tuple[str, BaseEstimator]], 
    X: Any, 
    y: Any, 
    preprocessor: Optional[Any] = None, 
    cv: int = 5, 
    seed: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    plot_feature_importance: bool = True,
    task_type: str = "classification"
) -> MultipleModelResults:
    """Train and evaluate multiple models using cross-validation for both classification and regression."""
    # Need to handle preprocessor cloning separately for CV
    def _cv_evaluation_wrapper(model: BaseEstimator, model_name: str) -> BaseMetrics:
        current_preprocessor = clone(preprocessor) if preprocessor is not None else None
        return train_evaluate_model_cv(
            model, model_name, X, y, current_preprocessor, cv, seed, 
            feature_names, plot_feature_importance, task_type
        )
    
    return _evaluate_multiple_models_pydantic(models, _cv_evaluation_wrapper, task_type)


def train_evaluate_models(
    models: List[Tuple[str, BaseEstimator]], 
    X_train: Any, 
    y_train: Any, 
    X_test: Any, 
    y_test: Any, 
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Train and evaluate multiple classification models."""
    return _evaluate_multiple_models(
        models, train_evaluate_model, X_train, y_train, X_test, y_test, seed=seed
    )


def winsorize_outliers(
    df: pd.DataFrame, 
    column_name: str, 
    lower_bound: Optional[float] = None, 
    upper_bound: Optional[float] = None
) -> pd.DataFrame:
    """Winsorize outliers by clipping values to specified bounds."""
    df = df.copy()
    
    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound
    
    return df

def train_pytorch_epoch(model: torch.nn.Module, dataloader: Any, optimizer: torch.optim.Optimizer, criterion: Any, device: torch.device) -> Tuple[float, float]:
    """
    Обучает PyTorch модель одну эпоху.
    Возвращает:
        Средний лосс за эпоху (float)
        Accuracy за эпоху (float)
    """
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    for batch_texts, batch_labels in dataloader:
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход. squeeze(1) убирает размерность [batch_size, 1] -> [batch_size]
        outputs = model(batch_texts).squeeze(1) 
        
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        # Статистика ошибки и метрик
        total_loss += loss.item() * batch_labels.size(0)
        
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        
        correct_preds += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
        
    return total_loss / total_samples, correct_preds / total_samples


def evaluate_pytorch_model(model: torch.nn.Module, dataloader: Any, device: torch.device, model_name: str = "PyTorch Model", enable_plot: bool = True) -> ClassificationMetrics:
    """
    Проверяет модель на валидационной выборке и собирает предсказания 
    в Pydantic-модель ClassificationMetrics через вашу функцию evaluate_classification.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_trues = []
    
    with torch.no_grad():
        for batch_texts, batch_labels in dataloader:
            batch_texts = batch_texts.to(device)
            
            outputs = model(batch_texts).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_trues.extend(batch_labels.numpy())
            
    # Конвертируем в чистые numpy-массивы для sklearn-метрик
    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Передаем данные в ВАШУ готовую функцию оценки из helper.py!
    metrics_model = evaluate_classification(
        y_test=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model_name,
        enable_plot=enable_plot
    )
    
    return metrics_model
def load_glove_embeddings(glove_file_path: str, token_to_id: dict, embedding_dim: int = 100) -> torch.Tensor:
    """
    Читает файл GloVe и строит матрицу эмбеддингов строго под ваш словарь токенов.
    """
    vocab_size = len(token_to_id)
    # Инициализируем матрицу случайными числами (для слов, которых не окажется в GloVe)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    # Явно зануляем вектор для <PAD> токена (обычно у него id = 0)
    if "<PAD>" in token_to_id:
        embedding_matrix[token_to_id["<PAD>"]] = np.zeros(embedding_dim)
        
    print(f"Парсинг файла GloVe: {glove_file_path}...")
    try:
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                # Если это слово есть в нашем словаре — заменяем случайный вектор на вектор GloVe
                if word in token_to_id:
                    idx = token_to_id[word]
                    embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
        print("Матрица эмбеддингов GloVe успешно собрана!")
    except FileNotFoundError:
        print(f"ВНИМАНИЕ: Файл {glove_file_path} не найден! Модель инициализирована случайными весами.")
        
    return torch.tensor(embedding_matrix, dtype=torch.float32)