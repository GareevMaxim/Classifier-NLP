from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator


class BaseMetrics(BaseModel):
    name: Optional[str] = None
    training_time: Optional[float] = Field(default=None, description="Training time in seconds")
    estimators: Optional[Any] = Field(default=None, description="Trained estimators from cross-validation")

    def get_numeric_metrics(self) -> Dict[str, float]:
        raise NotImplementedError


class ClassificationReportRow(BaseModel):
    class_label: str
    precision: float
    recall: float


class RocCurveData(BaseModel):
    fpr: Any
    tpr: Any
    thresholds: Any


class ClassificationMetrics(BaseMetrics):
    roc_auc: Optional[float] = Field(default=None, description="Area under ROC curve")
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    confusion_matrix: Any
    classification_report: Optional[List[ClassificationReportRow]] = None
    roc_curve: Optional[RocCurveData] = None

    @validator("accuracy", "f1_score", "precision", "recall", pre=True)
    def _ensure_float(cls, v: Any) -> float:  # type: ignore[override]
        if v is None:
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    def get_numeric_metrics(self) -> Dict[str, float]:
        metrics = {
            'ROC AUC': float(self.roc_auc) if self.roc_auc is not None else float('nan'),
            'F1 Score': float(self.f1_score),
            'Precision': float(self.precision),
            'Recall': float(self.recall),
            'Accuracy': float(self.accuracy),
        }
        if self.training_time is not None:
            metrics['Training Time (s)'] = float(self.training_time)
        return metrics

    def to_compact_dict(self) -> Dict[str, Any]:
        """Minimal metrics dict used in returns (no heavy arrays)."""
        return self.get_numeric_metrics()

    def to_plot_dict(self) -> Dict[str, Any]:
        """Dict compatible with plotting helpers expecting specific keys."""
        out: Dict[str, Any] = {}
        if self.confusion_matrix is not None:
            out['Confusion Matrix'] = self.confusion_matrix
        if self.roc_curve is not None and self.roc_auc is not None:
            out['ROC Curve'] = {
                'fpr': self.roc_curve.fpr,
                'tpr': self.roc_curve.tpr,
                'thresholds': self.roc_curve.thresholds,
            }
            out['ROC AUC'] = self.roc_auc
        return out

    def to_report_dict(self) -> Dict[str, Any]:
        """Dict used by print helpers for tabular display."""
        report = {
            'ROC AUC': self.roc_auc,
            'F1 Score': self.f1_score,
            'Precision': self.precision,
            'Recall': self.recall,
            'Accuracy': self.accuracy,
        }
        if self.classification_report is not None:
            report['Classification Report'] = {
                'Class': [row.class_label for row in self.classification_report],
                'Precision': [row.precision for row in self.classification_report],
                'Recall': [row.recall for row in self.classification_report],
            }
        return report


class RegressionMetrics(BaseMetrics):
    mae: float
    mse: float
    rmse: float
    r2: float
    explained_variance: Optional[float] = None

    def _as_float(self, v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float('nan')

    def get_numeric_metrics(self) -> Dict[str, float]:
        data: Dict[str, float] = {
            'MAE': self._as_float(self.mae),
            'MSE': self._as_float(self.mse),
            'RMSE': self._as_float(self.rmse),
            'R2': self._as_float(self.r2),
        }
        if self.explained_variance is not None:
            data['Explained Variance'] = self._as_float(self.explained_variance)
        if self.training_time is not None:
            data['Training Time (s)'] = float(self.training_time)
        return data


class GenericMetrics(BaseMetrics):
    values: Dict[str, float]

    def get_numeric_metrics(self) -> Dict[str, float]:
        return dict(self.values)


class MultipleModelResults(BaseModel):
    """Container for multiple model evaluation results."""
    results: List[BaseMetrics] = Field(description="List of model evaluation results")
    task_type: str = Field(description="Type of task: 'classification' or 'regression'")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for backward compatibility."""
        all_metrics = {}
        for result in self.results:
            model_name = result.name if result.name else "Unknown Model"
            all_metrics[model_name] = result.get_numeric_metrics()
        return pd.DataFrame.from_dict(all_metrics, orient='index')
    
    def get_model_names(self) -> List[str]:
        """Get list of model names."""
        return [result.name for result in self.results if result.name]
    
    def get_best_model(self, metric: str = None) -> Optional[BaseMetrics]:
        """Get the best performing model based on a specific metric."""
        if not self.results:
            return None
        
        if metric is None:
            # Use default metric based on task type
            metric = 'ROC AUC' if self.task_type == 'classification' else 'R2'
        
        best_model = None
        best_score = float('-inf') if self.task_type == 'classification' else float('-inf')
        
        for result in self.results:
            metrics = result.get_numeric_metrics()
            if metric in metrics:
                score = metrics[metric]
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_model = result
        
        return best_model

class TextCNNClassifier(nn.Module):
    """
    Быстрая сверточная нейросеть для классификации текстов (Text-CNN).
    Использует несколько параллельных окон свертки разного размера (например, 3, 4, 5 слов).
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, num_filters: int = 100, 
                 filter_sizes: list[int] =[3, 4, 5], output_dim: int = 1, dropout: float = 0.5):
        super().__init__()
        
        # Слой эмбеддингов. Задаем padding_idx=0, чтобы маскировать нули
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Создаем список сверточных слоев для окон разной длины
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=num_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        # Финальный полносвязный слой для классификации
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text размерность: [batch_size, seq_len]
        
        embedded = self.embedding(text) # -> [batch_size, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1) # -> [batch_size, 1, seq_len, embedding_dim] (добавляем псевдо-канал)
        
        # Прогоняем через свертки, функцию активации ReLU и Max-Pooling вдоль всей длины текста
        pooled_outputs = []
        for conv in self.convs:
            x = F.relu(conv(embedded)).squeeze(3) # -> [batch_size, num_filters, seq_len - filter_size + 1]
            x = F.max_pool1d(x, x.size(2)).squeeze(2) # -> [batch_size, num_filters]
            pooled_outputs.append(x)
            
        # Объединяем карты признаков от всех фильтров вместе
        cat = self.dropout(torch.cat(pooled_outputs, dim=1)) # -> [batch_size, num_filters * len(filter_sizes)]
        
        return self.fc(cat) # -> [batch_size, output_dim]


class LSTMClassifier(nn.Module):
    """
    Стандартная рекуррентная нейросеть (LSTM) для извлечения долгосрочных зависимостей в тексте.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128, 
                 num_layers: int = 1, output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Слой LSTM. Флаг batch_first=True критически важен для правильной размерности тензоров
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text размерность: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(text)) # -> [batch_size, seq_len, embedding_dim]
        
        # Прогоняем последовательность через ячейки LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Нам нужно только скрытое состояние последнего временного шага (последнего слова текста)
        # hidden имеет размерность [num_layers, batch_size, hidden_dim]
        last_hidden = hidden[-1, :, :] # -> [batch_size, hidden_dim]
        
        return self.fc(self.dropout(last_hidden)) # -> [batch_size, output_dim]
class RNNClassifier(nn.Module):
    """
    Простая базовая рекуррентная нейросеть (Vanilla RNN) для классификации текстов.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128, 
                 num_layers: int = 1, output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Используем базовый слой nn.RNN вместо nn.LSTM
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text)) # -> [batch_size, seq_len, embedding_dim]
        
        # Прогоняем через базовую RNN
        rnn_out, hidden = self.rnn(embedded)
        
        # Берём скрытое состояние последнего временного шага
        last_hidden = hidden[-1, :, :] # -> [batch_size, hidden_dim]
        
        return self.fc(self.dropout(last_hidden)) # -> [batch_size,output_dim]

class GloVeTextCNNClassifier(nn.Module):
    """
    Архитектура Text-CNN, использующая предобученные замороженные векторы GloVe.
    """
    def __init__(self, embedding_matrix: torch.Tensor, num_filters: int = 100, 
                 filter_sizes: list[int] =[3, 4, 5], output_dim: int = 1, dropout: float = 0.5, freeze: bool = True):
        super().__init__()
        
        # Загружаем готовые веса из матрицы GloVe
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=freeze,       # True означает, что веса слов НЕ будут меняться при обучении
            padding_idx=0        # Маскируем нули для PAD-токенов
        )
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=num_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text).unsqueeze(1) # -> [batch_size, 1, seq_len, embedding_dim]
        
        pooled_outputs = []
        for conv in self.convs:
            x = F.relu(conv(embedded)).squeeze(3)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            pooled_outputs.append(x)
            
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))
        return self.fc(cat)
