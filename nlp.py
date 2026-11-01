import tiktoken
from typing import List, Sequence, Set, Any, Tuple, Optional

import torch
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def tokenize_tiktoken(text: str) -> List[str]:
    """Tokenize text with tiktoken and return decoded byte tokens."""
    token_bytes = [
        TIKTOKEN_ENCODING.decode_single_token_bytes(token)
        for token in TIKTOKEN_ENCODING.encode(text)
    ]
    tokens = [token.decode('utf-8', errors='replace').strip().lower() for token in token_bytes]
    return [token for token in tokens if token]  # Remove empty tokens

def get_stopwords_set(custom_stopwords: Set[str] = None) -> Set[str]:
    """Return English stopwords set combined with custom ones."""
    default_stopwords = set(stopwords.words('english'))
    if custom_stopwords:
        return default_stopwords.union(custom_stopwords)
    return default_stopwords

# -------------------------------
# Internal helpers (DRY)
# -------------------------------

def _get_punctuation_set(custom_punctuation: Set[str] = None) -> Set[str]:
    """Return punctuation set combined with custom ones."""
    default_punctuation = set(string.punctuation)
    if custom_punctuation:
        return default_punctuation.union(custom_punctuation)
    return default_punctuation

def _is_punctuation_token(token: str, custom_punctuation: Set[str] = None) -> bool:
    """Return True if token is punctuation or all chars are punctuation."""
    punctuation_set = _get_punctuation_set(custom_punctuation)
    stripped = token.strip()
    return stripped in punctuation_set or (stripped != '' and all(c in punctuation_set for c in stripped))


def filter_tokens(
    tokens: Sequence[str],
    *,
    remove_stopwords: bool = False,
    remove_punct_tokens: bool = False,
    custom_stopwords: Set[str] = None,
    custom_punctuation: Set[str] = None,
    lowercase_for_counting: bool = False,
) -> List[str]:
    """Apply common token filters in one pass. Optionally converts to lowercase for counting purposes."""
    stop_words = get_stopwords_set(custom_stopwords) if remove_stopwords else None
    result: List[str] = []

    # print(tokens)
    for tok in tokens:
        t = str(tok).strip()  # Only strip whitespace, don't modify the token content
        if not t:
            continue
        if remove_punct_tokens and _is_punctuation_token(t, custom_punctuation):
            continue
        if stop_words is not None and t.lower() in stop_words:
            continue
            
        # Apply lowercase conversion if requested for counting purposes
        final_token = t.lower() if lowercase_for_counting else t
        result.append(final_token)
    return result

def _flatten_array_column(df: pd.DataFrame, column_name: str) -> List[Any]:
    """Flatten list-like column into a Python list."""
    return [item for sublist in df[column_name] for item in sublist]

def _value_counts(values: Sequence[Any]) -> pd.Series:
    """Return pandas value_counts for a sequence."""
    return pd.Series(list(values)).value_counts()

def _generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def generate_all_ngrams(tokens: List[str], max_n: int) -> List[Tuple[str, ...]]:
    """Generate all n-grams from 1 to max_n for a given list of tokens."""
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    
    all_ngrams = []
    for i in range(1, max_n + 1):
        all_ngrams.extend(_generate_ngrams(tokens, i))
    return all_ngrams


def stopwords_count(df: pd.DataFrame, column_name: str, custom_stopwords: Set[str] = None) -> pd.Series:
    """Count stopwords in a list-like column."""
    all_items = _flatten_array_column(df, column_name)
    stop_words = get_stopwords_set(custom_stopwords)
    filtered = [t for t in all_items if str(t).strip().lower() in stop_words]
    return _value_counts(filtered)

def punctuation_counts(df: pd.DataFrame, column_name: str, custom_punctuation: Set[str] = None) -> pd.Series:
    """Count punctuation tokens in a list-like column."""
    all_items = _flatten_array_column(df, column_name)
    punct = [t for t in all_items if _is_punctuation_token(str(t), custom_punctuation)]
    return _value_counts(punct)

def token_counts(
    df: pd.DataFrame,
    column_name: str,
    remove_stopwords: bool = False,
    remove_punctuation: bool = False,
    custom_stopwords: Set[str] = None,
    custom_punctuation: Set[str] = None,
    lowercase_for_counting: bool = False
) -> pd.Series:
    """Count tokens with optional stopword/punctuation filtering and lowercase for counting."""
    all_items = _flatten_array_column(df, column_name)
    processed = filter_tokens(
        [str(item) for item in all_items],
        remove_stopwords=remove_stopwords,
        remove_punct_tokens=remove_punctuation,
        custom_stopwords=custom_stopwords,
        custom_punctuation=custom_punctuation,
        lowercase_for_counting=lowercase_for_counting,
    )
    return _value_counts(processed)

def compute_ngram_metrics(
    texts_tokenized: List[List[str]],
    labels: List[int],
    n: int = 1,
    metric: str = 'anova_f',
    min_count: int = 1,
) -> dict[str, float]:
    """Compute n-gram metrics for a multi-class classification corpus.
    
    Args:
        texts_tokenized: List of tokenized texts
        labels: Class labels (any integers)
        n: Maximum n-gram size to analyze (includes all 1-grams up to n-grams, default: 1)
        metric: Feature selection metric to compute ('anova_f' or 'mutual_info', default: 'anova_f')
        min_count: Minimum number of times a token/n-gram must appear to be included (default: 1)
    
    Returns:
        Dictionary mapping n-gram strings to their metric values, sorted by metric value descending
    """
    
    # Validate input
    if len(texts_tokenized) != len(labels):
        raise ValueError("Length of texts_tokenized and labels must be equal")
    
    if n < 1:
        raise ValueError("n must be >= 1")
    
    if metric not in ['anova_f', 'mutual_info']:
        raise ValueError("metric must be 'anova_f' or 'mutual_info'")
    
    # Get unique classes
    unique_classes = sorted(set(labels))
    
    # Generate all n-grams from 1 to n and separate by class
    ngrams_by_class = {cls: [] for cls in unique_classes}
    
    for tokens, label in zip(texts_tokenized, labels):
        # Generate all n-grams from 1 to n
        all_ngrams = generate_all_ngrams(tokens, n)
        
        ngrams_by_class[label].extend(all_ngrams)
    
    # Calculate frequencies for each class
    freq_by_class = {cls: Counter(ngrams_by_class[cls]) for cls in unique_classes}
    
    # Get all unique n-grams across all classes
    all_ngrams = set()
    for freq_dict in freq_by_class.values():
        all_ngrams.update(freq_dict.keys())
    
    # Filter n-grams by minimum count threshold
    if min_count > 1:
        # Calculate total count for each n-gram across all classes
        total_counts = Counter()
        for freq_dict in freq_by_class.values():
            total_counts.update(freq_dict)
        
        # Keep only n-grams that meet the minimum count threshold
        filtered_ngrams = {ngram for ngram in all_ngrams if total_counts[ngram] >= min_count}
        
        # Update all_ngrams and filter freq_by_class dictionaries
        all_ngrams = filtered_ngrams
        for cls in unique_classes:
            freq_by_class[cls] = {ngram: count for ngram, count in freq_by_class[cls].items() 
                                 if ngram in filtered_ngrams}
    
    # Prepare feature matrix for ANOVA F-value or Mutual Information computation
    # Create a document-term matrix where rows are documents and columns are n-grams
    vocab = list(all_ngrams)
    vocab_to_idx = {ngram: i for i, ngram in enumerate(vocab)}
    
    # Create feature matrix (documents x n-grams)
    X = np.zeros((len(texts_tokenized), len(vocab)))
    
    for doc_idx, tokens in enumerate(texts_tokenized):
        # Generate all n-grams for this document
        doc_ngrams = generate_all_ngrams(tokens, n)
        
        # Count n-grams in this document
        doc_ngram_counts = Counter(doc_ngrams)
        
        # Fill the feature matrix
        for ngram, count in doc_ngram_counts.items():
            if ngram in vocab_to_idx:
                X[doc_idx, vocab_to_idx[ngram]] = count
    
    # Encode labels for sklearn
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Compute feature selection metric
    if metric == 'anova_f':
        # ANOVA F-value
        f_scores, p_values = f_classif(X, y_encoded)
        metric_values = f_scores
    else:  # mutual_info
        # Mutual Information
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        metric_values = mi_scores
    
    # Create dictionary mapping n-gram strings to metric values
    ngram_to_string = {ngram: ' '.join(ngram) for ngram in all_ngrams}
    metric_dict = {}
    for i, ngram in enumerate(vocab):
        token_str = ngram_to_string[ngram]
        metric_dict[token_str] = metric_values[i]
    
    # Sort by metric value (descending order - higher scores first)
    return dict(sorted(metric_dict.items(), key=lambda x: x[1], reverse=True))


def count_based_analysis(
    texts_tokenized: List[List[str]],
    labels: List[int],
    n: int = 1,
    metric: str = 'anova_f',
    min_count: int = 1,
) -> pd.DataFrame:
    """Compute n-gram stats for a multi-class classification corpus.
    
    Args:
        texts_tokenized: List of tokenized texts
        labels: Class labels (any integers)
        n: Maximum n-gram size to analyze (includes all 1-grams up to n-grams, default: 1)
        metric: Feature selection metric to compute ('anova_f' or 'mutual_info', default: 'anova_f')
        min_count: Minimum number of times a token/n-gram must appear to be included (default: 1)
    """
    
    # Get metric values using the extracted method
    metric_dict = compute_ngram_metrics(texts_tokenized, labels, n, metric, min_count)
    
    # Get unique classes for building count/frequency columns
    unique_classes = sorted(set(labels))
    
    # Generate all n-grams from 1 to n and separate by class (needed for detailed stats)
    ngrams_by_class = {cls: [] for cls in unique_classes}
    
    for tokens, label in zip(texts_tokenized, labels):
        # Generate all n-grams from 1 to n
        all_ngrams = generate_all_ngrams(tokens, n)
        
        ngrams_by_class[label].extend(all_ngrams)
    
    # Calculate frequencies for each class
    freq_by_class = {cls: Counter(ngrams_by_class[cls]) for cls in unique_classes}
    
    # Filter by minimum count (consistent with _compute_ngram_metrics)
    if min_count > 1:
        total_counts = Counter()
        for freq_dict in freq_by_class.values():
            total_counts.update(freq_dict)
        
        filtered_ngrams = {ngram for ngram in total_counts if total_counts[ngram] >= min_count}
        
        for cls in unique_classes:
            freq_by_class[cls] = {ngram: count for ngram, count in freq_by_class[cls].items() 
                                 if ngram in filtered_ngrams}
    
    # Create DataFrame from metric dictionary
    df = pd.DataFrame([
        {'token': token, 'metric': metric_value} 
        for token, metric_value in metric_dict.items()
    ])
    
    # Add count and frequency columns for each class
    for cls in unique_classes:
        # Count columns
        count_col = f'count_{cls}'
        freq_col = f'freq_{cls}'
        
        # Create frequency mapping with string keys
        ngram_to_string = {}
        for ngram in freq_by_class[cls].keys():
            ngram_to_string[' '.join(ngram)] = freq_by_class[cls][ngram]
        
        # Merge counts
        df = df.merge(
            pd.Series(ngram_to_string, name=count_col).reset_index().rename(columns={'index': 'token'}),
            on='token', how='left'
        ).fillna(0)
        
        # Convert counts to integers
        df[count_col] = df[count_col].astype(int)
        
        # Calculate frequencies
        total_ngrams = len(ngrams_by_class[cls])
        df[freq_col] = df[count_col] / total_ngrams if total_ngrams > 0 else 0
    
    # Calculate total count across all classes
    count_cols = [f'count_{cls}' for cls in unique_classes]
    df['total_count'] = df[count_cols].sum(axis=1)
    
    return df.reset_index(drop=True)

def filter_tokens_column(df: pd.DataFrame, column_name: str, tokens: List[str]) -> pd.DataFrame:
    """Filter tokens in a column."""
    return df[column_name].transform(
        lambda arr: [item for item in arr if item in tokens]
    )


def filter_tokens_by_frequency(
    df: pd.DataFrame, 
    text_columns: List[str], 
    min_frequency: int = 50
) -> pd.DataFrame:
    """Filter tokens that appear less than min_frequency times across all text columns.
    
    Args:
        df: DataFrame with tokenized text columns (lists of tokens)
        text_columns: List of column names containing tokenized text
        min_frequency: Minimum frequency threshold for keeping tokens
        
    Returns:
        DataFrame with filtered tokens in the specified columns
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from collections import Counter
    
    # Flatten all tokens from all specified columns
    all_tokens = []
    for col in text_columns:
        if col in df.columns:
            # Flatten the list of lists into a single list
            column_tokens = [token for token_list in df[col] for token in token_list]
            all_tokens.extend(column_tokens)
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Get tokens that meet the minimum frequency threshold
    frequent_tokens = {token for token, count in token_counts.items() if count >= min_frequency}
    
    # Filter tokens in each column
    df_filtered = df.copy()
    for col in text_columns:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].apply(
                lambda token_list: [token for token in token_list if token in frequent_tokens]
            )
    
    return df_filtered

class IMDBDataset(Dataset):
    """
    Легковесный датасет PyTorch. 
    Принимает уже готовую числовую матрицу после SequenceVectorizer.
    """
    def __init__(self, vectorized_sequences: np.ndarray, labels: Optional[pd.Series] = None):
        """
        Args:
            vectorized_sequences: Матрица индексов shape (samples, seq_len) от SequenceVectorizer.
            labels: Серия с таргетом (positive/negative или 1/0). Если None — это тест.
        """
        self.sequences = vectorized_sequences
        self.labels = labels.reset_index(drop=True) if labels is not None else None
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx: int):
        # Превращаем строку матрицы в тензор для PyTorch
        x_tensor = torch.tensor(self.sequences[idx], dtype=torch.long)
        
        # Если есть метки (обучение/валидация)
        if self.labels is not None:
            raw_label = self.labels.iloc[idx]
            # Обрабатываем строковые метки, если они не переведены в числа в EDA
            if isinstance(raw_label, str):
                label_val = 1.0 if raw_label.lower() == 'positive' else 0.0
            else:
                label_val = float(raw_label)
                
            return x_tensor, torch.tensor(label_val, dtype=torch.float32)
            
        # Если меток нет (тестовый сабмишн)
        return x_tensor