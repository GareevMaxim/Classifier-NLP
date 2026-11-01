from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Any, Union
import pandas as pd
import numpy as np
from collections import Counter
from nlp import compute_ngram_metrics

class NgramFeatureSelector(BaseEstimator, TransformerMixin):
    """Select and keep only top-k tokens in each token list."""

    def __init__(self, top_k: int = 100):
        self.top_k: int = top_k
        self.selected_tokens_: Optional[List[str]] = None

    def fit(self, X: Any, y: Optional[Any] = None):
        # X is an iterable of token lists
        if y is not None:
            ngram_metrics = compute_ngram_metrics(
                texts_tokenized=X,
                labels=y,
                n=1,
                metric='anova_f',
                min_count=100
            )
            self.selected_tokens_ = list(ngram_metrics.keys())[:self.top_k]
        else:
            from collections import Counter
            all_tokens = [token for tokens in X if tokens for token in tokens]
            token_counts = Counter(all_tokens)
            self.selected_tokens_ = [token for token, _ in token_counts.most_common(self.top_k)]

        return self

    def transform(self, X: Any):
        if self.selected_tokens_ is None:
            return X
        selected = set(self.selected_tokens_)
        # Keep original structure: list of tokens per sample
        return [[token for token in (tokens or []) if token in selected] for tokens in X]


class SequenceVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize token sequences into integer ids for embedding layers."""

    def __init__(
        self,
        min_frequency: int = 1,
        max_vocab_size: Optional[int] = None,
        pad_token: str = "<PAD>",
        oov_token: str = "<OOV>",
        pad_left: bool = True,
        max_sequence_length: Optional[int] = None,
        dtype: str = "int32",
        vocabulary: Optional[dict] = None,
    ) -> None:
        self.min_frequency: int = int(min_frequency)
        self.max_vocab_size: Optional[int] = int(max_vocab_size) if max_vocab_size is not None else None
        self.pad_token: str = pad_token
        self.oov_token: str = oov_token
        self.pad_left: bool = bool(pad_left)
        self.max_sequence_length: Optional[int] = int(max_sequence_length) if max_sequence_length is not None else None
        self.dtype: str = dtype
        self.vocabulary: Optional[dict] = vocabulary

        # Fitted attributes
        self.token_to_id_: Optional[dict] = None
        self.id_to_token_: Optional[List[str]] = None
        self.pad_id_: int = 0
        self.oov_id_: int = 1
        self.vocab_size_: Optional[int] = None
        self.sequence_length_: Optional[int] = None

    def fit(self, X: Any, y: Optional[Any] = None) -> "SequenceVectorizer":
        """Build vocabulary and sequence length from tokenized data."""
        if self.vocabulary is not None:
            self._set_vocabulary(self.vocabulary)
        else:
            tokens_iter = self._iter_tokens(X)
            counts: Counter = Counter(tokens_iter)

            # Remove special tokens from counts to avoid duplicates
            for special in (self.pad_token, self.oov_token):
                if special in counts:
                    del counts[special]

            # Apply frequency threshold
            items = [(tok, cnt) for tok, cnt in counts.items() if cnt >= self.min_frequency]
            # Sort by freq desc, then token asc for determinism
            items.sort(key=lambda x: (-x[1], x[0]))

            if self.max_vocab_size is not None:
                # Reserve 2 spots for PAD and OOV
                max_regular = max(0, int(self.max_vocab_size) - 2)
                items = items[:max_regular]

            vocab = {self.pad_token: 0, self.oov_token: 1}
            for idx, (tok, _cnt) in enumerate(items, start=2):
                vocab[tok] = idx
            self._set_vocabulary(vocab)

        # Decide sequence length
        if self.max_sequence_length is not None:
            self.sequence_length_ = int(self.max_sequence_length)
        else:
            self.sequence_length_ = self._infer_max_len(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Map tokens to ids and pad/truncate to fixed length."""
        if self.token_to_id_ is None or self.sequence_length_ is None:
            raise ValueError("Transformer must be fitted before transform")

        sequences: List[List[int]] = []
        to_id = self.token_to_id_
        oov_id = self.oov_id_
        max_len = int(self.sequence_length_)

        for tokens in self._iter_sequences(X):
            # Filter out tokens that don't exist in vocabulary (low frequency tokens)
            # Only keep tokens that are in the vocabulary
            filtered_tokens = [tok for tok in tokens if tok in to_id]
            seq = [to_id[tok] for tok in filtered_tokens]
            
            if len(seq) > max_len:
                if self.pad_left:
                    seq = seq[-max_len:]
                else:
                    seq = seq[:max_len]
            # Pad
            pad_needed = max_len - len(seq)
            if pad_needed > 0:
                pad_chunk = [self.pad_id_] * pad_needed
                seq = (pad_chunk + seq) if self.pad_left else (seq + pad_chunk)
            sequences.append(seq)

        return np.asarray(sequences, dtype=self.dtype)

    def inverse_transform(self, X: Any) -> List[List[str]]:
        """Convert id sequences back to tokens (PAD kept)."""
        if self.id_to_token_ is None:
            raise ValueError("Transformer must be fitted before inverse_transform")
        id_to_tok = self.id_to_token_
        result: List[List[str]] = []
        for row in np.asarray(X):
            result.append([id_to_tok[int(i)] if 0 <= int(i) < len(id_to_tok) else self.oov_token for i in row])
        return result

    def get_vocabulary(self) -> dict:
        """Return token->id vocabulary copy."""
        if self.token_to_id_ is None:
            raise ValueError("Transformer must be fitted before getting vocabulary")
        return dict(self.token_to_id_)

    # Internal helpers
    def _set_vocabulary(self, vocabulary: dict) -> None:
        self.token_to_id_ = dict(vocabulary)
        # Set special ids (fallback to defaults if not present)
        self.pad_id_ = int(self.token_to_id_.get(self.pad_token, 0))
        self.oov_id_ = int(self.token_to_id_.get(self.oov_token, 1))
        # Build reverse map sized to max id
        max_id = max(self.token_to_id_.values()) if self.token_to_id_ else 1
        id_to_token: List[str] = [self.oov_token] * (max_id + 1)
        for tok, idx in self.token_to_id_.items():
            if 0 <= int(idx) <= max_id:
                id_to_token[int(idx)] = tok
        self.id_to_token_ = id_to_token
        self.vocab_size_ = max_id + 1

    def _iter_tokens(self, X: Any):
        for seq in self._iter_sequences(X):
            for tok in seq:
                if tok is not None:
                    yield str(tok)

    def _iter_sequences(self, X: Any):
        if X is None:
            return []
        # Handle pandas objects explicitly to avoid treating DataFrame as an
        # iterable of column names (which would collapse samples).
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                series = X.iloc[:, 0]
            else:
                # Try to pick a list-like column; otherwise instruct user to pass a Series
                candidate_col = None
                for col in X.columns:
                    if len(X[col]) > 0 and isinstance(X[col].iloc[0], (list, tuple)):
                        candidate_col = col
                        break
                if candidate_col is None:
                    raise ValueError(
                        "SequenceVectorizer expected a single sequence column. Pass a Series of token lists or a one-column DataFrame."
                    )
                series = X[candidate_col]
            X = series.tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif hasattr(X, "tolist") and not isinstance(X, list):
            X = X.tolist()
        for item in X:
            if item is None:
                yield []
            elif isinstance(item, list):
                yield [str(t) for t in item]
            else:
                # Single string token or other -> wrap
                yield [str(item)]

    def _infer_max_len(self, X: Any) -> int:
        # Infer max length after applying vocabulary-based filtering,
        # so low-frequency tokens removed by min_frequency actually shorten sequences.
        max_len = 0
        vocab = self.token_to_id_ or {}
        for seq in self._iter_sequences(X):
            filtered_len = 0 if not vocab else sum(1 for t in seq if t in vocab)
            if filtered_len > max_len:
                max_len = filtered_len
        return int(max_len if max_len > 0 else 1)