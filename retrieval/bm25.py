"""
bm25.py — Pure Python implementation of BM25 lexical search.
Avoids external dependencies and resolves word/camelCase/snake_case tokens nicely for code.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Any


class BM25Retriever:
    """A lightweight, pure-Python BM25 retriever for code chunks."""

    def __init__(self, b: float = 0.75, k1: float = 1.5) -> None:
        self.b = b
        self.k1 = k1
        self.documents: List[dict] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        # Map of doc_idx -> { term: count }
        self.term_freqs: List[Dict[str, int]] = []

    def tokenize(self, text: str) -> List[str]:
        """Tokenize code text, splitting camelCase, snake_case, and non-alphanumeric words."""
        if not text:
            return []
        # Extract raw alphanumeric words with original casing
        raw_tokens = re.findall(r"[a-zA-Z0-9]+", text)
        tokens = []
        for token in raw_tokens:
            # Further split camelCase if any
            camel_tokens = re.findall(r"[0-9]+|[a-z]+|[A-Z]+[a-z]*", token)
            if camel_tokens:
                tokens.extend([t.lower() for t in camel_tokens if t])
            else:
                tokens.append(token.lower())
        return tokens

    def fit(self, documents: List[dict]) -> None:
        """Fit BM25 parameters on a list of document chunks.

        Args:
            documents: List of dicts representing database chunks. Each dict must
                contain a "document" string.
        """
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.df = {}

        if self.doc_count == 0:
            self.avg_doc_len = 0.0
            return

        for doc in documents:
            text = doc.get("document", "")
            tokens = self.tokenize(text)
            self.doc_lengths.append(len(tokens))

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf.keys():
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / self.doc_count
        
        # Calculate IDF with standard BM25 formula
        for term, count in self.df.items():
            self.idf[term] = math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1.0)

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        """Retrieve top-k documents matching the query based on BM25 scores."""
        if not self.documents or self.doc_count == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            tf = self.term_freqs[i]
            doc_len = self.doc_lengths[i]

            for token in query_tokens:
                if token not in tf:
                    continue
                token_tf = tf[token]
                idf_score = self.idf.get(token, 0.0)

                num = token_tf * (self.k1 + 1)
                denom = token_tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_score * (num / denom)

            scores.append((score, i))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            doc = self.documents[idx].copy()
            doc["relevance_score"] = score
            results.append(doc)
        return results
