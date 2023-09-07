"""Load question answering with reference chains.

We believe hallucinations pose a significant issue. Enabling straightforward referencing
of what led to the answer's derivation is imperative.

This chain extracts the information from the documents that was used to answer the
question. The output `source_documents` contains only the documents that were used.

Don't forget to activate: max_tokens=2000 or more,
"""
from .base import QAWithReferencesChain
from .retrieval import RetrievalQAWithReferencesChain

__all__ = ["QAWithReferencesChain", "RetrievalQAWithReferencesChain"]
