"""Load question answering with reference chains and verbatim.

We believe hallucinations pose a significant issue. Enabling straightforward referencing
of what led to the answer's derivation is imperative.

This chain extracts the information from the documents that was used to answer the
question. The output `source_documents` contains only the documents that were used,
and for each one, only the text fragments that were used to answer are included.
If possible, the list of text fragments that justify the answer is added to
`metadata['verbatims']` for each document.

"""
from .loading import (
    load_qa_with_references_chain,
)

__all__ = ["load_qa_with_references_chain"]
