"""**Chains** are easily reusable components linked together.

Chains encode a sequence of calls to components like models, document retrievers,
other Chains, etc., and provide a simple interface to this sequence.

The Chain interface makes it easy to create apps that are:

    - **Stateful:** add Memory to any Chain to give it state,
    - **Observable:** pass Callbacks to a Chain to execute additional functionality,
      like logging, outside the main sequence of component calls,
    - **Composable:** combine Chains with other components, including other Chains.

**Class hierarchy:**

.. code-block::

    Chain --> <name>Chain  # Examples: LLMChain, MapReduceChain, RouterChain
"""

from .qa_with_references import QAWithReferencesChain, RetrievalQAWithReferencesChain
from .qa_with_references_and_verbatims import (
    QAWithReferencesAndVerbatimsChain,
    RetrievalQAWithReferencesAndVerbatimsChain,
)

__all__ = [
    "QAWithReferencesChain",
    "RetrievalQAWithReferencesChain",
    "QAWithReferencesAndVerbatimsChain",
    "RetrievalQAWithReferencesAndVerbatimsChain",
]
