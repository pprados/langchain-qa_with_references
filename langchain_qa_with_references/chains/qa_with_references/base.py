"""Question answering with references over documents."""

from __future__ import annotations

import logging
import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

from langchain.callbacks.base import Callbacks
from langchain.pydantic_v1 import Extra, Field
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.schema import BasePromptTemplate, BaseOutputParser
from .loading import (
    load_qa_with_references_chain,
)
from .map_reduce_prompt import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
    default_verbatim_parser,
)

logger = logging.getLogger(__name__)


class BaseQAWithReferencesChain(Chain, ABC):
    combine_documents_chain: BaseCombineDocumentsChain

    """Chain to use to combine documents."""
    question_key: str = "question"  #: :meta private:
    input_docs_key: str = "docs"  #: :meta private:
    answer_key: str = "answer"  #: :meta private:
    source_documents_key: str = "source_documents"  #: :meta private:

    verbatim_parser: BaseOutputParser = Field(default_factory=default_verbatim_parser)
    """ parser to extract verbatim"""
    """ Maximum number of verbatim by document"""

    map_parser: BaseOutputParser
    """If it's necessary to parse the output of map step."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        map_parser: BaseOutputParser = default_verbatim_parser(),
        **kwargs: Any,
    ) -> BaseQAWithReferencesChain:
        """Construct the chain from an LLM."""
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        combine_document_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            combine_document_chain=combine_results_chain,
            document_variable_name="context",
            return_intermediate_steps=True,
            # map_parser=map_parser,
        )
        return cls(
            combine_documents_chain=combine_document_chain,
            map_parser=map_parser
            **kwargs,
        )

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        map_parser: BaseOutputParser = default_verbatim_parser(),
        **kwargs: Any,
    ) -> BaseQAWithReferencesChain:
        """Load chain from chain type."""
        _chain_kwargs = chain_type_kwargs or {}
        combine_document_chain = load_qa_with_references_chain(
            llm,
            chain_type=chain_type,
            # map_parser=map_parser,
            **_chain_kwargs,
        )
        return cls(
            combine_documents_chain=combine_document_chain,
            map_parser=map_parser,
            **kwargs,
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        _output_keys = [self.answer_key, self.source_documents_key]
        return _output_keys

    @abstractmethod
    def _get_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(inputs)  # type: ignore[call-arg]

        # Inject position in the list
        for idx, doc in enumerate(docs):
            doc.metadata["_idx"] = idx

        answers = self.combine_documents_chain(
            {
                self.combine_documents_chain.input_key: docs,
                self.question_key: inputs[self.question_key],
            },
            callbacks=_run_manager.get_child(),
        )
        if "_idx" in answers:
            # Detect usage of MapRerank
            return {
                self.answer_key: answers[
                    self.combine_documents_chain.output_key
                ].strip(),
                self.source_documents_key: [docs[int(answers["_idx"])]],
            }

        answer, all_idx = self._split_answsers(answers, docs)
        selected_docs = [docs[idx] for idx in all_idx if idx < len(docs)]

        return {
            self.answer_key: answer,
            self.source_documents_key: selected_docs,
        }

    def _split_answsers(
        self, answers: str, docs: List[Document]
    ) -> Tuple[str, Set[int]]:
        # Add verbatim of the original document
        if "intermediate_steps" in answers:
            for i, str_verbatim in enumerate(answers["intermediate_steps"]):
                if str_verbatim.strip() != "None":
                    try:
                        extracted_verbatims = (
                            self.map_parser.parse(str_verbatim)
                        )
                        # Extract from the original verbatim from docs
                        if hasattr(
                            extracted_verbatims, "update_verbatims"
                        ) and callable(
                            getattr(extracted_verbatims, "update_verbatims")
                        ):
                            extracted_verbatims.update_verbatims(docs[i].page_content)
                        if extracted_verbatims.verbatims:
                            docs[i].metadata[
                                "verbatims"
                            ] = extracted_verbatims.verbatims
                    except OutputParserException as e:
                        logger.debug(f"Ignore output parserException ({e})")

        answer = answers[self.combine_documents_chain.output_key]
        s = re.match(r"(.*)\nIDXS:(.*)", answer, re.MULTILINE | re.DOTALL)
        if s:
            answer, idxs = s[1].strip(), s[2].strip()
        else:
            idxs = ""

        # Purge _idx
        for doc in docs:
            del doc.metadata["_idx"]
        return answer, set(map(int, re.split("[ ,]+", idxs))) if idxs else {}

    @abstractmethod
    async def _aget_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(inputs, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(inputs)  # type: ignore[call-arg]

        # Inject position in the list
        for idx, doc in enumerate(docs):
            doc.metadata["_idx"] = idx

        answers = await self.combine_documents_chain.acall(
            {
                self.combine_documents_chain.input_key: docs,
                self.input_keys[0]: inputs[self.input_keys[0]],
            },
            callbacks=_run_manager.get_child(),
        )
        if "_idx" in answers:
            # Detect usage of MapRerank
            return {
                self.answer_key: answers[
                    self.combine_documents_chain.output_key
                ].strip(),
                self.source_documents_key: [docs[int(answers["_idx"])]],
            }

        # Add verbatim of the original document
        if "intermediate_steps" in answers:
            for i, verbatim in enumerate(answers["intermediate_steps"]):
                verbatim = verbatim.strip()
                if verbatim in docs[i].page_content:
                    if verbatim != "None":
                        docs[i].metadata["verbatims"] = verbatim
                else:
                    pass  # No verbatim in the original document

        answer = answers[self.combine_documents_chain.output_key]
        s = re.match(r"(.*)IDXS:(.*)", answer, re.MULTILINE | re.DOTALL)
        if s:
            answer, idxs = s[1], s[2].strip()
        else:
            idxs = ""

        all_idx = set(map(int, re.split("[ ,]+", idxs))) if idxs else []
        selected_docs = [docs[idx] for idx in all_idx]

        return {
            self.answer_key: answer,
            self.source_documents_key: selected_docs,
        }


class QAWithReferencesChain(BaseQAWithReferencesChain):
    """
    Question answering with references over documents.

    This chain extracts the information from the documents that was used to answer the
    question. The output `source_documents` contains only the documents that were used,
    and for each one, only the text fragments that were used to answer are included.
    If possible, the list of text fragments that justify the answer is added to
    `metadata['verbatims']` for each document.

    The result["source_documents"] returns only the list of documents used, enriched
    if possible with the text extract used (doc.metadata["verbatims"]).
    """

    input_docs_key: str = "docs"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_docs_key, self.question_key]

    def _get_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""
        return inputs.pop(self.input_docs_key)

    async def _aget_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""
        return inputs.pop(self.input_docs_key)

    @property
    def _chain_type(self) -> str:
        return "qa_with_references_chain"
