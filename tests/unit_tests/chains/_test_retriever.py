import logging
import re
from typing import Any, Dict, Optional, Tuple, List

from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import Callbacks
from langchain.document_loaders.base import BaseLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms.base import BaseLLM
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.text_splitter import MarkdownTextSplitter, TextSplitter

from tests.unit_tests.fake_llm import FakeLLM

logger = logging.getLogger(__name__)

FAKE_LLM = False
VERBOSE_PROMPT = False
VERBOSE_RESULT = not FAKE_LLM
CALLBACKS: Callbacks = []

if VERBOSE_PROMPT or VERBOSE_RESULT:

    class ExStdOutCallbackHandler(StdOutCallbackHandler):
        def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "",
            **kwargs: Any,
        ) -> None:
            if VERBOSE_PROMPT:
                print("====")
                super().on_text(text=text, color=color, end=end)

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            """Ajoute une trace des outputs du llm"""
            if VERBOSE_RESULT:
                print("\n\033[1m> Finished chain with\033[0m")
                knows_keys = {
                    "answer",
                    "output_text",
                    "text",
                    "result",
                    "outputs",
                    "output",
                }
                if "outputs" in outputs:
                    print("\n\033[33m")
                    print(
                        "\n---\n".join(
                            [text["text"].strip() for text in outputs["outputs"]]
                        )
                    )
                    print("\n\033[0m")
                elif knows_keys.intersection(outputs):
                    # Prend la première cles en intersection
                    print(
                        f"\n\033[33m{outputs[next(iter(knows_keys.intersection(outputs)))]}\n\033[0m"
                    )
                else:
                    pass

    CALLBACKS = [ExStdOutCallbackHandler()]


def init_llm(queries: Dict[int, str]) -> BaseLLM:
    llm: BaseLLM
    if FAKE_LLM:  # Use simulation ?
        llm = FakeLLM(
            queries=queries,
            sequential_responses=True,
        )
    else:  # Not a simulation
        import langchain
        from dotenv import load_dotenv
        from langchain.cache import SQLiteCache

        load_dotenv()

        langchain.llm_cache = SQLiteCache(
            database_path="/tmp/cache_qa_with_reference.db"
        )
        llm = langchain.OpenAI(
            # temperature=0.5,
            # cache=False,
            max_tokens=2000,
        )
        logger.setLevel(logging.DEBUG)
    return llm


# %% -----------
# Simulate new methods for VectorStoreIndexWrapper
def query_with_reference(
    self: VectorStoreIndexWrapper,
    question: str,
    llm: Optional[BaseLanguageModel] = None,
    retriever_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """Query the vectorstore and get back sources."""
    # from langchain_qa_with_references.chains import RetrievalQAWithReferencesChain
    from langchain import OpenAI

    from langchain_qa_with_references.chains import RetrievalQAWithReferencesChain

    llm = llm or OpenAI(temperature=0)
    retriever_kwargs = retriever_kwargs or {}
    chain = RetrievalQAWithReferencesChain.from_chain_type(
        llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
    )
    return chain({chain.question_key: question})


def query_with_reference_and_verbatims(
    self: VectorStoreIndexWrapper,
    question: str,
    llm: Optional[BaseLanguageModel] = None,
    retriever_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """Query the vectorstore and get back sources."""
    # from langchain_qa_with_references.chains import \
    #     RetrievalQAWithReferencesAndVerbatimsChain
    from langchain import OpenAI

    from langchain_qa_with_references.chains import RetrievalQAWithReferencesAndVerbatimsChain

    llm = llm or OpenAI(temperature=0)
    retriever_kwargs = retriever_kwargs or {}
    chain = RetrievalQAWithReferencesAndVerbatimsChain.from_chain_type(
        llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
    )
    return chain({chain.question_key: question})


setattr(VectorStoreIndexWrapper, "query_with_reference", query_with_reference)
setattr(
    VectorStoreIndexWrapper,
    "query_with_reference_and_verbatims",
    query_with_reference_and_verbatims,
)


# %% -----------


def _test_loader(type: str, llm: BaseLLM) -> BaseLoader:
    if type == "docs":
        from langchain.document_loaders import DirectoryLoader, TextLoader

        return DirectoryLoader("../../docs/", glob="**/*.md", loader_cls=TextLoader)
    elif type == "apify":
        import os

        from langchain.utilities import ApifyWrapper

        if "APIFY_API_TOKEN" not in os.environ:
            os.environ["APIFY_API_TOKEN"] = "Your Apify API token"

        apify = ApifyWrapper()
        return apify.call_actor(
            actor_id="apify/website-content-crawler",
            run_input={
                "startUrls": [{"url": "https://python.langchain.com/en/latest/"}]
            },
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "", metadata={"source": item["url"]}
            ),
        )
    else:
        assert False


def _test_index(
    type: str,
    llm: BaseLLM,
    text_splitter: TextSplitter = MarkdownTextSplitter(
        chunk_size=1500, chunk_overlap=100, length_function=len
    ),
    embedding=None,
) -> VectorStoreIndexWrapper:
    from langchain.embeddings import HuggingFaceEmbeddings

    _embedding = embedding or HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if type in ["docs", "apify"]:
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.vectorstores import Chroma

        loader = _test_loader(type, llm)
        index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            text_splitter=text_splitter,
            embedding=_embedding,
            vectorstore_kwargs={},
        ).from_loaders([loader])
        return index
    else:
        assert None


def test_retriever(type: str, llm: BaseLLM) -> Tuple[BaseRetriever, str]:
    retriever: BaseRetriever
    if type == "wikipedia":
        question = "what can you say about ukraine?"
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.retrievers import WikipediaRetriever
        from langchain.vectorstores import Chroma

        wikipedia_retriever = WikipediaRetriever()
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory="/tmp/chroma_db_oai",
        )
        vectorstore.add_documents(wikipedia_retriever.get_relevant_documents(question))
        retriever = vectorstore.as_retriever()
    elif type == "google":
        import dotenv
        from langchain import GoogleSearchAPIWrapper
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.retrievers import WebResearchRetriever
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import Chroma

        question = "what's the news from ukraine?"

        dotenv.load_dotenv(override=True)
        # Search
        search = GoogleSearchAPIWrapper()

        # Vectorstore
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory="/tmp/chroma_db_oai",
        )
        # Initialize
        retriever = WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=llm,
            search=search,
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=50
            ),
        )
    elif type == "docs":
        index = _test_index(type, llm)
        retriever = index.vectorstore.as_retriever(search_kwargs={"k": 4})
        question = "How do I use OpenAI?"
    elif type == "apify":
        index = _test_index(type, llm)
        retriever = index.vectorstore.as_retriever(search_kwargs={"k": 4})
        question = "How do I use OpenAI?"
    else:
        raise ValueError()

    return retriever, question

def compare_words_of_responses(response:str, assert_response:str) -> bool:
    """The exact format of verbatim may be changed by the LLM.
    Extract only the words of the verbatim, and try to find a sequence
    of same words in the original document.
    """
    only_words = filter(len, re.split(r"[^\w]+", assert_response))
    regex_for_words_in_same_oder = (
            r"(?i)\b" + r"\b[^\w]+".join(only_words) + r"\b"
                                                       r"\s*[.!?:;]?"
    )
    match = re.search(regex_for_words_in_same_oder, response, re.IGNORECASE)
    if match:
        return True
    return False  # No verbatim found in the original document

def compare_responses(responses:List[str], assert_responses:List[str]) -> bool:
    for response, assert_response in zip(responses,assert_responses):
        if not compare_words_of_responses(response, assert_response):
            return False
    return True