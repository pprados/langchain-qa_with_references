import re
from typing import Dict, List, Set, Tuple

import pytest
from langchain.schema import Document

from langchain_qa_with_references.chains import (
    QAWithReferencesAndVerbatimsChain,
)

from ._test_retriever import CALLBACKS, init_llm, logger, compare_words_of_responses, compare_responses


@pytest.mark.parametrize(
    "question,docs,map_responses",
    [
        (
            "what does it eat?",
            [
                Document(
                    page_content="The night is black.",
                    metadata={},
                ),
                Document(
                    page_content="He eats\napples and plays football. "
                    "My name is Philippe. He eats pears.",
                    metadata={},
                ),
                Document(
                    page_content="He eats carrots. I like football.",
                    metadata={},
                ),
                Document(
                    page_content="The Earth is round.",
                    metadata={},
                ),
            ],
            {
                "stuff": (
                    {
                        0: "```"
                        '{"response": "He eats apples, pears and carrots.", '
                        '"documents": ['
                        '{"ids": [1, 2], "verbatims": ['
                        '"He eats apples and plays football.", '
                        '"He eats pears.", '
                        '"He eats carrots."]}]}```'
                    },
                    [
                        ["He eats\napples and plays football.", "He eats pears."],
                        [
                            "He eats carrots.",
                        ],
                    ],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_reduce": (
                    {
                        0: 'Output: {"ids": [], "verbatims": []}',
                        1: 'Output: {"ids": [], "verbatims": ["He eats apples", '
                        '"He eats pears"]}',
                        2: 'Output: {"ids": [], "verbatims": ["He eats carrots."]}',
                        3: 'Output: {"ids": [], "verbatims": []}',
                        4: '{"response": "He eats apples, He eats pears, '
                        'He eats carrots.", "documents": [{"ids": [1], '
                        '"verbatims": ["He eats apples", "He eats pears"]}, '
                        '{"ids": [2], "verbatims": ["He eats carrots."]}]}',
                    },
                    [["eats apples", "eats pears"], ["eats carrots"]],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "refine": (
                    {
                        0: "The output would be:"
                        "{\n"
                        '   "response": "I don\'t know",\n'
                        '    "documents": [\n'
                        "        {\n"
                        '            "ids": [0],\n'
                        '            "verbatims": []\n'
                        "        }\n"
                        "    ]\n"
                        "}\n",
                        1: "The output would be:"
                        "{\n"
                        '    "response": "He eats apples and pears",\n'
                        '    "documents": [\n'
                        "        {\n"
                        '            "ids": [1],\n'
                        '            "verbatims": ["He eats apples and plays football."'
                        ', "He eats pears."]\n'
                        "        }\n"
                        "    ]\n"
                        "}\n",
                        2: "The output would be:"
                        "{\n"
                        '    "response": "He eats apples, pears and carrots",\n'
                        '    "documents": [\n'
                        "        {\n"
                        '            "ids": [1,2],\n'
                        '            "verbatims": ["He eats apples and plays football."'
                        ', "He eats pears.", "He eats carrots. I like football."]\n'
                        "        }\n"
                        "    ]\n"
                        "}\n",
                        3: "The output would be:"
                        "{\n"
                        '    "response": "He eats apples, pears and carrots",\n'
                        '    "documents": [\n'
                        "        {\n"
                        '            "ids": [1,2,3],\n'
                        '            "verbatims": ["He eats apples and plays football."'
                        ',"He eats pears.","He eats carrots. I like football."]\n'
                        "        }\n"
                        "    ]\n"
                        "}",
                    },
                    [
                        [],
                        # ["eats apples", "eats pears"],
                        ["eats carrots"],
                    ],
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_rerank": (
                    {
                        0: '{"response": "This document does not answer the question", '
                        '"documents": []}\n'
                        "Score: 0\n",
                        1: '{"response": "apples and pears", "documents": '
                        '[{"ids": [99], "verbatims": ["He eats apples", '
                        '"He eats pears"]}]}\n'
                        "Score: 100\n",
                        2: '{"response": "carrots", "documents": '
                        '[{"ids": [99], "verbatims": ["He eats carrots"]}]}\n'
                        "Score: 100\n",
                        3: '{"response": "This document does not answer the question", '
                        '"documents": []}\n'
                        "Score: 0\n",
                        4: '{"response": "apples and pears", "documents": '
                        '[{"ids": [99], "verbatims": ["He eats apples", '
                        '"He eats pears"]}]}\n',
                    },
                    [
                        ["He eats\napples", "He eats pears."],
                    ],
                    r"(?i).*\bapples\b.*\bpears",
                    {1},
                ),
            },
        ),
    ],
)
@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "refine", "map_rerank"])
def test_qa_with_reference_and_verbatims_chain(
    question: str,
    docs: List[Document],
    map_responses: Dict[str, Tuple[Dict[int, str], List[List[str]], str, Set[int]]],
    chain_type: str,
) -> None:
    # chain_type = "map_reduce"  # stuff, map_reduce, refine, map_rerank

    queries, verbatims, expected_answer, references = map_responses[chain_type]
    llm = init_llm(queries)

    for i in range(0, 1):  # Retry if empty ?
        qa_chain = QAWithReferencesAndVerbatimsChain.from_chain_type(
            llm=llm,
            chain_type=chain_type,
        )
        answer = qa_chain(
            inputs={
                "docs": docs,
                "question": question,
            },
            callbacks=CALLBACKS,
        )
        answer_of_question = answer["answer"]
        if not answer_of_question:
            logger.warning("Return nothing. Retry")
            continue
        assert re.match(expected_answer, answer_of_question)
        for ref, original, assert_verbatims in zip(
            references, answer["source_documents"], verbatims
        ):
            assert docs[ref] is original, "Return incorrect original document"
            assert (compare_responses(original.metadata.get("verbatims", []), assert_verbatims)
                    ), "Return incorrect verbatims"
        break
    else:
        print(f"Response is Empty after {i + 1} tries.")
        assert False, "Impossible to receive a response"
    print(f"response après {i}")


@pytest.mark.skip(reason="Disabled, because the test invokes open")
@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "refine", "map_rerank"])
def test_qa_with_reference_and_verbatims_chain_and_retriever(chain_type: str) -> None:
    from ._test_retriever import CALLBACKS, FAKE_LLM, init_llm, test_retriever

    assert not FAKE_LLM
    type = "wikipedia"

    llm = init_llm({})

    retriever, question = test_retriever(type, llm)
    # retriever.get_relevant_documents(question)
    from langchain_qa_with_references.chains import RetrievalQAWithReferencesAndVerbatimsChain

    qa_chain = RetrievalQAWithReferencesAndVerbatimsChain.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        reduce_k_below_max_tokens=True,
    )
    answer = qa_chain(
        inputs={
            "question": question,
        },
        callbacks=CALLBACKS,
    )
    print(
        f'For the question "{question}", to answer "{answer["answer"]}", the LLM use:'
    )
    for doc in answer["source_documents"]:
        print(f"Source {doc.metadata.get('source', [])}")
        for verbatim in doc.metadata.get("verbatims", []):
            print(f'-  "{verbatim}"')


# def test_ppr():
#     from langchain import OpenAI
#     llm = OpenAI(
#         temperature=0,
#     )
#     from langchain_qa_with_references.chains import QAWithReferencesAndVerbatimsChain
#     chain_type = "map_reduce"  # Only map_reduce can extract the verbatim.
#     qa_chain = QAWithReferencesAndVerbatimsChain.from_chain_type(
#         llm=llm,
#         chain_type=chain_type,
#     )
#     from langchain.document_loaders import TextLoader
#     from langchain.document_loaders import DirectoryLoader
#     loader = DirectoryLoader('/home/pprados/workspace.bda/langchain-qa_with_references',
#                              glob="**/*.md", loader_cls=TextLoader)
#     docs = loader.load()
#
#     # Text Splitters
#     from langchain.text_splitter import MarkdownTextSplitter
#     markdown_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=100,
#                                              length_function=len)
#     md_docs = markdown_splitter.split_documents(docs)
#
#     # Embeddings
#     from langchain.embeddings import \
#         HuggingFaceEmbeddings  # create custom embeddings class that just calls API
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#     # Vector stores (pip install faiss or pip install faiss-cpu)
#     from langchain.vectorstores import FAISS
#     db = FAISS.from_documents(md_docs, embeddings)
#
#     # Retrievers
#     retriever = db.as_retriever(search_kwargs={"k": 4})
#     question = "what is the conventional approach to manage hallucinations of LLM?"
#
#     from langchain_qa_with_references.chains import RetrievalQAWithReferencesAndVerbatimsChain
#     from typing import Literal
#     chain_type: Literal["stuff", "map_reduce", "map_rerank", "refine"] = "map_reduce"
#
#     qa_chain = RetrievalQAWithReferencesAndVerbatimsChain.from_chain_type(
#         llm=llm,
#         chain_type=chain_type,
#         retriever=retriever,
#         reduce_k_below_max_tokens=True,
#     )
#     answer = qa_chain(
#         inputs={
#             "question": question,
#         }
#     )
#     print(
#         f'For the question "{question}", to answer "{answer["answer"]}", the LLM use:')
#     for doc in answer["source_documents"]:
#         print(f"Source {doc.metadata.get('source',[])}")
#         for verbatim in doc.metadata.get("verbatims", []):
#             print(f'-  "{verbatim}"')
