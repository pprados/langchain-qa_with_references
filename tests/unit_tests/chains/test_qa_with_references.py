import re
from typing import Dict, List, Set, Tuple

import pytest
from langchain.schema import Document

from langchain_qa_with_references.chains import QAWithReferencesChain

from ._test_retriever import CALLBACKS, init_llm, logger


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
                        0: "```\n"
                        '{"response": "He eats apples and pears and carrots.",'
                        ' "documents": [1,2]}\n'
                        "```\n",
                    },
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_reduce": (
                    {
                        0: 'Output: {"lines": []}',
                        1: 'Output: {"lines": ["He eats apples.", "He eats pears."]}',
                        2: 'Output: {"lines": ["He eats carrots."]}',
                        3: 'Output: {"lines": []}',
                        4: '{"response": "He eats apples, pears, and carrots.", '
                        '"documents": [1,2]}',
                    },
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "refine": (
                    {
                        0: "The output should be:\n"
                        "```\n"
                        '{"response": "I don\'t know", "documents": []}\n'
                        "```",
                        1: "Therefore, the final output should be:\n"
                        "```\n"
                        '{"response": "He eats apples and pears.", "documents": [1]}\n'
                        "```\n",
                        2: "Therefore, the final output should be:\n"
                        "```\n"
                        '{"response": "He eats apples, pears, and carrots.", '
                        '"documents": [1, 2]}\n'
                        "```\n",
                        3: "Therefore, the final output should be:\n"
                        "```"
                        '{"response": "He eats apples, pears, and carrots.", '
                        '"documents": [1, 2, 3]}\n'
                        "```\n",
                    },
                    r"(?i).*\bapples\b.*\bpears\b.*\bcarrots\b",
                    {1, 2},
                ),
                "map_rerank": (
                    {
                        0: '{"response": "This document does not answer the question", '
                        '"documents": []}\n'
                        "Score: 0\n",
                        1: '{"response": "apples and pears", "documents": []}\n'
                        "Score: 100\n",
                        2: '{"response": "carrots", "documents": []}\n' "Score: 100\n",
                        3: '{"response": "This document does not answer the question", '
                        '"documents": []}\n'
                        "Score: 0\n",
                        4: '{"response": "apples and pears", "documents": []}\n',
                    },
                    r"(?i).*\bapples\b.*\bpears",
                    {1},
                ),
            },
        ),
    ],
)
@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "refine", "map_rerank"])
def test_qa_with_reference_chain(
    question: str,
    docs: List[Document],
    map_responses: Dict[str, Tuple[Dict[int, str], str, Set[int]]],
    chain_type: str,
) -> None:
    # chain_type = "map_reduce"  # stuff, map_reduce, refine, map_rerank

    queries, expected_answer, references = map_responses[chain_type]
    llm = init_llm(queries)

    for i in range(0, 1):  # Retry if empty ?
        qa_chain = QAWithReferencesChain.from_chain_type(
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
        for ref, original in zip(references, answer["source_documents"]):
            assert docs[ref] is original, "Return incorrect original document"
        break
    else:
        print(f"response after {i + 1} tries.")
        assert not "Impossible to receive a response"
    print(f"response après {i}")


@pytest.mark.skip(reason="Disabled, because the test invokes open")
@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "refine", "map_rerank"])
def test_qa_with_reference_chain_and_retriever(chain_type: str) -> None:
    from ._test_retriever import CALLBACKS, FAKE_LLM, init_llm, test_retriever

    assert not FAKE_LLM
    type = "google"

    llm = init_llm({})
    retriever, question = test_retriever(type, llm)

    # retriever.get_relevant_documents("what time is it?")
    from langchain_qa_with_references.chains import RetrievalQAWithReferencesChain

    qa_chain = RetrievalQAWithReferencesChain.from_chain_type(
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
        print(f"Source {doc.metadata['source']}")
