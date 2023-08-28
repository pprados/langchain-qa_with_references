from typing import Set, List, Dict

import pytest

from langchain.schema import BaseRetriever, Document
#from langchain_qa_with_references.chains import QAWithReferencesChain
from langchain_qa_with_references.chains import QAWithReferencesChain
from langchain.callbacks.stdout import StdOutCallbackHandler
from tests.unit_tests.fake_llm import FakeLLM

VERBOSE=True
if VERBOSE:  # FIXME: paramètre à traiter dans la chain
    from typing import *
    class ExStdOutCallbackHandler(StdOutCallbackHandler):
        def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "",
            **kwargs: Any,
        ) -> None:
            print("====")
            return super().on_text(text=text, color=color, end=end)

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            """Ajoute une trace des outputs du llm"""
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
else:
    CALLBACKS = []

@pytest.mark.parametrize(
    "question,docs,map_responses,expected_answer,references,verbatims",
    [
        # (
        #     "Which state/country's law governs the interpretation of the contract?",
        #     [
        #         Document(
        #             page_content="This Agreement is governed by English law.",
        #             metadata={"source": "1.html"},
        #         ),
        #     ],
        #     {
        #         0: 'Here is the output:{"verbatims": ["This Agreement is governed by English law."]}',
        #         1: "This Agreement is governed by English law.",
        #     },
        #     "This Agreement is governed by English law.",
        #     {0},
        #     [["This Agreement is governed by English law."]],
        # ),
        (
            "what does it eat?",
            [
                Document(
                    page_content="he eats\napples and plays football. My name is Philippe. he eats pears.",
                    metadata={},
                ),
                Document(
                    page_content="he eats carrots. I like football.",
                    metadata={},
                ),
                Document(
                    page_content="The Earth is round.",
                    metadata={},
                ),
            ],
            {
                0: 'Here is the output:\n```\n{"verbatims": ["he eats apples.", "he eats pears."]}\n```',
                1: 'Here is the output:\n```\n{"verbatims": ["he eats carrots."]}\n```',
                2: 'Here is the output:\n```\n{"verbatims": ["The Earth is round."]}\n```',
                4: "He eats apples, pears, and carrots.",
            },
            "He eats apples, pears, and carrots.",
            {0, 1},
            [["he eats\napples", "he eats pears."], ["he eats carrots."]],
        ),
    ],
)
def test_qa_with_reference_chain(
    question: str,
    docs: List[Document],
    map_responses: Dict[int, str],
    expected_answer: str,
    references: Set[int],
    verbatims: List[List[str]],
) -> None:
    llm_response = f"{expected_answer}\n" f"IDXS:{','.join(map(str,references))}"
    chain_type="map_reduce"  # stuff, map_reduce, refine, map_rerank

    if True:  # Use simulation ?
        llm = FakeLLM(
            queries=map_responses,
            sequential_responses=True,
        )
    else:  # Not a simulation
        from langchain import OpenAI
        from dotenv import load_dotenv
        import langchain

        load_dotenv()
        from langchain.cache import SQLiteCache

        langchain.llm_cache = SQLiteCache(database_path="/tmp/cache.db")
        llm = OpenAI(
            temperature=0,
        )
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
    assert answer["answer"].strip() == expected_answer
    for ref, original, assert_verbatims in zip(
        references, answer["source_documents"], verbatims
    ):
        assert docs[ref] is original
        if chain_type in ["map_reduce",]:
            assert original.metadata["verbatims"] == assert_verbatims
