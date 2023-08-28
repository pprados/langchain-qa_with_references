# flake8: noqa
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.output_parsers import PydanticOutputParser

# from langchain.pydantic_v1 import BaseModel
from pydantic import BaseModel
from typing import List, Optional
import re


def _extract_original_verbatim(verbatim: str, page_content: str) -> Optional[str]:
    """The exact format of verbatim may be changed by the LLM.
    Extract only the words of the verbatim, and try to find a sequence
    of same words in the original document.
    """
    only_words = filter(len, re.split(r"[^\w]+", verbatim))
    regex_for_words_in_same_oder = (
        r"(?i)\b" + r"\b[^\w]+".join(only_words) + r"\b"
        r"\s*[.!?:;]?"  # Optional end of sentence
    )
    match = re.search(regex_for_words_in_same_oder, page_content, re.IGNORECASE)
    if match:
        return match[0].strip()
    return None  # No verbatim found in the original document


class Verbatim(BaseModel):
    # class Config:
    #     allow_mutation = True

    verbatims: List[str]

    def update_verbatims(self, page_content: str) -> None:
        result = []
        for j, verbatim in enumerate(self.verbatims):
            original_verbatim = _extract_original_verbatim(
                verbatim=verbatim, page_content=page_content
            )
            if original_verbatim:
                result.append(original_verbatim)
        self.verbatims = result

    def __str__(self) -> str:
        return "\n".join(
            [f'"{v}"' for v in [re.sub(r"\s", " ", v) for v in self.verbatims]]
        )


def default_verbatim_parser() -> BaseOutputParser:
    return PydanticOutputParser(pydantic_object=Verbatim)


question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
{context}

Question: {question}

Return any relevant text verbatims.
{format_instructions}
"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template,
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": default_verbatim_parser().get_format_instructions()
    },
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, 
create a final answer with all references ("IDXS"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
ALWAYS return a "IDXS" part in your answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Idxs: 0, 1
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.
Idxs: 2
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws,
Idxs: 3
Content: The english law is applicable for this agreement.
Idxs: 4
=========
FINAL ANSWER: This Agreement is governed by English law.
IDXS: 0, 1, 4

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet.
Idxs: 0
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life.
Idxs: 1, 2
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.
Idxs: 3
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health.
idxs: 4
=========
FINAL ANSWER: The president did not mention Michael Jackson.
IDXS:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template,
    input_variables=["summaries", "question"],
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nIdx: {_idx}",
    input_variables=["page_content", "_idx"],
)
