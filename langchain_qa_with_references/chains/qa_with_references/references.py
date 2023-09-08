import logging
import re
from typing import Set

from langchain.output_parsers import PydanticOutputParser, RegexParser
# Impossible to import in experimental. bug in the CI
from langchain.pydantic_v1 import BaseModel
# from pydantic import BaseModel
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)

# To optimize the consumption of tokens, it's better to use only 'text', without json.
# Else the schema consume ~300 tokens and the response 20 tokens by step
_OPTIMIZE = False


class References(BaseModel):
    """
    Response and referenced documents.
    """

    response: str
    """ The response """
    documents: Set[str] = set()
    """ The list of documents used to response """

    def __str__(self) -> str:
        if _OPTIMIZE:
            return f'{self.response}\nIDX:{",".join(map(str, self.documents))}'
        else:
            return self.json()


references_parser: BaseOutputParser
if _OPTIMIZE:

    class _ReferencesParser(RegexParser):
        """An optimised parser for Reference.
        It's more effective than the pydantic approach
        """

        def get_format_instructions(self) -> str:
            return ""

        def parse(self, text: str) -> References:
            dict = super().parse(text)
            if dict["ids"]:
                ids = set()
                for str_doc_id in dict["ids"].split(","):
                    m = re.match("_idx_(\d+)", str_doc_id)
                    if m:
                        set.add(int(m[1]))
            else:
                ids = []
            return References(response=dict["response"], documents=ids)


    references_parser = _ReferencesParser(
        regex=r"(.*)\nIDX:(.*)",
        output_keys=["response", "ids"],
        default_output_key="response",
    )
else:
    references_parser = PydanticOutputParser(pydantic_object=References)
