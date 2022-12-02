from keybert import KeyBERT
from typing import List, Tuple
from functools import cached_property


class Keywords():
    def __init__(self):
        self.kw_model = KeyBERT()

    """uses keybert to generate keywords from a sentence
    ex: [('continual', 0.6023), ('change', 0.4642), ('life', 0.4436), ('essence', 0.3975)]
    """

    def get_keywords(self, data: str) -> List[Tuple[str, float]]:
        return self.kw_model.extract_keywords(data)
