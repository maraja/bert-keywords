from keybert import KeyBERT
from typing import List, Tuple
from functools import cached_property
import tensorflow_hub


class Keywords():
    def __init__(self):
        embedding_model = tensorflow_hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4")
        self.kw_model = KeyBERT(model=embedding_model)

    """uses keybert to generate keywords from a sentence
    ex: [('continual', 0.6023), ('change', 0.4642), ('life', 0.4436), ('essence', 0.3975)]
    """

    def get_keywords(self, data: str) -> List[Tuple[str, float]]:
        return self.kw_model.extract_keywords(data, keyphrase_ngram_range=(1, 1), stop_words=None)

    
    def get_keyphrases(self, data: str, min_ngram=2, max_ngram=3) -> List[Tuple[str, float]]:
        return self.kw_model.extract_keywords(data, keyphrase_ngram_range=(min_ngram, max_ngram), stop_words=None)
