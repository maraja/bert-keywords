from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from functools import cached_property
from sklearn.feature_extraction.text import CountVectorizer
import torch
import tensorflow_hub
import re
import numpy as np


class Keywords():
    def __init__(self, bert_model: AutoModel, tokenizer: AutoTokenizer):
        embedding_model = tensorflow_hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4")
        self.kw_model = KeyBERT(model=embedding_model)
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        count_vectorizer = CountVectorizer()
        self.count_tokenizer = count_vectorizer.build_tokenizer()

    def get_word_embedding(self, sentence: str, word: str, index: int = None) -> torch.tensor:
        # remove subsequent spaces
        clean_sentence = re.sub(' +', ' ', sentence)
        tokenized_sentence = self.count_tokenizer(clean_sentence)
        if index is None:
            for i, _word in enumerate(tokenized_sentence):
                if _word.lower() == word.lower():
                    index = i
                    break

        assert index is not None, "Error: word not found in provided sentence."
        tokens = self.tokenizer(clean_sentence, return_tensors='pt')

        token_ids = [np.where(np.array(tokens.word_ids()) == idx)
                     for idx in [index]]

        with torch.no_grad():
            output = self.bert_model(**tokens)

        # Only grab the last hidden state
        hidden_states = output.hidden_states[-1].squeeze()

        # Select the tokens that we're after corresponding to the word provided
        embedding = hidden_states[[tup[0][0] for tup in token_ids]]
        return embedding

    """uses keybert to generate keywords from a sentence
    ex: [('continual', 0.6023), ('change', 0.4642), ('life', 0.4436), ('essence', 0.3975)]
    """
    def get_keywords_with_embeddings(self, data: str) -> List[Tuple[str, float, torch.Tensor]]:
        keywords = self.kw_model.extract_keywords(data, keyphrase_ngram_range=(1, 1), stop_words=None)
        kw_tuples = []
        for kw in keywords:
            embedding = self.get_word_embedding(data, kw[0])
            kw_tuples.append((kw[0], kw[1], embedding))
        return kw_tuples
        

    def get_keywords(self, data: str, emb:bool=True) -> List[Tuple[str, float, torch.Tensor]]:
        return self.kw_model.extract_keywords(data, keyphrase_ngram_range=(1, 1), stop_words=None)


    
    def get_keyphrases(self, data: str, min_ngram=2, max_ngram=3) -> List[Tuple[str, float]]:
        return self.kw_model.extract_keywords(data, keyphrase_ngram_range=(min_ngram, max_ngram), stop_words=None)
