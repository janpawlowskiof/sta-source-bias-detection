from typing import Dict, List
import stanza
from stanza.models.common.doc import Word

from app.lemma.lemma import Lemma

class StanzaLemma(Lemma):
    def __init__(self) -> None:
        stanza.download('sl')
        self.nlp: stanza.Pipeline = stanza.Pipeline('sl', download_method=None, processors="tokenize,pos,lemma")

    def lemma_into_words(self, text: str) -> List[Dict[str, str]]:
        doc = self.nlp(text)
        words: List[Word] = list(doc.iter_words())
        words_dicts = [
            word.to_dict()
            for word in words
        ]
        return words_dicts
 