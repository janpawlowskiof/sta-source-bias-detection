from abc import abstractmethod
from typing import Dict, List

from app.utils import remove_punctuation, remove_multiple_spaces

class Lemma:
    @abstractmethod
    def lemma_into_words(self, text: str) -> List[Dict[str, str]]:
        pass

    def lemma_into_text(self, text: str, lower: bool = True, no_punctuation: bool = True) -> str:
        words = self.lemma_into_words(text)
        lemma_words = [word["lemma"] for word in words]
        output = " ".join(lemma_words)
        if lower:
            output = output.lower()
        if no_punctuation:
            output = remove_punctuation(output)
        output = remove_multiple_spaces(output)
        return output
