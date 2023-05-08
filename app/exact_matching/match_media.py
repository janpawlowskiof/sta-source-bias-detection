
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from app.lemma import Lemma
from app.utils import load_json, dump_json


class MediaMatcher:
    def __init__(self, entities: List[str], lemma: Lemma) -> None:
        self.lemma: Lemma = lemma
        self.entities: List[Dict[str, str]] = [
            {
                "text": entity,
                "lemma": self.lemma.lemma_into_text(entity)
            }
            for entity in entities
        ]

    def process(self, input_json_path: Path, output_json_path: Path):
        articles = load_json(input_json_path)
        texts = list(articles["text"].values())[:100]
        processed_articles = [
            self.process_single_article(text) 
            for text in tqdm(texts) 
            if text
        ]
        output_json_path.parent.mkdir(exist_ok=True, parents=True)
        dump_json(processed_articles, output_json_path)


    def process_single_article(self, article: str) -> Dict:
        lemma_article = self.lemma.lemma_into_text(article)
        found_entites = [
            entity
            for entity in self.entities
            if entity["lemma"] in lemma_article
        ]
        return {
            "article": article,
            "lemma_article": lemma_article,
            "found_entites": found_entites
        }
