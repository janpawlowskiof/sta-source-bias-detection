from __future__ import annotations

import multiprocessing
from pathlib import Path
import pickle
from typing import Dict, Optional

from app.gpt_processing.gpt35_api import GPT35Paraphraser


class CachingParaphraser:
    def __init__(
        self, 
        base_paraphraser: GPT35Paraphraser,
        cache_path: Path,
        save_frequency: int = 20,
    ) -> None:
        self.base_paraphraser: GPT35Paraphraser = base_paraphraser

        self.cache_dict: Dict = multiprocessing.Manager().dict()
        self.save_frequency: int = save_frequency
        self.base_calls_counter: int = 0

        self.cache_path = cache_path
        self.load_cache_from_file()

    def process(self, text: str, *args, **kwargs) -> str:
        if text in self.cache_dict:
            return self.cache_dict[text]

        translation = self.base_paraphraser.process(text=text, *args, **kwargs)
        self.cache_dict[text] = translation
        self.base_calls_counter += 1
        if self.base_calls_counter % self.save_frequency == 0:
            self.save_cache_to_file()
        return translation

    def load_cache_from_file(self):
        if not self.cache_path:
            return
        if not self.cache_path.exists():
            print(f"No cache found at path {self.cache_path}")
            return
        with self.cache_path.open("rb") as cache_file:
            d = pickle.load(cache_file)
            self.cache_dict.update(d)
            print(f"Found cache of {len(self.cache_dict)} items")

    def forget_translation(self, key: str):
        del self.cache_dict[key]

    def save_cache_to_file(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(exist_ok=True, parents=True)
        with self.cache_path.open("wb") as cache_file:
            d = dict(self.cache_dict)
            pickle.dump(d, cache_file)
