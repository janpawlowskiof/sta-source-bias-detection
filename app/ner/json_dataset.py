import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class JsonDataset(Dataset):
    def __init__(self, path: Path, tokenizer: PreTrainedTokenizer, all_labels: List[str], split: str, max_margin_size: int, max_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.all_labels = all_labels
        self.max_margin_size = max_margin_size
        self.max_length = max_length

        with path.open("r") as file:
            self.entries = [
                entry for entry in json.load(file) 
                if entry["split"] == split
            ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.tokenize_and_align_labels(self.entries[index])
        return {
            "input_ids": entry["input_ids"].squeeze(0),
            "attention_mask": entry["attention_mask"].squeeze(0),
            "labels": torch.tensor(entry["labels"]),
            "text": entry["text"],
            "entities": entry["entities"],
        }

    def tokenize_and_align_labels(self, entry: Dict[str, Any]):
        text = entry["text"]

        num_skipped_characters = 0
        if self.max_margin_size:
            assert len(entry["entities"]) > 0
            # sampling text around one of the entities
            main_entity = random.choice(entry["entities"])
            left_margin_size = random.randint(0, self.max_margin_size)
            right_margin_size = self.max_margin_size - left_margin_size
            num_skipped_characters = max(main_entity["start"] - left_margin_size, 0)
            text = text[num_skipped_characters: main_entity["end"] + right_margin_size]

        # assigning classes to words
        words = text.split(" ")
        words_labels = ["O" for _ in words]
        char_to_word = []
        for word_index, word in enumerate(words):
            for _ in range(len(word) + 1):
                char_to_word.append(word_index)

        char_to_word = dict(enumerate(char_to_word))
        # +1 bacause of trailing space, but no harm done otherwise
        assert len(char_to_word) == len(text) + 1

        for entity in entry["entities"]:
            start_char_index = entity["start"] - num_skipped_characters
            if start_char_index not in char_to_word:
                continue
            first_char_in_entity_index = char_to_word[start_char_index]
            for char_index in range(entity["start"] - num_skipped_characters, entity["end"] - num_skipped_characters):
                if char_index not in char_to_word:
                    continue
                word_index = char_to_word[char_index]
                if word_index == first_char_in_entity_index:
                    words_labels[word_index] = f"B-entity"
                else:
                    words_labels[word_index] = f"I-entity"

        words_labels_ids = [self.all_labels.index(word_label) for word_label in words_labels]

        # tokenizing words into subtokens
        tokenized_inputs = self.tokenizer(
            words,
            truncation=True, 
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # only first subtoken of each words gets a class assigned
        word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(words_labels_ids[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        if not any(label > 0 for label in label_ids):
            print(f"NO POSITIVE LABELS IN TEXT {text}")
        tokenized_inputs["labels"] = label_ids
        tokenized_inputs["text"] = text
        tokenized_inputs["entities"] = ",".join([entity["entity_value"] for entity in entry["entities"]])
        return tokenized_inputs
