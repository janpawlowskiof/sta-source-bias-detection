import csv
import re
import json
from pathlib import Path
from typing import Any, List


def load_json(path: Path):
    with path.open("r") as file:
        return json.load(file)


def dump_json(obj: Any, path: Path):
    with path.open("w") as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)


def load_entites_csv(entities_path: Path) -> List[str]:
    with entities_path.open("r", encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        entites = list(reader)
        return [entity["name"] for entity in entites]


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


def remove_multiple_spaces(text: str) -> str:
    return re.sub(' +', ' ', text).lstrip(" ").strip(" ")
