from functools import partial
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import typer
import json5

from pqdm.threads import pqdm
import pandas as pd

from app.exact_matching.match_media import MediaMatcher
from app.gpt_processing.cleanup import GPT35EntitiesCleaner
from app.gpt_processing.gpt35_api import GPT35Paraphraser
from app.gpt_processing.caching import CachingParaphraser
from app.utils import load_entites_csv
from app.lemma import StanzaLemma


typer_app = typer.Typer()


@typer_app.command()
def exact_match_media(
    entities_path: Path = typer.Option("data/entities_list.csv", "--entities", "-e"),
    input_path: Path = typer.Option(..., "--input", "-i"),
    output_path: Path = typer.Option(..., "--output", "-o"),
):
    entites = load_entites_csv(entities_path=entities_path)
    lemma = StanzaLemma()

    matcher = MediaMatcher(entities=entites, lemma=lemma)
    matcher.process(input_path, output_path)


@typer_app.command()
def cleanup_entities(
    input_path: Path = typer.Option(..., "--input", "-i"),
    output_path: Path = typer.Option(..., "--output", "-o"),
):
    df = pd.read_json(input_path, lines=True)
    rows = [row for id, row in df.iterrows()]

    cleaner = GPT35EntitiesCleaner()
    output_rows = cleaner.cleanup_responses(rows)

    output_df = pd.DataFrame.from_records(output_rows)
    output_df.to_json(output_path, lines=True, force_ascii=False, orient="records")


@typer_app.command()
def gpt35_extract_entities(
    input_path: Path = typer.Option(..., "--input", "-i"),
    output_path: Path = typer.Option(..., "--output", "-o"),
    api_key: str = typer.Option(..., "--api"),
    head_n: int = typer.Option(None, "--head", "-h"),
):
    paraphraser = GPT35Paraphraser(api_key, system_prompt=GPT35_EXTRACT_ENTITIES_SYSTEM_PROMPT)
    paraphraser = CachingParaphraser(paraphraser, cache_path=Path("./cache/gpt35_cache.pickle"))

    df = pd.read_json(input_path, lines=True)
    if head_n:
        df = df.head(head_n)
    all_rows = [row for index, row in df.iterrows()]
    
    try:
        output_rows = pqdm(all_rows, partial(gpt_process_single_text, paraphraser=paraphraser), n_jobs=16, exception_behaviour='immediate')
        output_path.parent.mkdir(exist_ok=True)
        output_df = pd.DataFrame.from_records(output_rows)
        output_df.to_json(output_path, lines=True, force_ascii=False, orient="records")
    finally:
        paraphraser.save_cache_to_file()


def gpt_process_single_text(row: Dict, paraphraser: CachingParaphraser):
    input_text = f"{row['lead']} {row['text']}"
    output = paraphraser.process(input_text)
    row["model_input"] = input_text
    row["response"] = parse_response_to_json(output)
    return row


def parse_response_to_json(response: str):
    try:
        json5.loads(response)
        return str(response)
    except ValueError:
        print(f"model response cannot be parsed into json. reponse: {response}")
        return None

GPT35_EXTRACT_ENTITIES_SYSTEM_PROMPT = """You are a tool to help find Media and Public Institutions that the journalist writing the text used as a source. 
List only the sources that are explicitly mentioned like "as mentioned by", "according to".

Return results in JSON, alongside the context, which is a part of the text in which the name of the entity is explicitly mentioned:
{
"media outlets": [{"text": "name_of_the_entity", "context": "context_before_entity name_of_the_entity context_after_the_entity"}],
"institutions": [{"text": "name_of_the_entity", "context": "context_before_entity name_of_the_entity context_after_the_entity"}]
}
"""


if __name__ == "__main__":
    typer_app()
