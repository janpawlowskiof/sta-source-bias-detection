from functools import partial
from pathlib import Path
from typing import Dict
import typer

from pqdm.threads import pqdm
import pandas as pd

from app.exact_matching.match_media import MediaMatcher
from app.gpt_processing.cleanup import GPT35EntitiesCleaner
from app.gpt_processing.gpt35_api import GPT35Paraphraser
from app.gpt_processing.caching import CachingParaphraser
from app.utils import dump_json, load_entites_csv
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

    dump_json(output_rows, output_path.with_suffix(".json"))


@typer_app.command()
def gpt35_extract_entities(
        input_path: Path = typer.Option(..., "--input", "-i"),
        output_path: Path = typer.Option(..., "--output", "-o"),
        api_key: str = typer.Option(..., "--api"),
        head_n: int = typer.Option(None, "--head", "-h"),
):
    paraphraser = GPT35Paraphraser(api_key)
    paraphraser = CachingParaphraser(paraphraser, cache_path=Path("./cache/gpt35_cache.pickle"))

    df = pd.read_json(input_path, lines=True)
    if head_n:
        df = df.head(head_n)
    all_rows = [row for index, row in df.iterrows()]

    try:
        output_rows = pqdm(all_rows, partial(gpt_process_single_text, paraphraser=paraphraser), n_jobs=16,
                           exception_behaviour='immediate')
        output_path.parent.mkdir(exist_ok=True)
        output_df = pd.DataFrame.from_records(output_rows)
        output_df.to_json(output_path, lines=True, force_ascii=False, orient="records")
    finally:
        paraphraser.save_cache_to_file()


def gpt_process_single_text(row: Dict, paraphraser: CachingParaphraser):
    input_text = get_prompt(row['entity'], row['context'])
    output = paraphraser.process(input_text)
    row["model_input"] = input_text
    row['response'] = output
    row['chatgpt_label'] = get_label(output)
    return row


def get_label(response: str):
    response = response.lower()
    if 'true' in response and 'false' not in response:
        return True
    elif 'false' in response and 'true' not in response:
        return False
    else:
        return None


def get_prompt(entity: str, fragment: str):
    return f"""You have to identify people, media and institutions as direct information sources for news fragments, e.g. "according to Reuters", "Emmanuel Macron in his recent interview, said", "the information provided by the Ministry of Health, is". You will be given a news fragment in slovenian and an entity. Decide, if the entity is a direct information source for the fragment. Respond only with True or False. Entity: {entity}, fragment: {fragment}"""


if __name__ == "__main__":
    typer_app()
