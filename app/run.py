from pathlib import Path
import typer

from app.exact_matching.match_media import MediaMatcher
from app.utils import load_entites_csv
from app.lemma import StanzaLemma
from app.attribution_clf.inference import AttributionModel
from typing import List, Dict

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
def process_time_range(star_date: str, end_date: str):
    # TODO add api calls to retrieve articles
    # TODO pass to ner and return a path to the entities file
    ner_results: List[Dict] = ...

    attribution_model = AttributionModel()
    entities_in_texts = attribution_model.process_dataset(ner_results)
    return entities_in_texts


if __name__ == "__main__":
    typer_app()
