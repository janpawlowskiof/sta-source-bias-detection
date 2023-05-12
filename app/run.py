from pathlib import Path
import typer

from app.exact_matching.match_media import MediaMatcher
from app.utils import load_entites_csv
from app.lemma import StanzaLemma
from app.attribution_clf.inference import AttributionModel
from typing import List, Dict
from app.ner_tools.ner_tools import NER, reformat_df, postprocess_df
import pandas as pd

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


def ner_processing(
        base: pd.DataFrame,
):
    ner = NER(model_name_or_path="../models/sloner")
    base['entities'] = base.apply(lambda x: ner.find_entities(x['text']), axis=1)
    formatted_df = reformat_df(base)
    result = postprocess_df(formatted_df)

    # return df as a list of dicts
    return result.to_dict(orient='records')

@typer_app.command()
def process_time_range(star_date: str, end_date: str):
    # TODO add api calls to retrieve articles
    # TODO pass to ner and return a path to the entities file
    retrieved_articles : pd.Dataframe = ...
    ner_results = ner_processing(retrieved_articles)
    attribution_model = AttributionModel()
    entities_in_texts = attribution_model.process_dataset(ner_results)
    return entities_in_texts


if __name__ == "__main__":
    typer_app()
