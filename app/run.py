from pathlib import Path
import typer

from app.exact_matching.match_media import MediaMatcher
from app.utils import load_entites_csv
from app.lemma import StanzaLemma
from app.attribution_clf.inference import AttributionModel
from typing import List, Dict
from app.ner_tools.ner_tools import NER, reformat_df, postprocess_df
from app import ROOT_PATH
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
    ner = NER(model_name_or_path=str(ROOT_PATH / "models" / "sloner"))
    base['entities'] = base.apply(lambda x: ner.find_entities(x['text']), axis=1)
    formatted_df = reformat_df(base)
    result = postprocess_df(formatted_df)

    # return df as a list of dicts
    return result.to_dict(orient='records')

@typer_app.command()
def process_time_range(data: pd.DataFrame):
    # TODO add api calls to retrieve articles
    # TODO pass to ner and return a path to the entities file
    ner_results = ner_processing(data)
    attribution_model = AttributionModel()
    entities_in_texts = attribution_model.process_dataset(ner_results)
    return entities_in_texts


if __name__ == "__main__":
    typer_app()
