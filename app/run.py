from pathlib import Path
import typer

from app.exact_matching.match_media import MediaMatcher
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
def say_hello(
    name: str = typer.Option("olek", "--name", "-n")
):
    print(f"hello {name}")


@typer_app.command()
def say_goodbye(
    name: str = typer.Option("olek", "--name", "-n")
):
    print(f"goodbye {name}")


if __name__ == "__main__":
    typer_app()
