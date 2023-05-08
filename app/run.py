import typer


typer_app = typer.Typer()


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
