import tyro

from typing import Union
from typing_extensions import Annotated

from iarap.train import SDFTrainerConfig, DeformTrainerConfig



Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[SDFTrainerConfig, tyro.conf.subcommand(name="train-sdf")],
        Annotated[DeformTrainerConfig, tyro.conf.subcommand(name='deform-sdf')]
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).setup().run()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
