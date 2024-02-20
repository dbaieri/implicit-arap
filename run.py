import tyro

import iarap.config.defaults




def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(iarap.config.defaults.Commands).setup().run()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(iarap.config.defaults.Commands)  # noqa
