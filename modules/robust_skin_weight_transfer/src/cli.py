import argparse
from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    argument_parser: ArgumentParser = argparse.ArgumentParser(description="Process some meshes.")
    argument_parser.add_argument(
        "--source_mesh", type=str, default="../meshes/sphere.obj", help="Path to the source mesh file"
    )
    argument_parser.add_argument(
        "--target_mesh", type=str, default="../meshes/grid.obj", help="Path to the target mesh file"
    )

    arguments: Namespace = argument_parser.parse_args()
    return arguments
