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
    argument_parser.add_argument(
        "--output_file", type=str, default="../meshes/output.json", help="Path to the output file"
    )

    arguments: Namespace = argument_parser.parse_args()
    return arguments
