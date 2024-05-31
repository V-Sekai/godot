import argparse
import os

def parse_arguments():
    argument_parser = argparse.ArgumentParser(description='Process some meshes.')
    argument_parser.add_argument('--source_mesh', type=str, default='../meshes/sphere.obj',
                        help='Path to the source mesh file')
    argument_parser.add_argument('--target_mesh', type=str, default='../meshes/grid.obj',
                        help='Path to the target mesh file')

    arguments = argument_parser.parse_args()
    return arguments
