import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="HAMLET")

    parser.add_argument(
        "-conf_file",
        "--conf_file",
        nargs="?",
        type=str,
        default="green_126025.json",
        help="configuration file name",
    )

    return parser.parse_known_args()
