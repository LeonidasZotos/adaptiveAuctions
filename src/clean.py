"""This file contains the cleaning function. It is responsible for cleaning all files from previous runs."""
import os


def run(args):
    """This runs the cleaning function, which deletes all files from previous runs.
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    """
    # Delete all files in the results folder, if it exists
    if os.path.exists("results"):
        for file in os.listdir("results"):
            os.remove(os.path.join("results", file))

    # Delete python cache folder
    for file in os.listdir("src/__pycache__"):
        os.remove(os.path.join("src/__pycache__", file))
    os.rmdir("src/__pycache__")

    print("Clean Completed")
