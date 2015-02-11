#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
import argparse
from examples.reverse_words import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Case study of learning to reverse words from a natural text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "test"],
        help="The mode to run")
    parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process.")
    parser.add_argument(
        "--num-batches", default=20000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--from-dump", default=None,
        help="Path to the dump to be loaded")
    args = parser.parse_args()
    main(**vars(args))
