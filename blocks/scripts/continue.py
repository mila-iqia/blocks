#!/usr/bin/env python
from argparse import ArgumentParser
import sys

import dill


if __name__ == "__main__":
    parser = ArgumentParser("Continues your pickled main loop")
    parser.add_argument(
        "path", help="A path to a file with a pickled main loop")
    parser.add_argument(
        "--rec-limit", type=int, help="The recursion depth limit")
    args = parser.parse_args()

    if args.rec_limit:
        sys.setrecursionlimit(args.rec_limit)
    main_loop = dill.load(open(args.path, "rb"))
    main_loop.run()
