import sys

import dill


def continue_training(path, rec_limit=None):
    if rec_limit:
        sys.setrecursionlimit(rec_limit)
    main_loop = dill.load(open(path, "rb"))
    main_loop.run()
