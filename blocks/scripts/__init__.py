import sys

import dill

from blocks.dump import MainLoopDumpManager


def continue_training(path, rec_limit=None):
    if rec_limit:
        sys.setrecursionlimit(rec_limit)
    main_loop = dill.load(open(path, "rb"))
    main_loop.run()


def dump(pickle_path, dump_path, rec_limit=None):
    if rec_limit:
        sys.setrecursionlimit(rec_limit)
    main_loop = dill.load(open(pickle_path, "rb"))
    MainLoopDumpManager(dump_path).dump(main_loop)
