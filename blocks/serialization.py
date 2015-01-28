import logging
import os
import os.path
from collections import OrderedDict

import dill
import numpy

from blocks.bricks.base import Brick
from blocks.select import Selector

logger = logging.getLogger(__name__)


def save_parameter_values(param_values, path):
    """Compactly save parameter values.

    This is a thin wrapper over `numpy.savez`. It deals with
    `numpu`'s vulnerability to slashes in file names.

    Parameters
    ----------
    param_values : dict of (parameter name, numpy array)
        The parameter values.
    path : str of file
        The destination for saving.

    """
    param_values = {name.replace("/", "-"): param
                    for name, param in param_values.items()}
    numpy.savez(path, **param_values)


def load_parameter_values(path):
    """Load parameter values saved by :fun:`save_parameters`

    This is a thin wrapper over `numpy.load`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    path : str or file
        The source for loading from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    source = numpy.load(path)
    param_values = {name.replace("-", "/"): value
                    for name, value in source.items()}
    source.close()
    return param_values


def extract_parameter_values(bricks):
    """Extract parameter values from a bricks hierarchy.

    Parameters
    ----------
    bricks : Brick or Selector
        The top bricks.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    if isinstance(bricks, Brick):
        bricks = Selector([bricks])
    if not isinstance(bricks, Selector):
        raise ValueError
    return OrderedDict([(name, variable.get_value(borrow=True))
                        for name, variable in bricks.get_params().items()])


def inject_parameter_values(bricks, param_values):
    """Inject parameter values into a bricks hierarchy.

    Parameters
    ----------
    bricks : Brick or Selector
        The top bricks.
    param_values : dict of (parameter name, numpy array) pairs
        The parameter values.

    """
    if isinstance(bricks, Brick):
        bricks = Selector([bricks])
    if not isinstance(bricks, Selector):
        raise ValueError

    for name, value in param_values.items():
        selected = bricks.select(name)
        if len(selected) == 0:
            logger.error("Unknown parameter {}".format(name))
        if not len(selected) == 1:
            raise ValueError
        selected = selected[0]

        assert selected.get_value(
            borrow=True, return_internal_type=True).shape == value.shape
        selected.set_value(value)

    params = bricks.get_params()
    for name in params.keys():
        if name not in params:
            logger.error("No value is provided for the parameter {}"
                         .format(name))


class MainLoopStateManager(object):
    """Saves and loads the state of a main loop.

    The default way to save the training progress in Blocks
    is full serialization of the main loop object. This way is
    however often obstructed by various technical difficulties.

    This class provides saving and loading logic that circumvents
    serialization of the most problematic parts: the model (which is
    typically a brick hierarchy) and the training algorihm (which
    typically has Theano functions as attributes). The on-disk
    representation used is a folder with a few files containing
    model parameters, log and state of the data iteration.

    Parameter
    ---------
    folder : str
        The path to the root folder for the saved state.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, folder):
        self.folder = folder

    def path_to_parameters(self):
        return "{}/{}".format(self.folder, "params.npz")

    def path_to_iteration_state(self):
        return "{}/{}".format(self.folder, "iteration_state.pkl")

    def path_to_log(self):
        # The extension is omitted for the log because advanced
        # log classes might have a better format for storing on the
        # then pickled file. Currenly log is just pickled though.
        return "{}/{}".format(self.folder, "log")

    def save_parameters(self, main_loop):
        save_parameter_values(extract_parameter_values(main_loop.model),
                              self.path_to_parameters())

    def save_iteration_state(self, main_loop):
        iteration_state = (main_loop.data_stream, main_loop.epoch_iterator)
        with open(self.path_to_iteration_state(), "wb") as destination:
            dill.dump(iteration_state, destination)

    def save_log(self, main_loop):
        with open(self.path_to_log(), "wb") as destination:
            dill.dump(main_loop, destination)

    def save(self, main_loop):
        """Saves the main loop state to the root folder.

        Overwrites the old data if present.

        """
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        self.save_parameters(main_loop)
        self.save_iteration_state(main_loop)
        self.save_log(main_loop)

    def load_parameters(self):
        return load_parameter_values(self.path_to_parameters())

    def load_iteration_state(self):
        with open(self.path_to_iteration_state(), "rb") as source:
            return dill.load(source)

    def load_log(self, main_loop):
        with open(self.path_to_log(), "rb") as source:
            return dill.load(source)

    def load(self):
        return (self.load_parameters(),
                self.load_iteration_state(),
                self.load_log())

    def load_to(self, main_loop):
        """Loads the state from the root folder into the main loop."""
        parameters, (data_stream, epoch_iterator), log = self.load()
        inject_parameter_values(main_loop.model)
        main_loop.data_stream = data_stream
        main_loop.epoch_iterator = epoch_iterator
        main_loop.log = log
