
from __future__ import division, print_function

import dill
import fnmatch
import pandas

from six import iteritems
from collections import OrderedDict
from functools import reduce

from blocks import config
from blocks.utils import change_recursion_limit
from blocks.log import AbstractTrainingLog
from blocks.main_loop import MainLoop


def load_log(fname):
    """Load a :claas:`TrainingLog` object from disk.

    This function automatically handles various file formats that contain
    an instance of an :claas:`AbstractTrainingLog`. This includes a pickled
    Log object, a pickled :claas:`MainLoop` or an experiment dump (TODO).

    """
    with change_recursion_limit(config.recursion_limit):
        with open(fname, 'rb') as f:
            from_disk = dill.load(f)
        # TODO: Load "dumped" experiments

    if isinstance(from_disk, AbstractTrainingLog):
        log = from_disk
    elif isinstance(from_disk, MainLoop):
        log = from_disk.log
        del from_disk
    else:
        raise ValueError("Could not load '{}': Unrecognized content.")

    return log


def print_column_summary(experiments):
    """Print a list of all columns contained in the given experiments.

    Parameters
    ----------
    experiments : OrderedDict of {str: DataFrame}
        The key is expected to be an experiment identifier 
        (e.g. a filename) and the value a pandas.DataFrame.

    """
    channels_per_experiment = OrderedDict(
        [(fname, set(df.columns)) for fname, df in iteritems(experiments)]
    )
    all_channels = reduce(set.union, channels_per_experiment.values())

    print("{} experiment(s):".format(len(experiments)))
    for i, fname in enumerate(experiments):
        print("    {}: {}".format(i, fname))
    print()
    print("Containing the following channels:")
    for ch in sorted(all_channels):
        # create a string indicating which experiments contain which
        #  channels
        indicator = []
        for i, channels in enumerate(channels_per_experiment.values()):
            if ch in channels:
                indicator.append(str(i))
            else:
                indicator.append(" ")
        indicator = ",".join(indicator)
        print("    {}: {}".format(indicator, ch))


def match_column_specs(experiments, column_specs):
    """Filter a dictionary with experiments according to column_specs.

    Parameters
    ----------
    experiments : OrderedDict of {str: DataFrames}
    column_specs : list of str

    Returns
    -------
        Returns a single DataFrame containing the specified
        channels as columns.

    """
    # We iterate over all column and match each spec to the
    # channels of all experiments.
    df = pandas.DataFrame()
    for spec in column_specs:
        if ":" in spec:
            exp_spec, channel_spec = spec.split(":")
        else:
            exp_spec, channel_spec = None, spec

        for i, exp in enumerate(experiments.values()):
            for channel in fnmatch.filter(exp.columns, channel_spec):
                if exp_spec and exp_spec != i:
                    # We are looking for a specific experiment..
                    #  ... and it's not this one.
                    continue

                column_name = "{}:{}".format(i, channel)
                df[column_name] = exp[channel]

    return df


def plot_dataframe(dataframe):
    import pylab

    t = dataframe.index
    print("Plotting {} channels:".format(len(dataframe.columns)))
    for cname, series in iteritems(dataframe):
        print("    {}".format(cname))
        pylab.plot(t, series, label=cname)
    pylab.legend()
    pylab.show(block=True)
