
from __future__ import division, print_function

import dill
import fnmatch
import logging
import readline
import pandas

from six import iteritems
from six.moves import input
from collections import OrderedDict
from functools import reduce

from blocks import config
from blocks.utils import change_recursion_limit


def match_channel_specs(experiments, channel_specs):
    """Filter a dictionary with experiments according to channel_specs.

    Parameters
    ----------
    experiments : OrderedDict of {str: DataFrames}
    channel_specs : list of str

    Returns
    -------
        Returns a single DataFrame containing the specified
        channels as columns.

    """
    # We iterate over all channel_specs and match each spec to the
    # channels of all experiments.
    df = pandas.DataFrame()
    for spec in channel_specs:
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


PROMPT_HEADER = """

Type a comma separated list of channels to plot or [q]uit.

Channels may be prefixed by <number>: to refer to a specific experiment \
and may contain '*' or '?' characters to match multiple channels at once.

"""

IPYTHON_HEADER = """

Complete DataFrames for all experiments can be found in the \
'experiments[fname]' dictionary. The DataFrame containing only \
the selected channels can be access as 'matched'

"""


def plot(exp_fnames, channel_specs, list_only=False, ipython=False):
    # Load and convert experiments into DataFrames...
    experiments = OrderedDict()
    for fname in exp_fnames:
        logging.info("Loading '{0}'...".format(fname))
        try:
            # TODO: Load "dumped" experiments
            with change_recursion_limit(config.recursion_limit):
                main_loop = dill.load(open(fname, "rb"))
        except EnvironmentError:
            logging.error("Could not open '{}'".format(fname))

        rows_to_keep = [0] + main_loop.log._status._epoch_ends
        data_frame = main_loop.log.to_dataframe()
        data_frame = data_frame.iloc[rows_to_keep]
        experiments[fname] = data_frame
        del main_loop

    # Gain some overview
    n_experiments = len(experiments)
    channels_per_experiment = OrderedDict(
        [(fname, set(df.columns)) for fname, df in iteritems(experiments)]
    )
    all_channels = reduce(set.union, channels_per_experiment.values())

    if list_only:
        # Print all channels contained in the specified experiments
        print("\nLoaded {} experiment(s):".format(n_experiments))
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
        print()
        return

    if channel_specs:
        channel_specs = channel_specs.split(',')
        matched = match_channel_specs(experiments, channel_specs)

        if ipython:
            from IPython import embed
            print(IPYTHON_HEADER)
            embed()
            return

        plot_dataframe(matched)
        return
    else:
        # Interactive mode
        def completer(text, state):
            """Completion callback function for readline library."""
            options = []
            if text == "":
                options += [str(i)+":" for i in range(n_experiments)]
            if ":" in text:
                exp_id, text = text.split(":")
                channels = channels_per_experiment.values()[int(exp_id)]
                options += [
                    exp_id+":"+ch for ch in channels if ch.startswith(text)
                ]
            else:
                options += [
                    ch for ch in all_channels if ch.startswith(text)
                ]

            if state >= len(options):
                return None
            return sorted(options)[state]

        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(" \t,;")
        readline.set_completer(completer)

        print()
        print("Experiments loaded:")
        for i, exp_name in enumerate(experiments.keys()):
            print("    {}: {}".format(i, exp_name))
        print(PROMPT_HEADER)
        while 1:
            # Note: input() uses python3 semantics (provided by six)
            channel_spec = input("blocks-plot> ")
            if channel_spec in ["q", "quit", "exit", "e"]:
                break

            channel_spec = channel_spec.split(',')
            matched = match_channel_specs(experiments, channel_spec)

            if len(matched.columns) == 0:
                print("Your specification did not match any channels.")
                continue

            plot_dataframe(matched)
            print()
