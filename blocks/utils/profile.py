from __future__ import print_function

import sys
import timeit
from collections import defaultdict, OrderedDict


class Profile(object):
    """A profile of hierarchical timers.

    Keeps track of timings performed with :class:`Timer`. It also keeps
    track of the way these timings were nested and makes use of this
    information when reporting.

    """
    def __init__(self):
        self.total = defaultdict(int)
        self.current = []
        self.order = OrderedDict()

    def enter(self, name):
        self.current.append(name)
        # We record the order in which sections were first called
        self.order[tuple(self.current)] = None

    def exit(self, t):
        self.total[tuple(self.current)] += t
        self.current.pop()

    def report(self, f=sys.stderr):
        """Print a report of timing information to standard output.

        Parameters
        ----------
        f : object, optional
            An object with a ``write`` method that accepts string inputs.
            Can be a file object, ``sys.stdout``, etc. Defaults to
            ``sys.stderr``.

        """
        total = sum(v for k, v in self.total.items() if len(k) == 1)

        def print_report(keys, level=0):
            subtotal = 0
            for key in keys:
                if len(key) > level + 1:
                    continue
                subtotal += self.total[key]
                section = ' '.join(key[-1].split('_'))
                section = section[0].upper() + section[1:]
                print('{:30}{:15.2f}{:15.2%}'.format(
                    level * '  ' + section, self.total[key],
                    self.total[key] / total
                ), file=f)
                children = [k for k in keys
                            if k[level] == key[level] and
                            len(k) > level + 1]
                child_total = print_report(children, level + 1)
                if children:
                    print('{:30}{:15.2f}{:15.2%}'.format(
                        (level + 1) * '  ' + 'Other',
                        self.total[key] - child_total,
                        (self.total[key] - child_total) / total
                    ), file=f)
            return subtotal

        print('{:30}{:>15}{:>15}'.format('Section', 'Time', '% of total'),
              file=f)
        print('-' * 60, file=f)
        if total:
            print_report(self.order.keys())
        else:
            print('No profile information collected.', file=f)


class Timer(object):
    """A context manager to time the execution time of code within it.

    This timer is attached to a :class:`Profile` object that it reports
    timings to. The :class:`Profile` object accumulates the timings.
    Timers can be nested, which the :class:`Profile` will automatically
    keep track of and use in its reporting.

    Parameters
    ----------
    name : str
        The name of this section. Expected to adhere to variable naming
        styles.
    profile : :class:`Profile`
        The profile of the main loop. This is the object this context
        manager will report the execution time to. The accumulation and
        processing of timing information is handled by this object.

    Notes
    -----
    Timings are reported using :func:`timeit.default_timer`.

    """
    def __init__(self, name, profile):
        self.name = name
        self.profile = profile

    def __enter__(self):
        self.profile.enter(self.name)
        self.start = timeit.default_timer()

    def __exit__(self, *args):
        self.profile.exit(timeit.default_timer() - self.start)
