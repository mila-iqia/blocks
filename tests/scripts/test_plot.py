

import blocks.scripts.plot as plot

from collections import OrderedDict
from tests import silence_printing, skip_if_not_available

try:
    from pandas import DataFrame
    PANDAS_AVAILABLE = True
except:
    PANDAS_AVAILABLE = False


def some_experiments():
    """Create some 2 dummy experiments."""
    experiments = OrderedDict()
    experiments['exp0'] = DataFrame()
    experiments['exp0']['col0'] = (0, 1, 2)
    experiments['exp0']['col1'] = (3, 4, 5)
    experiments['exp1'] = DataFrame()
    experiments['exp1']['col0'] = (6, 7, 8)
    experiments['exp1']['col1'] = (9, 9, 9)
    return experiments


@silence_printing
def test_print_column_summary():
    skip_if_not_available(modules=['pandas'])
    experiments = some_experiments()
    plot.print_column_summary(experiments)


def test_match_column_specs():
    skip_if_not_available(modules=['pandas'])
    experiments = some_experiments()
    specs = ['0:col0', '*1']
    df = plot.match_column_specs(experiments, specs)

    assert isinstance(df, DataFrame)
    assert list(df.columns) == ['0:col0', '0:col1', '1:col1']
