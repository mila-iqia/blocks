# This is basically equivalent to
#
# from .utils import <everything in UtilsModule.utils_attrs>
# from .theano_utils import <everything in UtilsModule.theano_utils_attrs>
#
# but acts lazily, meaning that the submodule imported only if needed
import sys
import importlib
from types import ModuleType


class UtilsModule(ModuleType):
    utils_attrs = (
        "pack", "unpack", "reraise_as", "dict_subset", "dict_union",
        "repr_attrs", "ipdb_breakpoint", "print_sum", "print_shape",
        "change_recursion_limit", "extract_args", "find_bricks")
    theano_utils_attrs = (
        "shared_floatx_zeros_matching", "shared_floatx_zeros",
        "shared_floatx_nans", "shared_floatx", "shared_like",
        "check_theano_variable", "is_graph_input",
        "is_shared_variable", "put_hook")
    __all__ = utils_attrs + theano_utils_attrs
    __doc__ = __doc__
    __file__ = __file__
    __path__ = __path__

    def __getattr__(self, item):
        # Do lazy import so the submodule imported only if needed.
        # Python manages second import in a way that it is almost free.
        if item in self.utils_attrs:
            utils = importlib.import_module(".utils", __name__)
            return getattr(utils, item)
        elif item in self.theano_utils_attrs:
            theano_utils = importlib.import_module(".theano_utils", __name__)
            return getattr(theano_utils, item)
        else:
            super(UtilsModule, self).__getattribute__(item)


# In Python2 (legacy Python) garbage collector destroys the module
# unless we save a reference
old_module = sys.modules[__name__]
sys.modules[__name__] = UtilsModule(__name__)
