from abc import ABCMeta
from collections import MutableSequence

from six import add_metaclass


@add_metaclass(ABCMeta)
class AnnotatingList(MutableSequence):
    """Mutable sequence performing operations on inserted/removed items.

    Parameters
    ----------
    items : iterable, optional
        An iterable of items to initialize the sequence with.

    """
    def __init__(self, items=None):
        self._items = []
        if not items:
            items = []
        for item in items:
            self.append(item)

    def __repr__(self):
        return repr(self._items)

    def __eq__(self, other):
        return self._items == other

    def __ne__(self, other):
        return self._items != other

    def __getitem__(self, key):
        return self._items[key]

    def _setitem(self, key, value):
        """The operation to perform when an item is inserted/appended."""
        pass

    def _delitem(self, key):
        """The operation to perform when an item is deleted."""
        pass

    def __setitem__(self, key, value):
        self._setitem(key, value)
        self._items[key] = value

    def __delitem__(self, key):
        self._delitem(key)
        del self._items[key]

    def __len__(self):
        return len(self._items)

    def insert(self, key, value):
        self._setitem(key, value)
        self._items.insert(key, value)
