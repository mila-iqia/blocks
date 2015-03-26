from abc import ABCMeta, abstractmethod
from collections import MutableSequence, MutableSet

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

    @abstractmethod
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


class OrderedSet(MutableSet):
    """A set which remembers the order in which elements were inserted.

    Parameters
    ----------
    iterable : iterable
        An iterable with items that should be added to the ordered set.

    Notes
    -----
    Originally by Raymond Hettinger [1]_, with minor changes.

    .. [1] https://code.activestate.com/recipes/576694-orderedset/

    """
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next_ = self.map.pop(key)
            prev[2] = next_
            next_[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        """Pop the first or last inserted item.

        Parameters
        ----------
        last : bool
            If ``True``, pop the last inserted element. Otherwise pop the
            first inserted element.

        """
        if not self:
            raise KeyError('pop from an empty ordered set')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '{}()'.format(self.__class__.__name__,)
        return '{}({})'.format(self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        try:
            return set(self) == set(other)
        except TypeError:
            return False
