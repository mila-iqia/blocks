from numpy.testing import assert_raises

from blocks.extensions import SimpleExtension
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.predicates import OnLogRecord


def test_parse_args():
    assert (SimpleExtension.parse_args('before_batch', ('a', 'b')) ==
            (('a',), ('b',)))
    assert (SimpleExtension.parse_args('before_epoch', ('a', 'b')) ==
            ((), ('a', 'b')))


def test_add_list_condition():
    extension_list = Checkpoint('extension_list').add_condition(
        ['before_first_epoch', 'after_epoch'],
        OnLogRecord('notification_name'),
        ('dest_path.kl',))
    extension_iter = Checkpoint('extension_iter')
    extension_iter.add_condition(
        ['before_first_epoch'],
        OnLogRecord('notification_name'),
        ('dest_path.kl',))
    extension_iter.add_condition(
        ['after_epoch'],
        OnLogRecord('notification_name'),
        ('dest_path.kl',))
    assert len(extension_list._conditions) == len(extension_iter._conditions)
    assert_raises(ValueError, extension_iter.add_condition,
                  callbacks_names='after_epoch',
                  predicate=OnLogRecord('notification_name'),
                  arguments=('dest_path.kl',))
