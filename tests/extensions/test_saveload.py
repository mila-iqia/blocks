from blocks.extensions.saveload import Checkpoint


def test_checkpoint_save_separately_paths():
    class FakeMainLoop(object):
        def __init__(self):
            self.foo = 'abcdef'
            self.bar = {'a': 1}
            self.baz = 351921

    chkpt = Checkpoint(path='myweirdmodel.picklebarrel',
                       save_separately=['foo', 'bar'])
    expected = {'foo': 'myweirdmodel_foo.picklebarrel',
                'bar': 'myweirdmodel_bar.picklebarrel'}
    assert chkpt.save_separately_filenames(chkpt.path) == expected
    expected = {'foo': 'notmodelpath_foo',
                'bar': 'notmodelpath_bar'}
    assert chkpt.save_separately_filenames('notmodelpath') == expected
