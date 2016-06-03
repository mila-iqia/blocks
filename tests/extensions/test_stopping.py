from collections import defaultdict
import unittest
from blocks.extensions.stopping import FinishIfNoImprovementAfter


class FakeLog(defaultdict):
    epoch_length = 4

    def __init__(self):
        super(FakeLog, self).__init__(dict)
        self.status = {'iterations_done': 0, 'epochs_done': 0}

    def advance(self, epochs):
        self.status['iterations_done'] += self.epoch_length if epochs else 1
        self.status['epochs_done'] += (epochs if epochs
                                       else (self.status['iterations_done'] //
                                             self.epoch_length))

    @property
    def current_row(self):
        return self[self.status['iterations_done']]


class FakeMainLoop(object):
    def __init__(self, extensions=()):
        self.log = FakeLog()
        self.extensions = list(extensions)

    @property
    def status(self):
        return self.log.status


class FinishIfNoImprovementAfterTester(unittest.TestCase):
    def check_not_stopping(self):
        finish = self.main_loop.log.current_row.get(
            'training_finish_requested', False)
        self.assertFalse(finish)

    def check_stopping(self):
        finish = self.main_loop.log.current_row.get(
            'training_finish_requested', False)
        self.assertTrue(finish)

    def check_log(self, record_name, value):
        self.assertEqual(self.main_loop.log.current_row[record_name], value)

    def setUp(self):
        self.main_loop = FakeMainLoop()

    def check_finish_if_no_improvement_after(self, ext, notification_name,
                                             log_entry=None, epochs=False):
        # ext = FinishIfNoImprovementAfter('bananas', iterations=3)
        ext.main_loop = self.main_loop
        # This should silently pass, since there has been no new best set,
        # and 'bananas' is not in the log's current row.
        which_callback = 'after_batch' if not epochs else 'after_epoch'
        if log_entry is None:
            log_entry = notification_name + '_patience_' + (
                'iterations' if not epochs else 'epochs')
        # First is a new best.
        self.main_loop.log.current_row[notification_name] = True
        ext.do(which_callback)
        self.check_log(log_entry, 3)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # No best found for another 2 iterations.
        ext.do(which_callback)
        self.check_log(log_entry, 2)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # One iteration down, one to go.
        ext.do(which_callback)
        self.check_log(log_entry, 1)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # Oh look, a new best!
        self.main_loop.log.current_row[notification_name] = True
        ext.do(which_callback)
        self.check_log(log_entry, 3)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # Now, run out our patience. 3 iterations with no best.
        ext.do(which_callback)
        self.check_log(log_entry, 2)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        ext.do(which_callback)
        self.check_log(log_entry, 1)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        ext.do(which_callback)
        self.check_log(log_entry, 0)
        self.check_stopping()

    def test_finish_if_no_improvement_after_iterations(self):
        ext = FinishIfNoImprovementAfter('bananas', iterations=3)
        self.check_finish_if_no_improvement_after(ext, 'bananas')

    def test_finish_if_no_improvement_after_epochs(self):
        ext = FinishIfNoImprovementAfter('mangos', epochs=3)
        self.check_finish_if_no_improvement_after(ext, 'mangos', epochs=True)

    def test_finish_if_no_improvement_after_epochs_log_record_specified(self):
        ext = FinishIfNoImprovementAfter('melons',
                                         patience_log_record='blueberries',
                                         iterations=3)
        self.check_finish_if_no_improvement_after(ext, 'melons', 'blueberries')

    def test_finish_if_no_improvement_after_specify_both(self):
        self.assertRaises(ValueError, FinishIfNoImprovementAfter, 'boo',
                          epochs=4, iterations=5)
