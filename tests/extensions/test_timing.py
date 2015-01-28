from blocks.extensions import Timing
from blocks.log import TrainingLog

def test_timing():
    class FakeMainLoop():

        def __init__(self):
            self.log = TrainingLog()

    now = 0
    timing = Timing(lambda: now)
    ml = FakeMainLoop()
    timing.main_loop = ml

    now = 1
    timing.before_training()
    now = 3
    timing.before_epoch()
    assert ml.log[0].initialization_took == 2

    # Batch 1
    timing.before_batch(None)
    now = 10
    ml.log.status.iterations_done += 1
    timing.after_batch(None)
    assert ml.log[1].iteration_took == 7

    # Batch 2
    timing.before_batch(None)
    now = 18
    ml.log.status.iterations_done += 1
    timing.after_batch(None)
    assert ml.log[2].iteration_took == 8

    timing.after_epoch()
    assert ml.log[2].epoch_took == 15

    timing.after_training()
    assert ml.log[2].training_took == 17



