from numpy.testing import assert_allclose

from blocks.extensions import Timing, FinishAfter
from blocks.utils.testing import MockMainLoop


def test_timing():
    epochs = 2
    main_loop = MockMainLoop(delay_time=0.1,
                             extensions=[Timing(prefix='each'),
                                         Timing(prefix='each_second'),
                                         FinishAfter(after_n_epochs=epochs)])
    main_loop.run()
    iterations = main_loop.log.status['iterations_done'] / epochs
    assert_allclose(
        (main_loop.log[iterations]['each_time_train_this_epoch'] +
         main_loop.log[iterations]['each_time_train_this_epoch']) / 2,
        main_loop.log.current_row['each_second_time_train_this_epoch'])
