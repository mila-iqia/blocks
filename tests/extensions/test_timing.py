from numpy.testing import assert_allclose

from blocks.extensions import Timing, FinishAfter
from blocks.utils.testing import MockMainLoop


def test_timing():
    main_loop = MockMainLoop(extensions=[Timing(prefix='each_'),
                                         Timing(prefix='each_second_'),
                                         FinishAfter(after_n_epochs=2)])
    main_loop.run()
    assert_allclose(
        (main_loop.log.current_row['each_time_train_this_epoch'] +
         main_loop.log.previous_row['each_time_train_this_epoch']) / 2,
        main_loop.log.current_row['each_second_time_train_this_epoch'])
