from operator import getitem

from numpy.testing import assert_raises

from blocks.log import TrainingLog


def test_training_log():
    log = TrainingLog()

    # test basic writing capabilities
    log[0]['field'] = 45
    assert log[0]['field'] == 45
    assert log[1] == {}
    assert log.current_row['field'] == 45
    log.status['iterations_done'] += 1
    assert log.status['iterations_done'] == 1
    assert log.previous_row['field'] == 45

    assert_raises(ValueError, getitem, log, -1)

    # test iteration
    assert len(list(log)) == 2
