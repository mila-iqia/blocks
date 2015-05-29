from blocks.extensions.stopping import FinishIfNoImprovementAfter


class FakeLog(object):
    current_row = {}
    status = {'iterations_done': 0}


class FakeMainLoop(object):
    log = FakeLog()


def test_finish_if_no_improvement_after():
    main_loop = FakeMainLoop()

    def check_not_stopping():
        finish = main_loop.log.current_row.get('training_finish_requested',
                                               False)
        assert not finish

    def check_stopping():
        finish = main_loop.log.current_row.get('training_finish_requested',
                                               False)
        assert finish

    ext = FinishIfNoImprovementAfter(3, 'bananas')
    ext.main_loop = main_loop
    # First is a new best.
    main_loop.log.current_row['bananas'] = True
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_not_stopping()
    # No best found for another  2 iterations.
    del main_loop.log.current_row['bananas']
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_not_stopping()
    # One iteration down, one to go.
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_not_stopping()
    # Oh look, a new best!
    main_loop.log.current_row['bananas'] = True
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_not_stopping()
    # Now, run out our patience. 3 iterations with no best.
    del main_loop.log.current_row['bananas']
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_not_stopping()
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_not_stopping()
    main_loop.log.status['iterations_done'] += 1
    ext.do('after_batch')
    check_stopping()
