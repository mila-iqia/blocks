from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter


def test_main_loop():

    class TestDataStream(object):
        def __iter__(self):
            yield [1, 2, 3]
            yield [4, 5]
            yield [6, 7, 8, 9]

    class TestTrainer(object):

        def initialize(self):
            pass

        def process_batch(self, batch):
            self.log.current_row.batch = batch

    finish_extension = FinishAfter()
    finish_extension.add_condition(
        'after_epoch', predicate=lambda log: log.status.epochs_done == 2)
    main_loop = MainLoop(None, TestDataStream(), TestTrainer(),
                         None, [finish_extension])
    main_loop.run()

    assert main_loop.log.status.iterations_done == 5
    assert len(list(main_loop.log)) == 7
    for i in range(5):
        assert main_loop.log[i].batch == i + 1
