from blocks.extensions import Timing, FinishAfter
from tests import MockMainLoop


def test_timing():
    main_loop = MockMainLoop(extensions=[Timing(),
                                         FinishAfter(after_n_epochs=2)])
    main_loop.run()
