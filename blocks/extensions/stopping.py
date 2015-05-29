from . import FinishAfter


class FinishIfNoImprovementAfter(FinishAfter):
    """Stop after improvements have ceased for a given period.

    Parameters
    ----------
    iterations : int
        The number of iterations to wait for a new best.
    notification_name : str
        The name of the log record to look for which indicates a new
        best performer has been found.  Note that the value of this
        record is not inspected.

    """
    def __init__(self, iterations, notification_name, **kwargs):
        self.iterations = iterations
        self.notification_name = notification_name
        kwargs.setdefault('after_batch', True)
        self.last_best = None
        super(FinishIfNoImprovementAfter, self).__init__(**kwargs)

    def update_best(self):
        # Here mainly so we can easily subclass different criteria.
        if self.notification_name in self.main_loop.log.current_row:
            self.last_best = self.main_loop.log.status['iterations_done']

    def do(self, which_callback, *args):
        self.update_best()
        iters_since = (self.main_loop.log.status['iterations_done'] -
                       self.last_best)
        if iters_since >= self.iterations:
            super(FinishIfNoImprovementAfter, self).do(which_callback,
                                                       *args)
