from . import FinishAfter


class FinishIfNoImprovementAfter(FinishAfter):
    """Stop after improvements have ceased for a given period.

    Parameters
    ----------
    notification_name : str
        The name of the log record to look for which indicates a new
        best performer has been found.  Note that the value of this
        record is not inspected.
    iterations : int, optional
        The number of iterations to wait for a new best. Exactly one of
        `iterations` or `epochs` must be not `None` (default).
    epochs : int, optional
        The number of epochs to wait for a new best. Exactly one of
        `iterations` or `epochs` must be not `None` (default).
    patience_log_record : str, optional
        The name under which to record the number of iterations we
        are currently willing to wait for a new best performer.
        Defaults to `notification_name + '_patience_epochs'` or
        `notification_name + '_patience_iterations'`, depending
        which measure is being used.

    Notes
    -----
    By default, runs after each epoch. This can be manipulated via
    keyword arguments (see :class:`blocks.extensions.SimpleExtension`).

    """
    def __init__(self, notification_name, iterations=None, epochs=None,
                 patience_log_record=None, **kwargs):
        if (epochs is None) == (iterations is None):
            raise ValueError("Need exactly one of epochs or iterations "
                             "to be specified")
        self.notification_name = notification_name
        self.iterations = iterations
        self.epochs = epochs
        kwargs.setdefault('after_epoch', True)
        self.last_best_iter = self.last_best_epoch = None
        if patience_log_record is None:
            self.patience_log_record = (notification_name + '_patience' +
                                        ('_epochs' if self.epochs is not None
                                         else '_iterations'))
        else:
            self.patience_log_record = patience_log_record
        super(FinishIfNoImprovementAfter, self).__init__(**kwargs)

    def update_best(self):
        # Here mainly so we can easily subclass different criteria.
        if self.notification_name in self.main_loop.log.current_row:
            self.last_best_iter = self.main_loop.log.status['iterations_done']
            self.last_best_epoch = self.main_loop.log.status['epochs_done']

    def do(self, which_callback, *args):
        self.update_best()
        # If we haven't encountered a best yet, then we should just bail.
        if self.last_best_iter is None:
            return
        if self.epochs is not None:
            since = (self.main_loop.log.status['epochs_done'] -
                     self.last_best_epoch)
            patience = self.epochs - since
        else:
            since = (self.main_loop.log.status['iterations_done'] -
                     self.last_best_iter)
            patience = self.iterations - since

        self.main_loop.log.current_row[self.patience_log_record] = patience
        if patience == 0:
            super(FinishIfNoImprovementAfter, self).do(which_callback,
                                                       *args)
