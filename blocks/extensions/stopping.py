import logging

from . import FinishAfter, CompositeExtension
from .training import TrackTheBest
from .predicates import OnLogRecord


logger = logging.getLogger(__name__)


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
        logger.debug('%s: Writing patience of %d to current log record (%s) '
                     'at iteration %d', self.__class__.__name__, patience,
                     self.patience_log_record,
                     self.main_loop.log.status['iterations_done'])
        self.main_loop.log.current_row[self.patience_log_record] = patience
        if patience == 0:
            super(FinishIfNoImprovementAfter, self).do(which_callback,
                                                       *args)


class EarlyStopping(CompositeExtension):
    """A 'batteries-included' early stopping extension.

    Parameters
    ----------
    record_name : str
        The log record entry whose value represents the quantity to base
        early stopping decisions on, e.g. some measure of validation set
        performance.
    checkpoint_extension : :class:`~blocks.extensions.Checkpoint`, optional
        A :class:`~blocks.extensions.Checkpoint` instance to configure to
        save a checkpoint when a new best performer is found.
    checkpoint_filename : str, optional
        The filename to use for the 'current best' checkpoint. Must be
        provided if ``checkpoint_extension`` is specified.
    notification_name : str, optional
        The name to be written in the log when a new best-performing
        model is found. Defaults to ``record_name + '_best_so_far'``.
    choose_best : callable, optional
        See :class:`TrackTheBest`.
    iterations : int, optional
        See :class:`FinishIfNoImprovementAfter`.
    epochs : int, optional
        See :class:`FinishIfNoImprovementAfter`.

    Notes
    -----
    Trigger keyword arguments will affect how often the log is inspected
    for the record name (in order to determine if a new best has been
    found), as well as how often a decision is made about whether to
    continue training. By default, ``after_epoch`` is set,
    as is ``before_training``, where some sanity checks are performed
    (including the optional self-management of checkpointing).

    If ``checkpoint_extension`` is not in the main loop's extensions list
    when the `before_training` trigger is run, it will be added as a
    sub-extension of this object.

    """

    def __init__(self, record_name, checkpoint_extension=None,
                 checkpoint_filename=None, notification_name=None,
                 choose_best=min, iterations=None, epochs=None, **kwargs):
        if notification_name is None:
            notification_name = record_name + '_best_so_far'
        kwargs.setdefault('after_epoch', True)
        tracking_ext = TrackTheBest(record_name, notification_name,
                                    choose_best=choose_best, **kwargs)
        stopping_ext = FinishIfNoImprovementAfter(notification_name,
                                                  iterations=iterations,
                                                  epochs=epochs,
                                                  **kwargs)
        self.checkpoint_extension = checkpoint_extension
        if checkpoint_extension and checkpoint_filename:
            checkpoint_extension.add_condition(['after_batch'],
                                               OnLogRecord(notification_name),
                                               (checkpoint_filename,))
        elif checkpoint_extension is not None and checkpoint_filename is None:
            raise ValueError('checkpoint_extension specified without '
                             'checkpoint_filename')
        kwargs.setdefault('before_training', True)
        super(EarlyStopping, self).__init__([tracking_ext, stopping_ext],
                                            **kwargs)

    def do(self, which_callback, *args):
        if which_callback == 'before_training' and self.checkpoint_extension:
            if self.checkpoint_extension not in self.main_loop.extensions:
                logger.info('%s: checkpoint extension %s not in main loop '
                            'extensions, adding as sub-extension of %s',
                            self.__class__.__name__, self.checkpoint_extension,
                            self)
                self.checkpoint_extension.main_loop = self.main_loop
                self.sub_extensions.append(self.checkpoint_extension)
            else:
                exts = self.main_loop.extensions
                if exts.index(self.checkpoint_extension) < exts.index(self):
                    logger.warn('%s: configured checkpointing extension '
                                'appears after this extension in main loop '
                                'extensions list. This may lead to '
                                'unwanted results, as the notification '
                                'that would trigger serialization '
                                'of a new best will not have been '
                                'written yet when the checkpointing '
                                'extension is run.', self.__class__.__name__)
