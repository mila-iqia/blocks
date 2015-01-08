"""The event-based main loop of Blocks."""
from abc import ABCMeta

from blocks.log import RAMTrainingLog
from blocks.utils import update_instance


class MainLoop(object):
    """The standard main loop of Blocks.

    In the `MainLoop` a model is trained by a trainer using data extracted
    from a data stream. This process is scrupulously documented in a log
    object.

    The `MainLoop` itself does very little: only fetching the data from the
    data stream and feeding it to the trainer. It expects the extensions to
    do most of the job. A respective callback of every extension is called
    at every stage of training. The extensions should communicate between
    themselves and with the main loop object by means of making records in
    the log. For instance in order to stop the training procedure an
    extension can make a record `training_finish_requested=True` in the
    log. The main loop checks for such a record after every batch and every
    epoch and terminates when finds it.

    Parameters
    ----------
    model : object
        The model object. It is entirely transparent for the main loop
        but may be used by extensions.
    data_stream : instance of :class:`DataStream`.
        The data stream.
    trainer : object
        The trainer.
    log : instance of :class:`TrainingLog`
        The log. When not given, a :class:`RAMTrainingLog` is created.
    extensions : list of :class:`TrainingExtension` instances
        The training extensions. Will be called in the same order as given
        here.

    """
    __metaclass__ = ABCMeta

    def __init__(self, model, data_stream, trainer,
                 log=None, extensions=None):
        if not log:
            log = RAMTrainingLog()
        if not extensions:
            extensions = []
        update_instance(self, locals())

    def _run_extensions(self, method_name, *args):
        for extension in self.extensions:
            getattr(extension, method_name)(*args)

    def _check_finish_training(self):
        if self.log.current_row.training_finish_requested:
            raise StopIteration()

    def run(self):
        """Starts the main loop.

        The main loop ends when a training extension makes
        a `training_finish_requested` record in the log.

        """
        try:
            for extension in self.extensions:
                extension.main_loop = self
            self.trainer.log = self.log
            self._run_extensions('before_training')
            self.trainer.initialize()
            for epoch in self.data_stream:
                self._run_extensions('before_epoch')
                for batch in epoch:
                    self._run_extensions('before_batch', batch)
                    self.trainer.process_batch(batch)
                    self.log.status.iterations_done += 1
                    self._run_extensions('after_batch', batch)
                    self._check_finish_training()
                self.log.status.epochs_done += 1
                self._run_extensions('after_epoch')
                self._check_finish_training()
                self.log.status.last_epoch_end = (
                    self.log.status.iterations_done)
        except KeyboardInterrupt:
            self._run_extensions('on_interrupt')
        except StopIteration:
            self.log.current_row.training_finished = True
        finally:
            self._run_extensions('after_training')
