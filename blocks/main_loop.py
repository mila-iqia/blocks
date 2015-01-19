"""The event-based main loop of Blocks."""
import six

from blocks.log import TrainingLog
from blocks.utils import update_instance, unpack


class MainLoop(object):
    """The standard main loop of Blocks.

    In the `MainLoop` a model is trained by a training algorithm using data
    extracted from a data stream. This process is scrupulously documented
    in a log object.

    The `MainLoop` itself does very little: only fetching the data from the
    data stream and feeding it to the algorithm. It expects the extensions
    to do most of the job. A respective callback of every extension is
    called at every stage of training. The extensions should communicate
    between themselves and with the main loop object by means of making
    records in the log. For instance in order to stop the training
    procedure an extension can make a record
    `training_finish_requested=True` in the log. The main loop checks for
    such a record after every batch and every epoch and terminates when
    finds it.

    Parameters
    ----------
    model : object
        The model object. It is entirely transparent for the main loop
        but may be used by extensions.
    data_stream : instance of :class:`DataStream`.
        The data stream.
    algorithm : object
        The training algorithm.
    log : instance of :class:`TrainingLog`
        The log. When not given, a :class:`RAMTrainingLog` is created.
    extensions : list of :class:`TrainingExtension` instances
        The training extensions. Will be called in the same order as given
        here.

    """
    def __init__(self, model, data_stream, algorithm,
                 log=None, extensions=None):
        if not log:
            log = TrainingLog()
        if not extensions:
            extensions = []
        update_instance(self, locals())

        self.status._training_started = False
        self.status._epoch_started = False

    @property
    def status(self):
        """A shortcut for `self.log.status`."""
        return self.log.status

    def run(self):
        """Starts the main loop.

        The main loop ends when a training extension makes
        a `training_finish_requested` record in the log.

        """
        try:
            if not self.status._training_started:
                for extension in self.extensions:
                    extension.main_loop = self
                self.algorithm.log = self.log
                self._run_extensions('before_training')
                self.algorithm.initialize()
                self.status._training_started = True
            while self._run_epoch():
                pass
        except KeyboardInterrupt:
            self._run_extensions('on_interrupt')
        except TrainingFinish:
            self.log.current_row.training_finished = True
        finally:
            self._run_extensions('after_training')

    def find_extension(self, extension_class):
        """Find an extension of a given class in the list of extensions.

        Parameters:
        ----------
        extension_class : type or str
            The type of extension searched. If a string is given, it is
            matched with class names of the extensions.

        """
        def predicate(extension):
            if isinstance(extension_class, six.string_types):
                return extension.__class__.__name__ == extension_class
            return isinstance(extension, extension_class)
        return unpack([extension for extension in self.extensions
                       if predicate(extension)], singleton=True)

    def _run_extensions(self, method_name, *args):
        for extension in self.extensions:
            extension.dispatch(method_name, *args)

    def _check_finish_training(self):
        if self.log.current_row.training_finish_requested:
            raise TrainingFinish

    def _run_iteration(self):
        try:
            batch = next(self.epoch_iterator)
        except StopIteration:
            return False
        self._run_extensions('before_batch', batch)
        self.algorithm.process_batch(batch)
        self.status.iterations_done += 1
        self._run_extensions('after_batch', batch)
        self._check_finish_training()
        return True

    def _run_epoch(self):
        if not self.status._epoch_started:
            try:
                self.epoch_iterator = self.data_stream.get_epoch_iterator(as_dict=True)
            except StopIteration:
                return False
            self.status._epoch_started = True
            self._run_extensions('before_epoch')
        while self._run_iteration():
            pass
        self.status._epoch_started = False
        self.status.epochs_done += 1
        self.status._epoch_ends.append(
            self.status.iterations_done)
        self._run_extensions('after_epoch')
        self._check_finish_training()
        return True


class TrainingFinish(Exception):
    """An exception raised when a finish request is found in the log."""
    pass
