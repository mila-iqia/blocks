from __future__ import print_function
from abc import ABCMeta, abstractmethod

class TrainingExtension(object):
    """The base class for training extensions.

    An extension is a set of callbacks sharing a joint context that are
    invoked at certain stages of the training procedure. This callbacks
    typically add a certain functionality to the training procedure,
    e.g. running validation on auxiliarry datasets or early stopping.

    Attributes
    ----------
    main_loop : :class:`MainLoop`
        The main loop to which the extensions belongs.

    """
    __metaclass__ = ABCMeta

    def before_training(self):
        """The callback invoked before training is started."""
        pass

    def before_epoch(self):
        """The callback invoked before starting an epoch."""
        pass

    def before_batch(self, batch):
        """The callback invoked before a batch is processed.

        Parameters
        ----------
        batch : object
            The data batch to be processed.

        """
        pass

    def after_batch(self, batch):
        """The callback invoked after a batch is processed.

        Parameters
        ----------
        batch : object
            The data batch just processed.

        """
        pass

    def after_epoch(self):
        """The callback invoked after an epoch is finished."""
        pass

    def after_training(self):
        """The callback invoked after training is finished."""
        pass

    def on_interrupt(self):
        """The callback invoked when training is interrupted."""
        pass


# To avoid copy-pasting we override callbacks in SimpleExtension
# programatically.
def replace_callbacks(class_):
    for key in TrainingExtension.__dict__:
        if not key.startswith('_'):
            def create_callback_overrider(key):
                def simple_callback_overrider(self):
                    self.execute(key)

                def batch_callback_overrider(self, batch):
                    self.execute(key, batch)
                # TODO: use proper reflection here
                return (simple_callback_overrider
                        if not 'batch' in key
                        else batch_callback_overrider)
            callback_overrider = create_callback_overrider(key)
            callback_overrider.__name__ = key
            callback_overrider.__doc__ = (
                """Execute methods corresponding to this callback.""")
            setattr(class_, key, callback_overrider)
    return class_


@replace_callbacks
class SimpleExtension(TrainingExtension):
    """A base class for simple extensions.

    All logic of simple extensions is concentrated in the method
    :meth:`do`.  This method is called when certain conditions are
    fulfilled. The user can manage the conditions by calling the
    `add_condition` method and by passing arguments to the constructor.  In
    addition to specifying when :meth:`do` is called, it is possible to
    specify additional arguments passed to :meth:`do` under different
    conditions.

    Parameters
    ----------
    before_first_epoch : bool
        If ``True``, :meth:`do` is invoked before the first epoch.
    after_every_epoch : bool
        If ``True``, :meth:`do` is invoked after every epoch.
    after_every_iteration : bool
        If ``True``, :meth:`do` is invoked after every iteration.
    after_training : bool
        If ``True``, :meth:`do` is invoked after training.

    """
    def __init__(self, before_first_epoch=False, after_every_epoch=False,
                 after_every_iteration=False, after_training=False):
        self._conditions = []
        if before_first_epoch:
            self.add_condition(
                "before_epoch",
                predicate=lambda log: log.status.epochs_done == 0)
        if after_every_epoch:
            self.add_condition("after_epoch")
        if after_every_iteration:
            self.add_condition("after_iteration")
        if after_training:
            self.add_condition("after_training")

    def add_condition(self, callback_name, predicate=None, arguments=None):
        """Adds a condition under which a :meth:`do` is called.

        Parameters
        ----------
        callback_name : str
            The name of the callback in which the method.
        predicate : function
            A predicate function the main loop's log as the
            single parameter and returning ``True`` when the method
            should be called and ``False`` when should not. If ``None``,
            an always ``True`` predicate is used.
        arguments : iterable
            Additional arguments to be passed to :meth:`do`. They will
            be concatenated with the ones passed from the main loop
            (e.g. the batch in case of `after_epoch` callback).

        """
        if not arguments:
            arguments = []
        if not predicate:
            predicate = lambda log: True
        self._conditions.append((callback_name, predicate, arguments))

    @abstractmethod
    def do(self, which_callback, *args):
        """Does the job of the training extension.

        Parameters
        ----------
        which_callback : str
            The name of the callback in the context of which :meth:`do` is
            run.
        *args : tuple
            The arguments from the main loop concatenated with additional
            arguments from user.

        """
        pass

    def execute(self, callback_invoked, *from_main_loop):
        """Execute methods corresponding to the invoked callback."""
        for callback_name, predicate, arguments in self._conditions:
            if (callback_name == callback_invoked
                    and predicate(self.main_loop.log)):
                self.do(callback_invoked, *(from_main_loop + tuple(arguments)))


class FinishAfter(SimpleExtension):
    """Finishes the training process when triggered.

    Parameters
    ----------
    num_epochs : int
        If not ``None``, training finish is requested after `num_epochs`
        are done.

    """

    def __init__(self, num_epochs=None):
        super(FinishAfter, self).__init__()
        if num_epochs:
            self.add_condition(
                "after_epoch",
                predicate=lambda log: log.status.epochs_done == num_epochs)

    def do(self, which_callback):
        self.main_loop.log.current_row.training_finish_requested = True


class Printing(SimpleExtension):
    """Prints log messages to the screen."""

    def __init__(self, **kwargs):
        def set_if_absent(name):
            if not name in kwargs:
                kwargs[name] = True
        set_if_absent("before_first_epoch")
        set_if_absent("after_training")
        set_if_absent("after_every_epoch")
        super(Printing, self).__init__(**kwargs)

    def _print_attributes(self, attribute_tuples):
        for attr, value in attribute_tuples:
            if not attr.startswith("_"):
                print("\t", "{}:".format(attr), value)

    def do(self, which_callback):
        log = self.main_loop.log
        print("".join(79 * "-"))
        if which_callback == "before_epoch" and log.status.epochs_done == 0:
            print("BEFORE FIRST EPOCH")
        elif which_callback == "after_training":
            print("TRAINING HAS BEEN FINISHED:")
        elif which_callback == "after_epoch":
            print("AFTER ANOTHER EPOCH")
        print("".join(79 * "-"))
        print("Training status:")
        self._print_attributes(log.status)
        print("Log records from the iteration {}:".format(
            log.status.iterations_done))
        self._print_attributes(log.current_row)
