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
                        if key.find('batch') == -1
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

    Often an extension has one or a few methods that should when certain
    conditions are met, e.g. after each tenth epoch or when a certain
    record record is found in the log. This class provides a platform
    to build such extensions.

    When all logic of the extension is concentrated in one method, this
    method is called the main method.

    Attributes
    ----------
    main_method : str
        The name of the main method of the extension.

    """
    def __init__(self):
        self._conditions = []

    def add_condition(self, callback_name, method_name=None, predicate=None):
        """Adds a condition under which a certain method is called.

        Parameters
        ----------
        callback_name : str
            The name of the callback in which the method.
        method_name : str
            The name of the method to be called. If ``None``, the main
            method is called.
        predicate : function
            A predicate function the main loop's log as the
            single parameter and returning ``True`` when the method
            should be called and ``False`` when should not. If ``None``,
            an always ``True`` predicate is used.

        """
        if not method_name:
            method_name = self.main_method
        if not predicate:
            predicate = lambda log: True
        self._conditions.append((method_name, callback_name, predicate))

    def execute(self, callback_invoked, *args):
        """Execute methods corresponding to the invoked callback."""
        for method_name, callback_name, predicate in self._conditions:
            if (callback_name == callback_invoked
                    and predicate(self.main_loop.log)):
                getattr(self, method_name)(*args)


class FinishAfter(SimpleExtension):
    """Finishes the training process when triggered."""
    main_method = 'finish_training'

    def __init__(self):
        super(FinishAfter, self).__init__()

    def finish_training(self):
        self.main_loop.log.current_row.training_finish_requested = True
