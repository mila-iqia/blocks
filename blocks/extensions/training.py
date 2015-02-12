import inspect
from blocks.extensions import TrainingExtension


class SharedVariableModifier(TrainingExtension):
    """Adjusts shared variable parameter using some function.

    Applies function to compute the new value of a shared parameter each
    iteration.

    This class can be used to adapt over the training process parameters like
    learning rate, momentum, etc.

    Parameters
    ----------
    parameter : :class:`~tensor.TensorSharedVariable`
        shared variable to be adjusted
    function : callable
        a function which outputs a numeric value to which the
        given shared variable will be set and may take one or two arguments.

        In the first case, function that takes the total number of examples
        seen (``int``) as an input.

        In the second case, it is a function which takes number of examples
        seen (``int``) and old value of the shared variable.

    """
    def __init__(self, parameter, function, **kwargs):
        super(SharedVariableModifier, self).__init__(**kwargs)
        self.parameter = parameter
        self.function = function
        self.num_examples = 0
        self.num_args = len(inspect.getargspec(function).args)

    def after_batch(self, batch):
        self.num_examples += batch.values()[0].shape[0]
        if self.num_args == 1:
            new_value = self.function(self.num_examples)
        else:
            old_value = self.parameter.get_value()
            new_value = self.function(self.num_examples, old_value)
        self.parameter.set_value(new_value)