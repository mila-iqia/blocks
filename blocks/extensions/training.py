from blocks.extensions import TrainingExtension


class AdjustParameter(TrainingExtension):
    """Adjusts shared variable parameter using some function.

    Applies function to compute the new value of a shared parameter each
    iteration.

    This class can be used to adapt over the training process parameters like
    learning rate, momentum, etc.

    Parameters
    ----------
    parameter : shared variable to be adjusted
    function : function which input number of aggregated examples and outputs
               the parameter value

    """
    def __init__(self, parameter, function, **kwargs):
        super(AdjustParameter, self).__init__(**kwargs)
        self.learning_rate = parameter
        self.function = function
        self.num_examples = 0

    def after_batch(self, batch):
        self.num_examples += batch.values()[0].shape[0]
        self.learning_rate.set_value(self.function(self.num_examples))