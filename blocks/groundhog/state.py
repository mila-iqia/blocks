class GroundhogState(object):
    """Good default values for groundhog state."""

    def __init__(self, prefix, batch_size, learning_rate, **kwargs):
        self.prefix = prefix
        self.bs = batch_size
        self.lr = learning_rate
        self.seed = 1
        # TODO: what does it mean?
        self.patience = 1
        self.minerr = -1
        self.timeStop = 10 ** 9
        self.minlr = 0
        self.overwrite = True
        self.hookFreq = -1
        self.saveFreq = 30
        self.validFreq = 10 ** 9
        self.trainFreq = 1
        self.loopIters = 10 ** 6

    def as_dict(self):
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}
