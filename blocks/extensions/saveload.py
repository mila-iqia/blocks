"""Extensions for saving and loading the state of a training process."""
import dill

from blocks.extensions import SimpleExtension

class Checkpointing(SimpleExtension):
    """Creates a checkpoint of the training process.

    This extension saves a pickled version of the main loop to the disk.
    This pickle can be later reloaded and training can be resumed.

    Parameters
    ----------
    path : str
        The destination path for pickling.

    Notes
    -----
    Using pickle for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.

    * Instead of the standard pickling library, the dill package is used.

    """
    def __init__(self, path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(Checkpointing, self).__init__(self, **kwargs)
        self.path = path

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk."""
        with open(self.path, "wb") as destination:
            dill.dump(self.main_loop, destination, dill.HIGHEST_PROTOCOL)
