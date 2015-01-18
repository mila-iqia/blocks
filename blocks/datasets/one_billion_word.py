import os

from blocks import config
from blocks.datasets import Dataset
from blocks.datasets.schemes import ConstantScheme
from blocks.utils import update_instance


class OneBillionWord(Dataset):
    """Google's One Billion Word benchmark.

    This monolingual corpus contains 829,250,940 tokens (including sentance
    boundary markers). The data is split into 100 partitions, one of which
    is the held-out set. This held-out set is further divided into 50
    partitions. More information about the dataset can be found in
    [CMSG14].

    .. [CSMG14] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, and
    Thorsten Brants, *One Billion Word Benchmark for Measuring Progress in
    Statistical Language Modeling*, `arXiv:1312.3005 [cs.CL]
    <http://arxiv.org/abs/1312.3005>`.

    Parameters
    ----------
    which_set : 'training' or 'heldout'
        Which dataset to load.
    which_partitions : list of ints
        For the training set, valid values must lie in [1, 99]. For the
        heldout set they must be in [0, 49].
    vocabulary : dict
        A dictionary mapping tokens to integers. This dictionary is
        expected to contain the tokens ``<S>``, ``</S>`` and ``<UNK>``,
        representing "start of sentence", "end of sentence", and
        "out-of-vocabulary" (OoV). The latter will be used whenever a token
        cannot be found in the vocabulary.
    preprocess : function, optional
        A function that takes a string (a sentence including new line) as
        an input and returns a modified string. A useful function to pass
        could be ``str.lower``.

    """
    default_scheme = ConstantScheme(1)
    sources = ('features',)

    def __init__(self, which_set, which_partitions, vocabulary,
                 preprocess=None):
        update_instance(self, locals())

    def open(self):
        class State(object):
            pass
        state = State()
        state.current_index = 0
        state.file = self._open_file(state.current_index)
        return state

    def _open_file(self, partition_index):
        partition = self.which_partitions[partition_index]
        if self.which_set == 'training':
            data_path = os.path.join(
                config.data_path, '1-billion-word',
                'training-monolingual.tokenized.shuffled',
                'news.en-{:05d}-of-00100'.format(partition))
        else:
            data_path = os.path.join(
                config.data_path, '1-billion-word',
                'heldout-monolingual.tokenized.shuffled',
                'news.en.heldout-{:05d}-of-00050'.format(partition))
        return open(data_path)

    def get_data(self, state=None, request=None):
        data = []
        while len(data) < request:
            sentence = state.file.readline()
            if not sentence:
                state.file.close()
                if state.current_index == len(self.which_partitions) - 1:
                    if not data:
                        raise StopIteration
                    else:
                        break
                else:
                    state.current_index += 1
                    state.file = self._open_file(state.current_index)
            else:
                data.append(
                    [self.vocabulary['<S>']] +
                    [self.vocabulary.get(word, self.vocabulary['<UNK>'])
                     for word in sentence.split()] + [self.vocabulary['</S>']])
        return (data,)
