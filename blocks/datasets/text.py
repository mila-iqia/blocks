import os

from blocks import config
from blocks.datasets import Dataset


class TextFileState(object):
    pass


class TextFile(Dataset):
    r"""Reads text files and numberizes them given a dictionary.

    Parameters
    ----------
    files : list of str
        The names of the files in order which they should be read. Each
        file is expected to have a sentence per line.
    dictionary : str or dict
        Either the path to a Pickled dictionary mapping tokens to integers,
        or the dictionary itself. At the very least this dictionary must
        map the unknown word-token to an integer.
    bos_token : str or None, optional
        The beginning-of-sentence (BOS) token in the dictionary that
        denotes the beginning of a sentence. Is ``<S>`` by default. If
        passed ``None`` no beginning of sentence markers will be added.
    eos_token : str or None, optional
        The end-of-sentence (EOS) token is ``</S>`` by default, see
        ``bos_taken``.
    unk_token : str, optional
        The token in the dictionary to fall back on when a token could not
        be found in the dictionary.
    level : 'word' or 'character', optional
        If 'word' the dictionary is expected to contain full words. The
        sentences in the text file will be split at the spaces, and each
        word replaced with its number as given by the dictionary, resulting
        in each example being a single list of numbers. If 'character' the
        dictionary is expected to contain single letters as keys. A single
        example will be a list of lists, each sublist representing a word.
    preprocess : function, optional
        A function which takes a sentence (string) as an input and returns
        a modified string. For example ``str.lower`` in order to lowercase
        the sentence before numberizing.

    Examples
    --------
    >>> with open('sentences.txt', 'w') as f:
    ...     f.write("This is a sentence\n")
    ...     f.write("This another one")
    >>> dictionary = {'<UNK>': 0, '</S>': 1, 'this': 2, 'a': 3, 'one': 4}
    >>> text_data = TextFile(files=['sentences.txt'],
    ...                      dictionary=dictionary, bos_token=None,
    ...                      preprocess=str.lower)
    >>> for data in text_data.get_default_stream().get_epoch_iterator():
    ...     print(data)
    ([2, 0, 3, 0, 1],)
    ([2, 0, 4, 1],)

    .. doctest::
       :hide:

       >>> import os
       >>> os.remove('sentences.txt')

    """
    provides_sources = ('features',)
    default_scheme = None

    def __init__(self, files, dictionary, bos_token='<S>', eos_token='</S>',
                 unk_token='<UNK>', level='word', preprocess=None):
        self.files = files
        self.dictionary = dictionary
        if bos_token is not None and bos_token not in dictionary:
            raise ValueError
        self.bos_token = bos_token
        if eos_token is not None and eos_token not in dictionary:
            raise ValueError
        self.eos_token = eos_token
        if unk_token not in dictionary:
            raise ValueError
        self.unk_token = unk_token
        if level not in ('word', 'character'):
            raise ValueError
        self.level = level
        self.preprocess = preprocess

    def open(self):
        state = TextFileState()
        state.current_index = 0
        state.file = self._open_file(state.current_index)
        return state

    def _open_file(self, partition_index):
        return open(self.files[partition_index])

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        while True:
            sentence = state.file.readline()
            if not sentence:
                state.file.close()
                if state.current_index == len(self.files) - 1:
                    raise StopIteration
                else:
                    state.current_index += 1
                    state.file = self._open_file(state.current_index)
            else:
                break
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data += [self.dictionary.get(word, self.dictionary[self.unk_token])
                     for word in sentence.split()]
        else:
            data += [self.dictionary.get(char, self.dictionary[self.unk_token])
                     for char in sentence.strip()]
        data += [self.dictionary[self.eos_token]] if self.eos_token else []
        return (data,)


class OneBillionWord(TextFile):
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

    See :class:`TextFile` for remaining keyword arguments.

    """
    def __init__(self, which_set, which_partitions, dictionary, **kwargs):
        if which_set not in ('training', 'heldout'):
            raise ValueError
        if which_set == 'training':
            if not all(partition in range(1, 100)
                       for partition in which_partitions):
                raise ValueError
            files = [os.path.join(
                config.data_path, '1-billion-word',
                'training-monolingual.tokenized.shuffled',
                'news.en-{:05d}-of-00100'.format(partition))
                for partition in which_partitions]
        else:
            if not all(partition in range(50)
                       for partition in which_partitions):
                raise ValueError
            files = [os.path.join(
                config.data_path, '1-billion-word',
                'heldout-monolingual.tokenized.shuffled',
                'news.en.heldout-{:05d}-of-00050'.format(partition))
                for partition in which_partitions]
        super(OneBillionWord, self).__init__(files, dictionary, **kwargs)
