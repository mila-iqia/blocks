import dill
from numpy.testing import assert_raises

from blocks.datasets.text import TextFile
from tests import temporary_files


@temporary_files('sentences1.txt', 'sentences2.txt', 'text_stream.pkl')
def test_text():
    with open('sentences1.txt', 'w') as f:
        f.write("This is a sentence\n")
        f.write("This another one")
    with open('sentences2.txt', 'w') as f:
        f.write("More sentences\n")
        f.write("The last one")
    dictionary = {'<UNK>': 0, '</S>': 1, 'this': 2, 'a': 3, 'one': 4}
    text_data = TextFile(files=['sentences1.txt', 'sentences2.txt'],
                         dictionary=dictionary, bos_token=None,
                         preprocess=str.lower)
    stream = text_data.get_default_stream()
    epoch = stream.get_epoch_iterator()
    assert len(list(epoch)) == 4
    epoch = stream.get_epoch_iterator()
    for sentence in zip(range(3), epoch):
        pass
    with open('text_stream.pkl', 'wb') as f:
        dill.dump(epoch, f, fmode=dill.CONTENTS_FMODE)
    sentence = next(epoch)
    with open('text_stream.pkl', 'rb') as f:
        epoch = dill.load(f)
    assert next(epoch) == sentence
    assert_raises(StopIteration, next, epoch)
