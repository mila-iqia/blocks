from blocks.datasets.mnist import MNIST


def test_mnist():
    mnist_train = MNIST('train', start=20000)
    assert len(mnist_train.X) == 40000
    assert len(mnist_train.y) == 40000
    mnist_test = MNIST('test', sources=('targets',))
    assert len(mnist_test.X) == 10000
    assert len(mnist_test.y) == 10000

    first_feature, first_target = mnist_train.get_data(request=[0])
    assert first_feature.shape == (1, 784)
    assert first_target.shape == (1, 1)

    first_target, = mnist_test.get_data(request=[0, 1])
    assert first_target.shape == (2, 1)
