import theano
from theano import tensor
from blocks.bricks import MLP, Rectifier
from blocks.bricks.cost import SquaredError
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx
from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks import config
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from theano.sandbox.rng_mrg import MRG_RandomStreams

x = tensor.matrix('features')
mnist_train = MNIST('train', sources=('features',), stop=50000)
mnist_valid = MNIST('train', sources=('features',), start=50000)

mlp = MLP(dims=[784, 100, 784], activations=[Rectifier(), None])
y_hat = mlp.apply(x)
cost = SquaredError().apply(y_hat, x)
cost.name = 'cost'
cg = ComputationGraph(cost)

mlp.weights_init = IsotropicGaussian(0.1)
mlp.biases_init = Constant(0)
mlp.initialize()

# This is an approximation of adding spherical noise
def apply_approx_noise(computation_graph, variables, level, seed=None):
    if not seed:
        seed = config.default_seed
    rng = MRG_RandomStreams(seed)
    replace = {}
    for variable in variables:
        noise = rng.normal(variable.shape)
        noise *= level / noise.norm(2)
        replace[variable] = variable + noise
    return computation_graph.replace(replace)

def apply_noise(computation_graph, variables, level, seed=None):
    if not seed:
        seed = config.default_seed
    rng = MRG_RandomStreams(seed)
    sizes = [var.get_value(borrow=True).size
             for var in variables]
    noise = rng.normal((sum(sizes),))
    noise *= level / noise.norm(2)
    replace = {}
    total_size = 0
    for size, variable in zip(sizes, variables):
        noise_part = noise[total_size:total_size + size].reshape(variable.shape)
        replace[variable] = variable + noise_part
        total_size += size
    return computation_graph.replace(replace)

noise_level = shared_floatx(5)
new_cg = apply_noise(cg, cg.parameters, noise_level)
cost, = new_cg.outputs

algorithm = GradientDescent(cost=cost, params=cg.parameters,
                            step_rule=Scale(learning_rate=0.01))

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream(
        mnist_train,
        iteration_scheme=ShuffledScheme(mnist_train.num_examples, 128)
    ),
    extensions=[
        DataStreamMonitoring(
            [cost], DataStream(
                mnist_valid,
                iteration_scheme=SequentialScheme(mnist_valid.num_examples,
                                                  256)
            )
        ),
        Printing(after_epoch=True)
    ],
)
main_loop.run()


# def multi_noise(variable, level, n, seed=None):
#     if not seed:
#         seed = config.default_seed
#     rng = MRG_RandomStreams(seed)
#     noise = rng.normal([n] + [variable.shape[i] for i in range(variable.ndim)])
#     noise /= noise.norm(2) * level
#     new_variable = variable.dimshuffle(*['x'] + range(variable.ndim))
#     return new_variable + noise
# 
# noise_level = theano.shared(1.)
# n = 10
# a = multi_noise(mlp.linear_transformations[0].W, noise_level, 10).dimshuffle(0, 2, 1).dot(x.T).dimshuffle(0, 2, 1)
# h = Rectifier().apply(a)
# y_hat = tensor.batched_dot(h, multi_noise(mlp.linear_transformations[1].W, noise_level, 10))
# 
