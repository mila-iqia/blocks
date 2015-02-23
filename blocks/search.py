from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from six.moves import range

import numpy

from theano import config
from theano import function
from theano import tensor

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

floatX = config.floatX


class Search(object):
    """Abstract search class

    Parameters
    ----------
    sequence_generator : sequence generator
        Sequence generator to be used

    """
    __metaclass__ = ABCMeta

    def __init__(self, sequence_generator):
        self.generator = sequence_generator
        self.compiled = False

    @abstractmethod
    def compile(self):
        self.compiled = True

    @abstractmethod
    def search(self, **kwargs):
        r"""Performs search

        Parameters
        ----------
        \*\*kwargs :
            Arguments needed by sequence generator

        Returns
        -------
        Generated sequences
        """
        if not self.compiled:
            self.compile()


class GreedySearch(Search):

    def __init__(self, sequence_generator):
        super(GreedySearch, self).__init__(sequence_generator)


class BeamSearch(Search):
    """Beam search.

    Parameters
    ----------
    beam_size : int
        Size of beam.
    batch_size : int
        Size of input batch.
    sequence_generator : sequence generator
        Sequence generator brick.
    attended : theano variable
        Theano variable for attended in sequence generator.
    attended_mask : theano variable
        Theano variable for attended mask in sequence generator.
    inputs_dict : dict
        Dictionary of inputs {name: theano variable for input}. The
        functions will be constructed with these inputs.

    """
    def __init__(self, beam_size, batch_size, sequence_generator, attended,
                 attended_mask, inputs_dict):
        super(BeamSearch, self).__init__(sequence_generator)
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.sequence_generator = sequence_generator
        self.attended = attended
        self.attended_mask = attended_mask
        self.inputs_dict = inputs_dict
        self.generate_names = sequence_generator.generate_outputs()
        self.init_computer = None
        self.next_computer = None
        self.real_batch_size = batch_size * beam_size

    def compile(self, *args, **kwargs):
        """Compiles functions for beam search.

        """
        generator = self.sequence_generator
        attended = self.attended
        attended_mask = self.attended_mask
        # Construct initial values
        init_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False, n_steps=1,
                                            batch_size=self.real_batch_size,
                                            return_dict=True)
        init_cg = ComputationGraph(init_generated.values())
        init_probs = VariableFilter(
            application=generator.readout.emitter.probs,
            name='output')(init_cg.variables)[-1]

        # Create theano function for initial values
        self.init_computer = function(self.inputs_dict.values(),
                                      init_generated.values() + [init_probs])

        # Define inputs for next values computer
        cur_variables = OrderedDict()
        for name, value in init_generated.iteritems():
            cur_value = tensor.zeros_like(value)
            cur_value.name = name
            cur_variables[name] = cur_value

        input_names = (generator.state_names + generator.glimpse_names +
                       ['outputs'])
        generator_inputs = {name: val for name, val
                            in cur_variables.iteritems()
                            if name in input_names}
        next_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False,
                                            n_steps=1,
                                            batch_size=self.real_batch_size,
                                            return_dict=True,
                                            **generator_inputs)
        cg_step = ComputationGraph(next_generated.values())
        next_probs = VariableFilter(
            application=generator.readout.emitter.probs,
            name='output')(cg_step.variables)[-1]
        # Create theano function for next values
        self.next_computer = function(self.inputs_dict.values() +
                                      [cur_variables['states']],
                                      next_generated.values() + [next_probs])
        super(BeamSearch, self).compile(*args, **kwargs)

    def compute_next(self, inputs_dict, cur_vals=None):
        """Computes next states, glimpses, outputs, and probabilities.

        Parameters
        ----------
        inputs_dict : dict
            Dictionary of inputs divided by chunks.
        cur_vals : dict
            Dictionary of current states, glimpses, and outputs.

        Returns
        -------
        Dictionary of next state, glimpe, output, probabilities values
        with names as returned by `generate_outputs`.

        """
        inputs = [self.merge_chunks(input) for input
                  in inputs_dict.itervalues()]
        if cur_vals is None:
            next_values = self.init_computer(*inputs)
        else:
            states = self.merge_chunks(cur_vals['states'])
            next_values = self.next_computer(*(inputs + [states[0]]))
        next_values = [self.divide_by_chunks(value.reshape((1,) + value.shape))
                       for value in next_values]
        return OrderedDict(zip(self.generate_names + ['probs'], next_values))

    @classmethod
    def _top_probs(cls, probs, beam_size, unique=False):
        """Returns indexes of elements with highest probabilities.

        Parameters
        ----------
        probs : numpy array
            A 3d array of probabilities (length of sequence, batch,
            readout_dim)
        beam_size : int
            Beam size, number of top probs to return

        Returns
        -------
        Tuple of (indexes, top probabilities)

        """
        flatten = probs.flatten()
        if unique:
            args = numpy.unique(numpy.argpartition(-flatten, beam_size))[:beam_size]
        else:
            args = numpy.argpartition(-flatten, beam_size)[:beam_size]
        args = args[numpy.argsort(-flatten[args])]
        if unique:
            # append best if needed
            if args.shape[0] < beam_size:
                args = numpy.append(args,
                                 numpy.tile(args[0], beam_size - args.shape[0]))
        # convert args back
        indexes = numpy.unravel_index(args, probs.shape[1:])
        return indexes, probs[0][indexes]

    def _rearrange(self, outputs, indexes):
        new_outputs = self.merge_chunks(outputs)
        new_outputs = new_outputs[:, indexes.flatten()]
        new_outputs = self.divide_by_chunks(new_outputs)
        return new_outputs.copy()

    def merge_chunks(self, array):
        """Merges chunks

        Parameters
        ----------
        array : numpy array
            3D or 4D (sequence length, beam size, batch size [, readout dim])
            array

        Returns
        -------
        2D or 3D (sequence length, beam size * batch size [, readout_dim])
        array

        """
        if len(array.shape) == 3:
            trans = (1, 2, 0)
            trans_back = (1, 0)
            out_shape = (self.beam_size * self.batch_size, array.shape[0])
        else:
            trans = (1, 2, 0, 3)
            trans_back = (1, 0, 2)
            out_shape = (self.beam_size * self.batch_size, array.shape[0],
                         array.shape[-1])
        array = array.transpose(trans)
        array = array.reshape(out_shape)
        return array.transpose(trans_back)

    def divide_by_chunks(self, array):
        """Divides input to chunks

        Parameters
        ----------
        array : numpy array
            2D or 3D (sequence length, beam size * batch size
            [, readout dim]) aray

        Returns
        -------
        3D or 4D (sequence length, beam size, batch size [, readout dim])
        array

        """
        if len(array.shape) == 2:
            first_transpose = (1, 0)
            reshape = (self.beam_size, self.batch_size, array.shape[0])
            back_transpose = (2, 0, 1)
        else:
            first_transpose = (1, 0, 2)
            reshape = (self.beam_size, self.batch_size, array.shape[0],
                       array.shape[2])
            back_transpose = (2, 0, 1, 3)
        reshaped = array.transpose(first_transpose)
        reshaped = reshaped.reshape(reshape)
        return reshaped.transpose(back_transpose)

    def search(self, inputs_val_dict, eol_symbol=-1, max_length=512, **kwargs):
        """Performs beam search.

        Parameters
        ----------
        inputs_val_dict : dict
            Dictionary of input values {name: value}. Input values may
            be in a batch of size greater than 1, in this case, several
            beam searches will be performed in parallel.
        eol_symbol : int
            End of sequence symbol, the search stops when the
            symbol is generated.
        max_length : int
            Maximum sequence length, the search stops when it
            is reached.

        Returns
        -------
        Most probable sequences, corresponding probabilities and costs

        """
        super(BeamSearch, self).search(**kwargs)
        # Inputs repeated beam_size times
        aux_inputs = OrderedDict(
            [(name,
              self.divide_by_chunks(numpy.tile(val, (1, self.beam_size))))
             for name, val in inputs_val_dict.iteritems()])

        current_outputs = numpy.zeros((0, self.beam_size, self.batch_size),
                                      dtype='int64')
        curr_out_mask = numpy.ones((0, self.beam_size, self.batch_size),
                                   dtype=floatX)

        cur_values = None
        for i in range(max_length):
            cur_values = self.compute_next(aux_inputs, cur_values)
            next_probs = cur_values['probs']

            # Top probs
            indexes, top_probs = zip(*[self._top_probs(next_probs[:, :, j],
                                                       self.beam_size,
                                                       unique=i == 0)
                                       for j in range(self.batch_size)])
            indexes = numpy.array(indexes)  # chunk, 2, beam
            # current_outputs.
            # here we suppose, that we have 2d outputs
            outputs = indexes[:, 1, :].copy()

            # rearrange outputs
            rearrange_ind = indexes[:, 0, :]
            current_outputs = self._rearrange(current_outputs, rearrange_ind)
            curr_out_mask = self._rearrange(curr_out_mask, rearrange_ind)
            for name in cur_values:
                cur_values[name] = self._rearrange(cur_values[name],
                                                   rearrange_ind)
            for name in aux_inputs:
                aux_inputs[name] = self._rearrange(aux_inputs[name],
                                                   rearrange_ind)

            # construct next output
            outputs = outputs.reshape((1, self.beam_size, self.batch_size))
            current_outputs = numpy.append(current_outputs,
                                           outputs.copy(), axis=0)
            # check if we meet eol
            next_out_mask = numpy.ones((1, self.beam_size, self.batch_size),
                                       dtype=floatX)

            next_out_mask[0, :, :] = (outputs[0, :, :] != eol_symbol)
            curr_out_mask = numpy.append(curr_out_mask, next_out_mask.copy(),
                                         axis=0)

            if numpy.all(current_outputs[-1, :, 0] == eol_symbol):
                break

        # Select only best
        current_outputs = current_outputs[:, :, 0]
        curr_out_mask = curr_out_mask[:, :, 0]

        return current_outputs, curr_out_mask
