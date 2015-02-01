Datasets
========

The data pipeline is an important part of training neural networks. Blocks
provides an abstraction to datasets which is complicated at first sight, but
can be very powerful.

.. digraph:: datasets
   :caption: A simplified overview of the interactions between the different parts of the data-handling classes in Blocks. Dashed lines are optional.

   Dataset -> DataStream [label=" Argument to"];
   DataStream -> Dataset [label=" Gets data from"];
   DataStream -> DataIterator [label=" Returns"];
   IterationScheme -> DataStream [style=dashed, label=" Argument to"];
   DataStream -> IterationScheme [style=dashed, label=" Gets request iterator"];
   IterationScheme -> RequestIterator [label=" Returns"];
   RequestIterator -> DataIterator [style=dashed, label=" Argument to"];
   DataIterator -> DataStream [label=" Gets data from"];
   DataStream -> DataStream [style=dashed, label=" Gets data from (wrapper)"];
   { rank=same; RequestIterator, DataIterator }

Datasets
  Datasets provide an interface to the data we are trying to acces. This data
  is usually stored on disk, but can also be created on the fly (e.g. drawn
  from a distribution), requested from a database or server, etc. Datasets are
  largely *stateless*. Multiple data streams can be iterating over the same
  dataset simultaneously, so the dataset couldn't have a single state to store
  e.g. its location in a file. Instead, the dataset provides a set of methods
  (:meth:`~.datasets.Dataset.open`, :meth:`~.datasets.Dataset.close`,
  :meth:`~.datasets.Dataset.get_data`, etc.) that interact with a particular
  state, which is managed by a data stream.

Data stream
  A data stream uses the interface of a dataset to e.g. iterate over the data.
  Data streams can produce data set iterators (epoch iterators) which will use
  the data stream's state to return actual data. Data streams can optionally
  use an iteration scheme to describe in what way (e.g. in what order) they
  will request data from the dataset.

Data stream wrapper
  A data stream wrapper is really just another data stream, except that it
  doesn't take a data set but another data stream (or wrapped data stream) as
  its input. This allows us to set up a data processing pipeline, which can be
  quite powerful. For example, given a data set that produces sentences from a
  text corpus, we could use a chain of data stream wrappers to read groups of
  sentences into a cache, sort them by length, group them into minibatches, and
  pad them to be of the same length.

Iteration scheme
  A iteration scheme describes *how* we should proceed to iterate over the
  data. Iteration schemes will normally describe a sequence of batch sizes
  (e.g.  a constant minibatch size), or a sequence of indices to our data (e.g.
  indices of shuffled minibatches). Iteration schemes return request iterators.

Request iterator
  A request iterator implements the Python iteration protocol. It represents a
  single epoch of requests, as determined by the iteration scheme that produced
  it.

Data iterator
  A data iterator also implements the Python iteration protocol. It optionally
  uses a request iterator and returns data at each step (requesting it from the
  data stream). A single iteration over a data iterator represents a single
  epoch.
