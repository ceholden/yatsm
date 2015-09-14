.. _guide_batch_interface:

=================
Batch Interface
=================

The default method for running image stacks is to run each line, or row,
separately from other lines. In a multiprocessing situation, the total
number of lines can be broken up among the ``n`` available CPUs. Before
using the batch interface, make sure you already have a parameter file
generated as described by :ref:`guide_model_config`.

The batch interface which runs each line separately is
:ref:`yatsm line <yatsm_line>`. It's usage is:

.. command-output:: yatsm line -h

Let's say our image stack contains 1,000 rows. If we use 50 total CPUs
to process the image stack, then each CPU will be responsible for only
20 lines. To evenly distribute the number of pixels that contain
timeseries (e.g., to ignore any NODATA buffers around the images), the
lines are divided up in sequence. Thus, job 5 of 50 total jobs would
work on the lines:

.. code:: sh

    $ job=5
    $ n=50
    $ seq -s , $job $n 1000
    5,55,105,155,205,255,305,355,405,455,505,555,605,655,705,755,805,855,905,955

Sun Grid Engine
---------------

In the example of the compute cluster at Boston University which
utilizes the Sun Grid Engine scheduler, one could run an image stack as
follows:

.. code:: sh

    $ njob=200
    $ for job in $(seq 1 $njob); do
        qsub -j y -V -l h_rt=24:00:00 -N yatsm_$job -b y \
            $(which python) -u yatsm line --resume -v config.ini $job $njob
      done

One useful tip is to optimize the use of the CPU nodes by first transforming the
dataset from an image based format to a timeseries format by saving all
observations for each row in a separate file. The transformed dataset will be
much easier to read as a timeseries because each processor only needs to read
in one file instead of finding, opening, seeking through, and reading from
many individual image files.

To accomplish this approach, the ``--do-not-run`` flag can be combined with a
specific request for computer nodes with fast ethernet speeds,
``-l eth_speed 10``:

.. code:: sh

    $ njob=16
    $ for job in $(seq 1 $njob); do
        qsub -j y -V -l h_rt=24:00:00 -l eth_speed=10 -N yatsm_$job -b y \
            $(which python) -u yatsm line --resume --do-not-run -v config.ini $job $njob
      done


.. |yatsm_line| replace:: ``yatsm line <yatsm_line_>``
