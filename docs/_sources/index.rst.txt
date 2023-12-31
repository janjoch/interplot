.. interplot documentation master file, created by
   sphinx-quickstart on Fri Dec 29 11:00:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. mdinclude:: ../README.md


Example
-------

.. code-block:: python

   >>> @interplot.magic_plot
   ... def plot_lines(samples=100, n=10, label="sigma={0}, mu={1}", fig=None):
   ...     """
   ...     Plot Gaussian noise.
   ...
   ...     The function must accept the `fig` parameter from the decorator.
   ...     """
   ...     for i in range(1, n+1):
   ...         fig.add_line(
   ...             np.random.normal(i*10,i,samples),
   ...             label=label.format(i, i*10),
   ...         )


.. code-block:: python

   >>> plot_lines(samples=200, title="Normally distributed Noise")

.. raw:: html
     :file: ../source/plot_examples/gauss_plot_pty.html

.. code-block:: python

   >>> plot_lines(
   ...     samples=200, interactive=False, title="Normally distributed Noise")

.. image:: plot_examples/gauss_plot_mpl.png
    :alt: [matplotlib plot "Normally distributed Noise]


Documentation and API reference
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
    :maxdepth: 2
    :caption: interplot

    api_plot

.. toctree::
    :maxdepth: 2
    :caption: zip iteration helper functions

    api_iter


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
