# BatchOptimizedTraining

Miriam Deml - Thesis: Batch-Optimized Training for Neural Networks - May 28, 2020

Except for the dependencies, all necessary code is provided in the folder "files" to replicate the experiments of the thesis. 
There are Jupyter Notebooks for the chapters 3, 4, and 5 which lead through the commands for all experiments that were done within the thesis.
Furthermore the Jupyter Notebook VisualizationGraph.ipynb helps to visualize the results by plotting the results from the log-files.

The experiments were done with PyTorch and CUDA (version 10). All code that includes the training of NNs is therefore implemented to be executed using a GPU.

The code that was (partly) copied from other researchers is marked by a comment leading to the Github repositiories. In general, code from two other teams was used:
1. Ghost Batch Normalization and implementation of models, pre-processing, etc.: Elad Hoffer, Itay Hubara, and Daniel Soudry. "Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks." Advances in Neural Information Processing Systems. 2017. https://github.com/eladhoffer/bigBatch
2. Visualization of the loss landscape and implementation of models: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. NIPS, 2018. https://github.com/tomgoldstein/loss-landscape


Dependencies:
- pytorch
- torchvision
- openmpi 3.1.2
- mpi4py 2.0.0
- numpy 1.15.1
- h5py 2.7.0
- matplotlib 2.0.2
- scipy 0.19



