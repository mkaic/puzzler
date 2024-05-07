
# The idea
Model inputs: a grid of pixel values sampled from an image, the location and scale of this grid, and a hidden state vector.
Model outputs: a new location and scale, and a hidden state vector.

Each iteration, the model gets some information about the image at a certain location and produces a hidden state vector encoding what it learned. This vector is passed to a classifier which tries to classify the image based on the hidden state. At first this will not work well. The hope is that by running the same model recurrently on its own outputs over and over and including intermediate losses, the model will learn to iteratively "ask questions" about the image until it can classify it appropriately. The hidden state vector will have useful information because we backprop through multiple iterations. The hidden state vector is initialized to all zeros, the location to (0.5,0.5), and the scale at whichever value makes it fill the image. The hidden state vector is updated residually for gradient stability.

# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```

# Repo structure
Implementations are in `src`, training script is in `scripts`, and sanity-checks I wrote while implementing stuff are in `tests`. `experiments.md` is a log where I track the results of different design choices and hyperparams. The training script expects CIFAR-100 to be in a folder called `data`, which is included in `.gitignore` so I don't accidentally attempt to push the dataset.