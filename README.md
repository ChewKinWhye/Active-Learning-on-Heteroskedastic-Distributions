# Active Learning on Heteroskedastic Distributions
An implementation for the paper **Catastrophic Failures of Neural Active Learning on Heteroskedastic Distributions**. The code was built by modifying [Jordan Ash's code for BADGE](https://github.com/JordanAsh/badge).

## Dependencies
To run this code, you'll need PyTorch (we're using version 1.4.0) and scikit-learn. We've been running our code in Python 3.6.8.

# Running an experiment
`python run.py --alg lhd --model resnet --nQuery 1000 --data CIFAR10 --mult 4 --mode class`\
runs an experiment using ResNet18 and *Noisy-Class* CIFAR10, querying batches of 1000 data points using the LHD algorithm. The number of noisy examples is 4x the number of clean examples.

`python run.py --alg badge --model mlp --nQuery 1000 --data SVHN --mult 4 --mode diverse`\
runs an experiment using MLP and *Noisy-Diverse* SVHN, querying batches of 1000 data points using the BADGE algorithm.
