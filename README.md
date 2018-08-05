# Andrea Valenti's Echo State Network implementation

This project provides a MATLAB implementation of an [Echo State Network](http://www.scholarpedia.org/article/Echo_state_network) (ESN for short) neural network. This implementation uses an addition unsupervised learning rule (called Intrinsic Plasticity, IP for short) to pre-train the reservoir's weights. The IP rule is insipred by [the homonym behaviour](http://www.scholarpedia.org/article/Intrinsic_plasticity) observed in biological neurons.

## Features

This project
- Implements the ESN model.
- Implements the IP learning rule.
- Applies the ESN to four different tasks: *Memory Capacity*, *30th order NARMA system*, *Mackey-Glass equation* and *Laser Dataset*. You can find an accurate description of these tasks in the reference paper *Improving reservoirs using intrinsic plasticity* by Benjamin Schrauwen et al. (which is provided in the "references" subfolder of this project).

## Getting Started

- File *narma.m* contains a helper function used to generate the dataset for the *30th order NARMA system* task.
- File *memory_capacity.m* contains a helper function used to compute the memory capacity of the model for the *Memory Capacity* task. 
- Files *echo_state_network.m* *esn_train.m* *esn_predict.m* *esn_score.m* *esn_states.m* and *esn_train_ip.m* implement the various parts of the ESN.
- Files *esn_laser.m* *esn_mg.m* *esn_narma.m* *esn_mc.m* apply the ESN model to, respectively, the *Laser Dataset*, *Mackey-Glass equation*, *30th order NARMA system* and *Memory Capacity* tasks. You should regard these files as the "main" scripts for their respective tasks.

**NB**: all datasets are provided, if needed, in the "dataset" subfolder.

### Installation/Dependencies

You just need a MATLAB installation. Version R2016b or later ones are fine (however it probably works with previous versions as well).

### Usage

You are ancouraged to try different hyperparameters combinations. The most important parameters are:
- Number of reservoir's units.
- Spectral radius of reservoir's weight matrix.
- Input scaling.
- Connectivity rate of reservoir's units.
- Regularization rate.

For the IP rule, you have:
- Learning rate.
- Mean and variance of target Normal distribution.

You also have other, arguably less important, parameters such as:
- Length of the initial transient that will be discarded during training (ntransient).
- Number of epochs of IP pretraining (nepoch).
- Number of ESN trained at the same time, whose output is averaged to produce the final output. This is needed to reduce the variance introduced by the reservoir random initialization (esn_pool).

## Getting Help

The "refernces" subfolder contains a few paper that are useful for this project. In particular:

- *Improving reservoirs using intrinsic plasticity* by Benjamin Schrauwen et al. is the foundational paper that introduces the idea of using the IP rule to pretrain the reservoir. This project is based on this paper, so I recommend to read it and to understand at least the general idea behind, before delving into the code.

- *A Practical Guide to Applying Echo State Networks* by Mantas Lukoševičius contains many useful insights for training ESN in practice. This paper dates back to 2012, so some of the suggestions might be a little outdated and to be taken with a grain of salt, but it still remains a very good starting point.

For any other additional information, you can email me at valentiandrea@rocketmail.com.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.