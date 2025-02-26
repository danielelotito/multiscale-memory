# Multiscale-memory

A computational model to describe the dynamics of neurons and synapses inspired by Hebb's rule, that is at the core of the Hopfield model.

An associative networks is defined and its units evolve to minimize an energy. In doing this the synapses that connect the neurons learn the supplied patterns. In this way the network can store information that can be retrieved through gradient descent.

The preprint of the paper is available at [add link]

## Model Description

The model consists of N neurons, each connected to all others via synapses. The system's dynamics follow two differential equations:

1. Neuron state evolution:

   ```
   τ d(σ_i)/dt = -σ_i + tanh(β * Σ(j≠i) J_ij * σ_j + β * u * h_i)
   ```

   Here, σ_i represents the state of neuron i, constrained by the hyperbolic tangent to the interval [-1,1]. The parameter β controls the system's temperature.
2. Synaptic strength evolution:

   ```
   τ' d(J_ij)/dt = -J_ij + k * σ_i * σ_j
   ```

   Synaptic weights J_ij evolve over time, with τ' as the synaptic time scale.

The external field h(t) is updated at intervals of τ', modeled as a piecewise constant stochastic process with discrete values drawn from a set of patterns {ξ_μ}_μ=1,...,K, each with probability p_μ.

## Key Features

- Configurable model parameters using Hydra
- Support for multiple simulation runs with different parameter sets
  - After a multirun, we can collect the result obtained to further analyse them and extract information about the variation of relevant quantities with respect to run parameters, or other things that cannot be directly deducted from a single run.
- Static and dynamic visualizations of simulation results
- Flexible model architecture allowing for different types of associative memory models

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/multiscale-memory.git
   cd multiscale-memory
   ```
2. Create a virtual environment and activate it:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

This repo is a research-oriented repository. Although the files are not properly organized in various folders, and I acknowledge that this would improve both the structure and the appearence of the repo, its use is straightforward.

To run a simulation with default parameters:

```
python main.py
```

To run a simulation with a specific configuration:

```
python main.py model=variants/gaussian_patterns
```

To run a multirun simulation, we refer to [Hydra CLI syntax](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/ "Hydra documentation: Sometimes you want to run the same application with multiple different configurations"):

```
python main.py --multirun  simulation.T=5 model.tau_J=50  model.N=10,20  model.K=5,10 model.beta=100  model.k=1  
```

Finally, you can collect the results of the multirun, to further analyse them and produce relevant plots:

```
python collect_and_plot_data.py
```

## Configuration

The project uses Hydra for configuration management. The main configuration file is `config/config.yaml`. You can create custom configuration files following[ Hydra default list](https://hydra.cc/docs/advanced/defaults_list/ "The Defaults List is a list in an input config that instructs Hydra how to build the output config. Each input config can have a Defaults List as a top level element.") management .

For example, the key parameters used in the model initialization, specified in the config files in `config/model/` include:

- N: Number of neurons
- K: Number of patterns
- beta: Inverse temperature
- tau_s: Neural time scale
- tau_J: Synaptic time scale
- u: External field scaling
- k: Synaptic strength scaling

The reason why this script does not require any other argument is that, when running a multirun job, the path folder in which the job is run is appended to the file `multirun_paths.txt` that is automatically created by the `main.py` script. Then, `collect_and_plot_data.py` automatically reads this line.

The configuration file for the `collect_and_plot_data.py` script is `config/config_multirun_collection.yaml`. The first two keywords allow to specify a path, and specify which line of `multirun_paths.txt` to read, respectively. Here an example, that corresponds to suggest default values for these parameters

```yaml
dir: null
path: 0
```

More specifically, in this case the first key tells us to read the path from `multirun_paths.txt`. The second keyword tells us to read the last line. While

```yaml
path: 1
```

would select the penultimate line of `multirun_paths.txt`. In `config/config.yaml` file, under the simulation key you can find two important parameters to regulate how much time the script will be running. The use_transients keyword activate an exact computation of the transient terms in the theoretical calculation of the fluctuation of a measure of the distance of the coupling matrix of our system to the Hebbian reference, differences from the proposed approximated transient are neglectable.

```yaml
simulation:
  T: 4000
  use_transients: false
```

## Visualization

The simulator generates both static and dynamic visualizations:

- Static visualizations: PNG and PDF files including neural activities, synaptic weights, and energy landscapes
- Dynamic visualizations: GIF files showing the evolution of neural states and synaptic weights over time (only for single runs)

## Output

Simulation results and visualizations are saved in timestamped folders under the `outputs` directory. This includes:

- Raw data in CSV format, this combines both
- A LaTeX table that contains the run parameters
- Visualizations

The output of a multirun is saved instead in the `multirun` directory, but in a subdirectory that labels the label, starting from `0`.

Finally, when the result of a multirun is analyzed `collect_and_plot_data.py` is saved in the `collection_outputs` directory.

Depending on the number of runs that this script will analyse the result can be different, ` max_runs_to_process: 40` in `config/config_multirun_collection.yaml` regulates this feature: if there are more runs than the value of this key (40 in this case) the plots that would become to cluttered are disabled.

## Extending the Model

To add new features or modify the model:

**1.** Update the **`model.py`** file to implement new dynamics, another model is already implemented, but it will be studied and discussed in a follow up paper.
**2.** Add new configuration options in **`config/config.yaml`** or create new variant files
**3.** Implement new visualization functions in **`visualization.py`**
**4.** Update **`main.py`** to use the new features

## Broad vision and contribution guidelines

Simple architectures scaled up beyond an architecture-dependent threshold start to display emergent computational behaviours similar to more complex architectures used in state of the art models [add reference to last Benjo].

At the same time, there is a huge interest in architectures that are more tractable from an analytical point of view.

An interesting research line aims at studying some of these architectures with the methods of statistical physics

At the moment I encourage adding functionalities to the project, making use of its modularity. I also encourage cloning the project, and/or suggest more substantial changes directy to me. Contact information are available on my personal website [ https://danielelotito.github.io/dl-codespace/ ].
