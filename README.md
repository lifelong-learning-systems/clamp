## Continual Learning -- Surrogate Model of Performance

CLAMP - Continual Learning Analysis via a Model of Performance

Creating artificial intelligence (AI) systems capable of demonstrating lifelong learning is a fundamental challenge, and many approaches and metrics have been proposed to analyze algorithmic properties. However, for existing lifelong learning metrics, algorithmic contributions are confounded by task and scenario structure. To mitigate this confounding, we introduce an algorithm-agnostic explainable surrogate-modeling approach to estimate latent properties of lifelong learning algorithms. We validate the approach for estimating these properties via experiments on synthetic data.  To validate the structure of the surrogate model, we analyze real performance data from a collection of popular lifelong learning approaches and baselines adapted for lifelong classification and lifelong reinforcement learning.

### Installation
We tested with Python 3.9 in a conda environment

conda create -n clamp python=3.9

pip install -e clamp/

### Requirements
torch >=1.10.1
matplotlib
seaborn
numpy
ipykernel
pandas
pyyaml
tqdm


### Reproduce experiments and manuscript figures

notebooks/clamp.ipynb

### Acknowledgments
This effort  was funded by the DARPA Lifelong Learning Machines (L2M) Program.
The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.



