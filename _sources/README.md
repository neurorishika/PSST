# PSST . . . It's well Documented!

[![arXiv](https://img.shields.io/badge/arXiv-1906.03958-b31b1b.svg)](https://arxiv.org/abs/1906.03958)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![tensorflow](https://raw.githubusercontent.com/aleen42/badges/master/src/tensorflow_flat_square_dfc.svg)](https://www.tensorflow.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neurorishika/PSST/HEAD)
[![GitHub license](https://img.shields.io/github/license/neurorishika/PSST.svg)](https://github.com/neurorishika/PSST/blob/master/LICENSE)
[![GitHub forks](https://img.shields.io/github/forks/neurorishika/PSST.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/neurorishika/PSST/network/)
[![GitHub stars](https://img.shields.io/github/stars/neurorishika/PSST.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/neurorishika/PSST/stargazers/)

Welcome to **P**arallelised **S**calable **S**imulations in **T**ensorFlow, A  tutorial series about... well... making highly parallelised differential equation based simulations that are scalable across multiple platforms using Python, an easily accessible general purpose programming language, and the power of Google's open source package TensorFlow.

![PSST](https://raw.githubusercontent.com/technosap/PSST/master/Book/PSST.png)

### About PSST
Neuronal networks are often modeled as systems of coupled, nonlinear, ordinary or partial differential equations. The number of differential equations used to model a network increases with the size of the network and the level of detail used to model individual neurons and synapses. As one scales up the size of the simulation it becomes important to use powerful computing platforms. Many tools exist that solve these equations numerically. However, these tools are often platform-specific. There is a high barrier of entry to developing flexible general purpose code that is platform independent and supports hardware acceleration on modern computing architectures such as GPUs/TPUs and Distributed Platforms. TensorFlow is a Python-based open-source package initially designed for machine learning algorithms, but it presents a scalable environment for a variety of computations including solving differential equations using iterative algorithms such as Euler and Runge-Kutta methods. We have organized as a series of tutorials that demonstrate how to harness the power of TensorFlow’s data-flow programming paradigm to solve differential equations. 

Our tutorial is a simple exposition of numerical methods to solve ordinary differential equations using Python and TensorFlow. It consists of a series of Python notebooks that accompany the proposed manuscript. Over the course of six sessions, we lead novice programmers from integrating simple 1-dimensional differential equations using Python, to solving a large system (1000’s of differential equations) of coupled conductance-based neurons using a highly parallelised and scalable framework that uses Python and TensorFlow. 


### Tutorial Content
| Tutorial Day | Run | Run | View only (no execution) |
| ------------------------------------------ | --- | --- | ---- |
| Day 1: Of Numerical Integration and Python | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Day%201%20Of%20Numerical%20Integration%20and%20Python/Day%201.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Day%201%20Of%20Numerical%20Integration%20and%20Python/Day%201.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Day%201%20Of%20Numerical%20Integration%20and%20Python/Day%201.ipynb?flush_cache=true) |
| Day 2: Let the Tensors Flow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Day%202%20Let%20the%20Tensors%20Flow/Day%202.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Day%202%20Let%20the%20Tensors%20Flow/Day%202.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Day%202%20Let%20the%20Tensors%20Flow/Day%202.ipynb?flush_cache=true) |
| Day 3: Cells in Silicon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Day%203%20Cells%20in%20Silicon/Day%203.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Day%203%20Cells%20in%20Silicon/Day%203.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Day%203%20Cells%20in%20Silicon/Day%203.ipynb?flush_cache=true) |
| Day 4: Neurons and Networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Day%204%20Neurons%20and%20Networks/Day%204.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Day%204%20Neurons%20and%20Networks/Day%204.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Day%204%20Neurons%20and%20Networks/Day%204.ipynb?flush_cache=true) |
| Day 5: Optimal Mind Control | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Day%205%20Optimal%20Mind%20Control/Day%205.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Day%205%20Optimal%20Mind%20Control/Day%205.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Day%205%20Optimal%20Mind%20Control/Day%205.ipynb?flush_cache=true) |
| Day 6: Example Implementation (Locust AL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Example%20Implementation%20Locust%20AL/Example.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Example%20Implementation%20Locust%20AL/Example.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Example%20Implementation%20Locust%20AL/Example.ipynb?flush_cache=true) |
| Day 7: (Optional) Distributed Tensorflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neurorishika/PSST/blob/master/Tutorial/Optional%20Material/Distributed%20TensorFlow/Distributed%20TensorFlow.ipynb) | [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/neurorishika/PSST/master/Tutorial/Optional%20Material/Distributed%20TensorFlow/Distributed%20TensorFlow.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neurorishika/PSST/blob/master/Tutorial/Optional%20Material/Distributed%20TensorFlow/Distributed%20TensorFlow.ipynb?flush_cache=true) |

*Pre-rendered HTML files are available in the 'static' folder on each day*

**WARNING: If you are running PSST using Kaggle, make sure you have logged in to your verified Kaggle account and enabled Internet Access for the kernel. If you do not do so, the code will give errors from Day 5 onwards. For instructions on enabling Internet on Kaggle Kernels, visit: https://www.kaggle.com/product-feedback/63544**

### Requirements
- Python 3.9 or above
- Jupyter Notebook
- NumPy 1.20 or above
- MatPlotLib 3.4.3 or above
- Seaborn 0.11.2 or above
- TensorFlow CPU/GPU (2.8 or above)

Happy Coding!

Cheers,
Rishika Mohanta and Collins Assisi
