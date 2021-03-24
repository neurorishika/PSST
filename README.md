# PSST . . . It's well Documented!
Welcome to **P**arallelised **S**calable **S**imulations in **T**ensorFlow, A tutorial series about... well... making highly parallelised differential equation based simulations that are scalable across multiple platforms using Python, an easily accessible general purpose programming language, and the power of Google's open source package TensorFlow.

![PSST](https://raw.githubusercontent.com/technosap/PSST/master/PSST.png)

### About PSST
Neuronal networks are often modeled as systems of coupled, nonlinear, ordinary or partial differential equations. The number of differential equations used to model a network increases with the size of the network and the level of detail used to model individual neurons and synapses. As one scales up the size of the simulation it becomes important to use powerful computing platforms. Many tools exist that solve these equations numerically. However, these tools are often platform-specific. There is a high barrier of entry to developing flexible general purpose code that is platform independent and supports hardware acceleration on modern computing architectures such as GPUs/TPUs and Distributed Platforms. TensorFlow is a Python-based open-source package initially designed for machine learning algorithms, but it presents a scalable environment for a variety of computations including solving differential equations using iterative algorithms such as Euler and Runge-Kutta methods. We have organized as a series of tutorials that demonstrate how to harness the power of TensorFlow’s data-flow programming paradigm to solve differential equations. 

Our tutorial is a simple exposition of numerical methods to solve ordinary differential equations using Python and TensorFlow. It consists of a series of Python notebooks that accompany the proposed manuscript. Over the course of six sessions, we lead novice programmers from integrating simple 1-dimensional differential equations using Python, to solving a large system (1000’s of differential equations) of coupled conductance-based neurons using a highly parallelised and scalable framework that uses Python and TensorFlow. 

### Requirements:
- Python 3.6 or above
- Jupyter Notebook/Lab
- Numpy
- Matplotlib
- Seaborn (optional)
- TensorFlow CPU/GPU (1.3 or above)

Happy Coding!
Saptarshi Soham Mohanta and Collins Assisi, 2019.
