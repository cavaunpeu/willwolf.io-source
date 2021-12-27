Title: Neural Methods in Simulator-Based Inference
Date: 2021-12-27 09:00
Author: Will Wolf
Lang: en
Slug: neural-methods-in-sbi
Status: published
Summary: A survey of how neural networks are currently being used in simulator-based inference routines.
Image:

Bayesian inference is the task of quantifying a posterior belief over parameters ${\theta}$ given observed data $\mathbf{x}$—where $\mathbf{x}$ was generated from a model $p(\mathbf{x}\vert{\theta})$—via Bayes' Theorem:

\begin{equation*}
    p({\theta}\vert\mathbf{x}) = \frac{p(\mathbf{x}\vert{\theta})p({\theta})}{p(\mathbf{x})}
\end{equation*}

In numerous applications of scientific interest, e.g. cosmological, climatic or urban-mobility phenomena, the likelihood of the data $\mathbf{x}$ under the data-generating function $p(\mathbf{x}\vert{\theta})$ is intractable to compute, precluding classical inference approaches. Notwithstanding, *simulating* new data $\mathbf{x}$ from this function is often possible—for example, by coding the generative process in a few lines of Python—

```python
def my_generative_process(params):
    data = ...  # some deterministic logic
    data = ...  # some stochastic logic
    data = ...  # whatever you want!
    return data

simulated_data = [generative_process(p) for p in [.2, .4, .6, .8, 1]]
```

—motivating the study of *simulation-based* Bayesian *inference* methods, termed SBI.

Recent work has explored the use of neural networks to perform key density estimation tasks, i.e. subroutines, of the SBI routine itself. We refer to this work as Neural SBI. In the following sections, we detail the various classes of these estimation tasks.

## Neural Likelihood Estimation


## References
[^1]:
