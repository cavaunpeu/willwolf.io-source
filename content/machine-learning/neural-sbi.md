Title: Neural Methods in Simulator-Based Inference
Date: 2021-12-27 09:00
Author: Will Wolf
Lang: en
Slug: neural-methods-in-sbi
Status: published
Summary: A survey of how neural networks are currently being used in simulator-based inference routines.
Image:

Bayesian inference is the task of quantifying a posterior belief over parameters $\bm{\theta}$ given observed data $\mathbf{x}$—where $\mathbf{x}$ was generated from a model $p(\mathbf{x}\vert{\bm{\theta}})$—via Bayes' Theorem:

\begin{equation*}
    p(\bm{\theta}\vert\mathbf{x}) = \frac{p(\mathbf{x}\vert\bm{\theta})p(\bm{\theta})}{p(\mathbf{x})}
\end{equation*}

In numerous applications of scientific interest, e.g. cosmological, climatic or urban-mobility phenomena, the likelihood of the data $\mathbf{x}$ under the data-generating function $p(\mathbf{x}\vert\bm{\theta})$ is intractable to compute, precluding classical inference approaches. Notwithstanding, *simulating* new data $\mathbf{x}$ from this function is often trivial—for example, by coding the generative process in a few lines of Python—

```python
def generative_process(params):
    data = ...  # some deterministic logic
    data = ...  # some stochastic logic
    data = ...  # whatever you want!
    return data

simulated_data = [generative_process(p) for p in [.2, .4, .6, .8, 1]]
```

—motivating the study of *simulation-based* Bayesian *inference* methods, termed SBI.

Recent work has explored the use of neural networks to perform key density estimation tasks, i.e. subroutines, of the SBI routine itself. We refer to this work as Neural SBI. In the following sections, we detail the various classes of these estimation tasks. For a more thorough analysis of their respective motivations, behaviors, and tradeoffs, we refer the reader to the original work.

# Neural Posterior Estimation

In this class of models, we estimate the true posterior $p(\bm{\theta}\vert\mathbf{x})$ with a conditional neural density estimator $q_{\phi}(\bm{\theta}\vert\mathbf{x})$. Simply, this estimator is a neural network with parameters $\phi$ that accepts $\mathbf{x}$ as input and produces $\bm{\theta}$ as output. It is trained on data tuples $\{\bm{\theta}_n, \mathbf{x}_n\}_{1:N}$ sampled from $p(\mathbf{x}, \bm{\theta}) = p(\mathbf{x}\vert\bm{\theta})p(\bm{\theta})$, where $p(\bm{\theta})$ is a prior we choose, and $p(\mathbf{x}\vert\bm{\theta})$ is our *simulator*. For example, constructing this training set might look as follows:

```python
data = []

theta = prior.sample()
x = generative_process(params=theta)

data.append((x, theta))
```

Then, we train our network.

```python
q_phi.train(data)
```

Finally, once trained, we can estimate the true posterior $p(\bm{\theta}\vert\mathbf{x} = \mathbf{x}_o)$—our posterior belief over parameters $\bm{\theta}$ given our *observed* (not simulated!) data $\mathbf{x}_o$ via $q_{\phi}(\bm{\theta}\vert\mathbf{x} = \mathbf{x}_o)$.

## Learning the wrong estimator

Ultimately, our goal is to perform the following computation:

$$
q_{\phi}(\bm{\theta}\vert\mathbf{x} = \mathbf{x}_o)
$$

Such that $q_{\phi}$ produces an *accurate* estimation of the parameters $\bm{\theta}$ given observed data $\mathbf{x}_o$, we require that $q_{\phi}$ be *trained* on tuples $\{\bm{\theta}_n, \mathbf{x}_n\}$ where:

1. $\mathbf{x}_n \sim p(\mathbf{x}\vert\bm{\theta}_n)$ via our simulation step.
2. $\vert\mathbf{x}_n - \mathbf{x}_o\vert$ is small, i.e. our simulated are nearby our observed data.

Otherwise, $q_{\phi}$ will learn estimator a posterior over parameters given data *unlike* our own.

## Learning a better estimator

So, how do we obtain parameters $\bm{\theta}_n$ that will produce $\mathbf{x}_n \sim p(\mathbf{x}\vert\bm{\theta}_n)$ near $\mathbf{x}_o$? We take those that have high (estimated) posterior density given $\mathbf{x}_o$!

In this vein, we build our training set as follows:

```python
data = []

theta = q_phi(x=x_o).sample()
x = generative_process(params=theta)

data.append((x, theta))
```

Stitching this all together, our SBI routine looks as follows:

```python
for e in range(N_EPOCHS):
    data = []
    for _ in range(N_SAMPLES):
        if e == 0:
            theta = prior.sample()
        else:
            theta = q_phi(x=x_o).sample()
        x = generative_process(params=theta)

        data.append((x, theta))
    q_phi.train(data)

posterior_samples = [q_phi(x=x_o).sample() for _ in range(ANY_NUMBER)]
```

## Learning the right estimator

Unfortunately, we're still left with a problem:

1. In the first epoch, we learn $q_{\phi, e=0}(\bm{\theta}\vert\mathbf{x}) \propto p(\mathbf{x}\vert\bm{\theta})p(\bm{\theta})$, i.e. the **right** estimator.
2. Otherwise, we learn $q_{\phi, e}(\bm{\theta}\vert\mathbf{x}) \propto p(\mathbf{x}\vert\bm{\theta})q_{\phi, e-1}(\bm{\theta}\vert\mathbf{x})$, i.e. the **wrong** estimator.

So, how do we correct this mistake?

# Neural Likelihood Estimation

# Neural Likelihood Ratio Estimation


# References
```
1. @article{
    10.1073/pnas.1912789117,
    year = {2020},
    title = {{The frontier of simulation-based inference}},
    author = {Cranmer, Kyle and Brehmer, Johann and Louppe, Gilles},
    journal = {Proceedings of the National Academy of Sciences},
    issn = {0027-8424},
    doi = {10.1073/pnas.1912789117},
    pmid = {32471948},
    pages = {30055--30062},
    number = {48},
    volume = {117}
}
```
