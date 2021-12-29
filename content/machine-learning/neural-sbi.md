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

Finally, once trained, we can estimate the true posterior $p(\bm{\theta}\vert\mathbf{x} = \mathbf{x}_o)$—our posterior belief over parameters $\bm{\theta}$ given our *observed* (not simulated!) data $\mathbf{x}_o$ as $\hat{p}(\bm{\theta}\vert\mathbf{x}_o) = q_{\phi}(\bm{\theta}\vert\mathbf{x} = \mathbf{x}_o)$.

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

So, how do we obtain parameters $\bm{\theta}_n$ that produce $\mathbf{x}_n \sim p(\mathbf{x}\vert\bm{\theta}_n)$ near $\mathbf{x}_o$? We take those that have high (estimated) posterior density given $\mathbf{x}_o$!

In this vein, we build our training set as follows:

```python
data = []

theta = q_phi(x=x_o).sample()
x = generative_process(params=theta)

data.append((x, theta))
```

Stitching this all together, our SBI routine looks as follows:

```python
for r in range(N_ROUNDS):
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

1. In the first round, we learn $q_{\phi, r=0}(\bm{\theta}\vert\mathbf{x}) \propto p(\mathbf{x}\vert\bm{\theta})p(\bm{\theta})$, i.e. the **right** estimator.
2. Otherwise, we learn $q_{\phi, r}(\bm{\theta}\vert\mathbf{x}) \propto p(\mathbf{x}\vert\bm{\theta})q_{\phi, r-1}(\bm{\theta}\vert\mathbf{x})$, i.e. the **wrong** estimator.

So, how do we correct this mistake?

In [2], the authors adjust the learned posterior $q_{\phi, r}(\bm{\theta}\vert\mathbf{x})$ by simply dividing it by $q_{\phi, r-1}(\bm{\theta}\vert\mathbf{x})$ then multiplying it by $p(\bm{\theta})$. Furthermore, as they choose $q_{\phi}$ to be a *Mixture Density Network*—a neural network which outputs the parameters of a mixture of Gaussians—and the prior to be "simple distribution (uniform or Gaussian, as is typically the case in practice)," this adjustment can be done analytically.

Conversely, the authors in [3] *train* $q_{\phi}$ on a target *reweighted* to similar effect: instead of maximizing the total (log) likelihood $\Sigma_{n} \log q_{\phi}(\bm{\theta}_n\vert\mathbf{x}_n)$, they maximize $\Sigma_{n} \log w_n q_{\phi}(\bm{\theta}_n\vert\mathbf{x}_n)$, where $w_n = \frac{p(\bm{\theta}_n)}{q_{\phi, r-1}(\bm{\theta}_n\vert\mathbf{x}_n)}$.

While both approaches carry further nuance and potential pitfalls, they bring us effective methods for using a neural network to directly estimate a faithful posterior in SBI routines.

# Neural Likelihood Estimation

In neural likelihood estimation (NLE), we use a neural network to directly estimate the (intractable) likelihood function of the simulator $p(\mathbf{x}\vert\bm{\theta})$ itself. We denote this estimator $q_{\phi}(\mathbf{x}\vert\bm{\theta})$. Finally, we compute our desired posterior as $\hat{p}(\bm{\theta}\vert\mathbf{x}_o) \propto q_{\phi}(\mathbf{x}_o\vert\bm{\theta})p(\bm{\theta})$.

Similar to Neural Posterior Estimation (NPE) approaches, we'd like to learn our estimator on inputs $\bm{\theta}$ that produce $\mathbf{x}_n \sim p(\mathbf{x}\vert\bm{\theta}_n)$ near $\mathbf{x}_o$. To do this, we again sample them from regions of high approximate posterior density. In each round $r$, in NPE, this posterior was $q_{\phi, r-1}(\bm{\theta}\vert\mathbf{x} = \mathbf{x}_o)$; in NLE, it is $q_{\phi, r-1}(\mathbf{x}_o\vert\bm{\theta})p(\bm{\theta})$. In both cases, we draw samples from our approximate posterior density, then feed them to the simulator to generate novel data for training our estimator $q_{\phi}$.

For a more detailed treatment, please refer to original works [4], [5].

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

2. @inproceedings{papamakarios2016,
    author = {Papamakarios, George and Murray, Iain},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {D. Lee and M. Sugiyama and U. Luxburg and I. Guyon and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {Fast $\epsilon$-free Inference of Simulation Models with Bayesian Conditional Density Estimation},
    url = {https://proceedings.neurips.cc/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf},
    volume = {29},
    year = {2016}
}

3. @article{
    lueckmann2017,
    year = {2017},
    title = {{Flexible statistical inference for mechanistic models of neural dynamics}},
    author = {Lueckmann, Jan-Matthis and Goncalves, Pedro J and Bassetto, Giacomo and Öcal, Kaan and Nonnenmacher, Marcel and Macke, Jakob H},
    journal = {arXiv},
    eprint = {1711.01861},
}

4. @InProceedings{
    pmlr-v89-papamakarios19a,
    title = {Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows},
    author = {Papamakarios, George and Sterratt, David and Murray, Iain},
    booktitle = {Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics},
    pages = {837--848},
    year = {2019},
    editor = {Chaudhuri, Kamalika and Sugiyama, Masashi},
    volume = {89},
    series = {Proceedings of Machine Learning Research},
    month = {16--18 Apr},
    publisher = {PMLR},
    pdf = {http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf},
    url = {http://proceedings.mlr.press/v89/papamakarios19a.html}
}

5. @InProceedings{
    pmlr-v96-lueckmann19a,
    title = {Likelihood-free inference with emulator networks},
    author = {Lueckmann, Jan-Matthis and Bassetto, Giacomo and Karaletsos, Theofanis and Macke, Jakob H.},
    booktitle = {Proceedings of The 1st Symposium on Advances in Approximate Bayesian Inference}, pages = {32--53}, year = {2019},
    editor = {Francisco Ruiz and Cheng Zhang and Dawen Liang and Thang Bui},
    volume = {96},
    series = {Proceedings of Machine Learning Research},
    address = {},
    month = {02 Dec},
    publisher = {PMLR},
    pdf = {http://proceedings.mlr.press/v96/lueckmann19a/lueckmann19a.pdf},
    url = {http://proceedings.mlr.press/v96/lueckmann19a.html} }
```
