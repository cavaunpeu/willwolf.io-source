Title: Foo Bar
Date: 2018-11-23 10:00
Author: Will Wolf
Lang: en
Slug: mean-field-variational-bayes
Status: published
Summary:
Image: figures/.png

"Mean-field variational Bayes" (MFVB), is similar to [expectation-maximization]({filename}../content/em-for-lda.md) (EM), yet distinct in two key ways:

1. We do not minimize $\text{KL}\big(q(\mathbf{Z})\Vert p(\mathbf{Z}\vert\mathbf{X}, \theta)\big)$, i.e. perform the E-step, as [in the problems in which we employ mean-field] the posterior distribution $p(\mathbf{Z}\vert\mathbf{X}, \theta)$ "is too complex to work with," e.g. it has no analytical form.
2. Our variational distribution $q(\mathbf{Z})$ is a *factorized distribution*, i.e.

$$
q(\mathbf{Z}) = \prod\limits_i^{M} q_i(\mathbf{Z}_i)
$$

for all latent variables $\mathbf{Z}_i \in \mathbf{Z} \in \mathbb{R}^M$.

Briefly, factorized distributions are cheaper to compute: if each $q_i(\mathbf{Z}_i)$ is Gaussian, $q(\mathbf{Z})$ requires optimization of $2M$ parameters (a mean and a variance for each factor), while an non-factorized $q(\mathbf{Z}) = \text{Normal}(\mu, \Sigma)$ would require optimization of $M(1 + M)$ parameters ($M$ for the mean, and $M^2$ for the covariance). Following intuition, this gain in computational efficiency comes at the cost of decreased accuracy in approximating the true posterior over latent variables.

## So, what is it?

Mean-field is an iterative maximization of the ELBO, i.e. an iterative M-step, with respect to the variational factors $q_i(\mathbf{Z}_i)$.

In the simplest case, we posit a variational factor over every latent variable, *as well as every parameter*. In other words, as compared to the log-marginal decomposition in EM, $\theta$ is absorbed into $\mathbf{Z}$.

$$
\log{p(\mathbf{X}\vert\theta)} = \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\bigg] + \text{KL}\big(q(\mathbf{Z})\Vert p(\mathbf{Z}\vert\mathbf{X}, \theta)\big)\quad \text{(EM)}
$$

becomes

$$
\log{p(\mathbf{X})} = \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}\bigg] + \text{KL}\big(q(\mathbf{Z})\Vert p(\mathbf{Z}\vert\mathbf{X})\big)\quad \text{(MFVB)}
$$

From there, we simply maximize the ELBO, i.e. $\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}\bigg]$, by iteratively *maximizing with respect to each individual variational factor $q_i(\mathbf{Z}_i)$* in turn.

## What's this do?

Curiously, we note that $\log{p(\mathbf{X})}$ is a *fixed quantity* with respect to $q(\mathbf{Z})$: updating our variational factors *will not change* the marginal log-likelihood of our data.

This said, we note that the ELBO and $p(\mathbf{Z}\vert\mathbf{X})\big)$ trade off linearly: when one goes up $\Delta$, the other goes down by $\Delta$.

As such, (iteratively) maximizing the ELBO in MFVB is akin to minimizing the divergence between the true posterior of the latent variables given data and our variational approximation thereof.

## Derivation

So, what do these updates look like?

First, let's break the ELBO into its two main components:

$$
\begin{align*}
\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}\bigg]
&= \int{q(\mathbf{Z})\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}}d\mathbf{Z}\\
&= \int{q(\mathbf{Z})\log{p(\mathbf{X, Z})}}d\mathbf{Z} - \int{q(\mathbf{Z})\log{q(\mathbf{Z})}}d\mathbf{Z}\\
&= A + B
\end{align*}
$$

From here, we isolate a single variational factor $q_j(\mathbf{Z}_j)$, i.e. the factor with respect to which we're maximizing the ELBO in a given iteration.

$$
\begin{align*}
A
&= \int{q(\mathbf{Z})\log{p(\mathbf{X, Z})}d\mathbf{Z}}\\
&= \int{\prod\limits_{i}q_i(\mathbf{Z}_i)\log{p(\mathbf{X, Z})}d\mathbf{Z}_i}\\
&= \int{q_j(\mathbf{Z}_j)\bigg[\int{\prod\limits_{i \neq j}q_i(\mathbf{Z}_{i})\log{p(\mathbf{X, Z})}}d\mathbf{Z}_i\bigg]}d\mathbf{Z}_j\\
&= \int{q_j(\mathbf{Z}_j){ \mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}] }d\mathbf{Z}_j}\\
\end{align*}
$$

- $\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}$ is a function of Z_j
- do the "inner for loop" exercise

## second term

we note that this is the entropy of the full variational joint

$$
\begin{align*}
B
&= - \int{q(\mathbf{Z})\log{q(\mathbf{Z})}}d\mathbf{Z}\\
&= - \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{q(\mathbf{Z})}\bigg]\\
&= - \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\prod\limits_{i}q_i(\mathbf{Z}_i)}\bigg]\\
&= - \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\sum\limits_{i}\log{q_i(\mathbf{Z}_i)}\bigg]\\
&= - \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{q_j(\mathbf{Z}_j)} + \sum\limits_{i \neq j}\log{q_i(\mathbf{Z}_i)}\bigg]\\
&= - \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{q_j(\mathbf{Z}_j)}\bigg] - \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\sum\limits_{i \neq j}\log{q_i(\mathbf{Z}_i)}\bigg]\\
&= - \mathop{\mathbb{E}}_{q_j(\mathbf{Z}_j)}\bigg[\log{q_j(\mathbf{Z}_j)}\bigg] - \mathop{\mathbb{E}}_{q_{i \neq j}(\mathbf{Z}_i)}\bigg[\sum\limits_{i \neq j}\log{q_i(\mathbf{Z}_i)}\bigg]\\
&= - \mathop{\mathbb{E}}_{q_j(\mathbf{Z}_j)}\bigg[\log{q_j(\mathbf{Z}_j)}\bigg] + \text{const}\\
&= - \int{q_j(\mathbf{Z}_j)\log{q_j(\mathbf{Z}_j)}}d\mathbf{Z}_j + \text{const}\\
\end{align*}
$$

## Putting it back together

$$
\begin{align*}
\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}\bigg]
&= A + B\\
&= \int{q_j(\mathbf{Z}_j){ \mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}] }d\mathbf{Z}_j} - \int{q_j(\mathbf{Z}_j)\log{q_j(\mathbf{Z}_j)}}d\mathbf{Z}_j + \text{const}\\
\end{align*}
$$

It would be great if we could remove the expectation, since we'd then have a (negative) KL divergence!

Acknowledging that the $\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}]$ in the integrand is an unnormalized log-likelihood written as a function of $\mathbf{Z}_j$, we temporarily rewrite it as:

$$
\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}] = \log{\tilde{p}(\mathbf{X}, \mathbf{Z}_j})
$$

Then:

$$
\begin{align*}
\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}\bigg]
&= \int{q_j(\mathbf{Z}_j){ \log{\tilde{p}(\mathbf{X}, \mathbf{Z}_j}) }d\mathbf{Z}_j} - \int{q_j(\mathbf{Z}_j)\log{q_j(\mathbf{Z}_j)}}d\mathbf{Z}_j + \text{const}\\
&= \int{q_j(\mathbf{Z}_j){ \log{\frac{\tilde{p}(\mathbf{X}, \mathbf{Z}_j)}{q_j(\mathbf{Z}_j)}} }d\mathbf{Z}_j} + \text{const}\\
&= - \text{KL}\big(q_j(\mathbf{Z}_j)\Vert \tilde{p}(\mathbf{X}, \mathbf{Z}_j)\big) + \text{const}\\
\end{align*}
$$

Finally, this expression, i.e. the ELBO, reaches its minimum when:

$$
\begin{align*}
q_j(\mathbf{Z}_j)
&= \tilde{p}(\mathbf{X}, \mathbf{Z}_j)\\
&= \exp{\bigg(\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}]\bigg)}
\end{align*}
$$

NB: When actually applying mean-field, we are computing this expression analytically. *Therefore, to "obtain the optimal density function," we effectively strive to recognize our result as of the form of some canonical density function, along with its parameters.*

Furthermore, we note that this density function is not necessarily normalized. "By inspection," we compute:

$$
\begin{align*}
q_j(\mathbf{Z}_j)
&= \frac{\exp{\bigg(\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}]\bigg)}}{\int{\exp{\bigg(\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}]\bigg)}d\mathbf{Z}_j}}\\
&= \exp{\bigg(\mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}]\bigg)} + \text{const}\\
\end{align*}
$$

So, finally, to work with the log:

$$
\begin{align*}
\log{q_j(\mathbf{Z}_j)}
&= \mathop{\mathbb{E}}_{i \neq j}[\log{p(\mathbf{X, Z})}] + \text{const}\\
\end{align*}
$$

# Approximating a Gaussian

Remember, maximizing the ELBO (which we do by minimizing the KL divergence between  $q_j(\mathbf{Z}_j)$ and $\tilde{p}(\mathbf{X}, \mathbf{Z}_j)$ for all factors $j$) is our mechanism for minimizing the KL divergence between the full factorized posterior $q(\mathbf{Z})$ and the true posterior $p(\mathbf{Z}\vert\mathbf{X})$.

Reread this. Let it sink in!

![png]({filename}/figures/mean-field-variational-bayes/mv-gaussian-approx-1.png)

![png]({filename}/figures/mean-field-variational-bayes/mv-gaussian-approx-2.png)

![png]({filename}/figures/mean-field-variational-bayes/mv-gaussian-approx-3.png)

## Code
The [repository](https://github.com/cavaunpeu/boltzmann-machines) and [rendered notebook](https://nbviewer.jupyter.org/github/cavaunpeu/boltzmann-machines/blob/master/boltzmann-machines-part-2.ipynb) for this project can be found at their respective links.

## References
[^1]: @book{Goodfellow-et-al-2016,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    note={\url{http://www.deeplearningbook.org}},
    year={2016}
}
[^2]: [Adversarial Contrastive Estimation](https://arxiv.org/abs/1805.03642)
