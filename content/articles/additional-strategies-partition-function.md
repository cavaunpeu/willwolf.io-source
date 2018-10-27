Title: Additional Strategies for Confronting the Partition Function
Date: 2018-10-20 14:00 #
Author: Will Wolf
Lang: en
Slug: additional-strategies-partition-function
Status: published
Summary:
Image:

In the [previous post](https://cavaunpeu.github.io/2018/10/20/thorough-introduction-to-boltzmann-machines/) we introduced Boltzmann machines and the infeasibility of computing the gradient of its log-partition function $\nabla_{\theta}\log{Z}$. To this end, we explored one strategy for its approximation: Gibbs sampling. This owes to the fact that the expression for the log-partition function is an expectation over the model distribution which can be approximated with Monte Carlo samples.

In this post, we'll highlight the imperfections of even this approach, then present more preferable alternatives.

## Pitfalls of vanilla Gibbs sampling

To refresh, the two gradients we seek to compute in a reasonable amount of time are:

$$
\nabla_{w_{i, j}}\log{Z} = \mathop{\mathbb{E}}_{x \sim p_{\text{model}}} [x_i  x_j]\\
\nabla_{b_{i}}\log{Z} = \mathop{\mathbb{E}}_{x \sim p_{\text{model}}} [x_i]
$$

Via Gibbs sampling, we approximate each by:

1. Burning in a Markov chain w.r.t. our model, then selecting $n$ samples from this chain
2. Evaluating both functions ($x_i  x_j$, and $x_i$) at these samples
3. Taking the average of each

Concretely:

$$
\nabla_{w_{i, j}}\log{Z} \approx \frac{1}{N}\sum\limits_{k=1}^N x^{(k)}_i  x^{(k)}_j\quad\text{where}\quad x^{(k)} \sim p_{\text{model}}\\
\nabla_{b_{i}}\log{Z} \approx \frac{1}{N}\sum\limits_{k=1}^N x^{(k)}_i\quad\text{where}\quad x^{(k)} \sim p_{\text{model}}
$$

**We perform this sampling process at each gradient step.**

### The cost of burning in each chain

Initializing a Markov chain at a random sample incurs a "burn-in" process which comes at non-trivial cost. If paying this cost at each gradient step, it begins to add up. How can we do better?

**In the remainder of the post, we'll explore two new directives for approximating the negative phase more cheaply, and the algorithms they birth.**

## Directive \#1: Cheapen the burn-in process

## Stochastic maximum likelihood

SML assumes the premise: let's initialize our chain at a point already close to the model's true distribution—reducing or perhaps eliminating the cost of burn-in altogether.  **This said, at what sample do we initialize the chain?**

In SML, we simply initialize at the terminal value of the previous chain (i.e. the one we manufactured in the previous gradient step). **As long as the model has not changed significantly since, i.e. as long as the previous gradient step was not too large, this sample should exist in a region of high probability w.r.t. the current model.**

In code, this might look like:

```python
n_obs, dim = X.shape  # X holds all of our observations

# Vanilla Gibbs sampling
samples = [np.zeros(dim)]

# SML
samples = [previous_samples[-1]]
```

### Implications
Per the expression for the full log-likelihood gradient, e.g. $\nabla_{w_{i, j}}\log{\mathcal{L}} = \mathop{\mathbb{E}}_{x \sim p_{\text{data}}} [x_i  x_j] - \mathop{\mathbb{E}}_{x \sim p_{\text{model}}} [x_i  x_j]$, the negative phase works to "reduce the probability of the points in which the model strongly believes,"[^1], i.e. its incorrect beliefs about the world. Since we approximate this term with samples *roughly from* the model's true distribution, **we do not encroach on this foundational task.**

## Contrastive divergence

Alternatively, in the contrastive divergence algorithm, we initialize the chain at each gradient step with a sample from the data distribution.

### Implications

With no guarantee that the data distribution resemble the model distribution, we may systematically fail to sample, and thereafter "suppress," points that are incorrectly likely under the latter (as they do not appear in the former!). **This incurs the growth of "spurious modes"** in our model, aptly named.[^1]

In code, this might look like:

```python
# Vanilla Gibbs sampling
samples = [np.zeros(dim)]

# SML
samples = [previous_samples[-1]]

# Contrastive divergence
samples = [X[np.random.choice(n_obs)]]
```

Cheapening the burn-in phase indeed gives us a more efficient training routine. Moving forward, what are some even more aggressive strategies we can explore?

## Code
The [repository](https://github.com/cavaunpeu/boltzmann-machines) and [rendered notebook](https://nbviewer.jupyter.org/github/cavaunpeu/boltzmann-machines/blob/master/boltzmann-machines-part-1.ipynb) for this project can be found at their respective links.

## References
[^1]: [CSC321 Lecture 19: Boltzmann Machines](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/slides/lec19.pdf)
[^2]: [Derivation: Maximum Likelihood for Boltzmann Machines](https://theclevermachine.wordpress.com/2014/09/23/derivation-maximum-likelihood-for-boltzmann-machines/)
[^3]: [Boltzmann Machines](https://www.cs.toronto.edu/~hinton/csc321/readings/boltz321.pdf)
