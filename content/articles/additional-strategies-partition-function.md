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

## Directive \#2: Skip the computation of $Z$ altogether

Canonically, we write the log-likelihood of our Boltzmann machine as follows:

$$
\begin{align*}
\log{\mathcal{L}(x)}
&= \log{\frac{\exp{(H(x))}}{Z}}\\
&= \log{\big(\exp{(H(x))}\big)} - \log{Z}\\
&= H(x) - \log{Z}
\end{align*}
$$

Instead, what if we simply wrote this as:

$$
\log{\mathcal{L}(x)} = H(x) - c
$$

or, more generally:

$$
\log{p_{\text{model}}(x)} = \log{\tilde{p}_{\text{model}}(x; \theta)} - c
$$

and estimated $c$ as a parameter?

**Immediately, we remark that if we optimize this model with maximum likelihood, our algorithm will, trivially, make $c$ arbitrarily negative.** In other words, the quickest way to increase the thing on the left is to decrease $c$.

How might we better phrase this problem?

## Noise contrastive estimation

Ingeniously, NCE proposes an alternative:

1. Posit two distributions: the model, and a noise distribution
2. Given a data point, predict from which distribution this point was generated

Let's unpack this a bit.

Under an (erroneous) MLE formulation, we would optimize the following objective:

$$
\theta, c = \underset{\theta, c}{\arg\max}\  \mathbb{E}_{x \sim p_{\text{data}}} [\log{p_{\text{model}}}(x)]
$$

Under NCE, we're going to replace two pieces so as to perform the binary classification task described above (with 1 = "model", and 0 = "noise").

First, let's swap $\log{p_{\text{model}}}(x)$ with $\log{p_{\text{joint}}}(y = 0\vert x)$, where:

$$
p_{\text{joint}}(x\vert y) =
\begin{cases}
p_{\text{noise}}(x)\quad y = 0\\
p_{\text{model}}(x)\quad y = 1\\
\end{cases}
$$

$$
p_{\text{joint}}(x, y)
= p_{\text{joint}}(y = 0)p_{\text{noise}}(x) + p_{\text{joint}}(y = 1)p_{\text{model}}(x)
$$

$$
p_{\text{joint}}(y = 0\vert x)
= \frac{p_{\text{joint}}(y = 0)p_{\text{noise}}(x)}{p_{\text{joint}}(y = 0)p_{\text{noise}}(x) + p_{\text{joint}}(y = 1)p_{\text{model}}(x)}
$$

Finally:

$$
\theta, c = \underset{\theta, c}{\arg\max}\  \mathbb{E}_{x \sim p_{\text{data}}} [\log{p_{\text{joint}}(y = 0\vert x)}]
$$

From here, we need to update $x \sim p_{\text{data}}$ to include $y$. We'll do this in two pedantic steps.

First, let's write:

$$
\theta, c = \underset{\theta, c}{\arg\max}\  \mathbb{E}_{x, y=0\ \sim\ p_{\text{noise}}} [\log{p_{\text{joint}}(y\vert x)}]
$$

This equation:

1. Builds a classifier that discriminates between samples generated from the model distribution and noise distribution **trained only on samples from the latter.** Clearly, this will not make for an effective classifier.
2. To accomplish the former, we note that the equation maximizes the likelihood of the noise samples under the noise distribution—where the noise distribution itself has no actual parameters we intend to train.

In solution, we trivially expand our expectation to one over both noise samples, and data samples. In doing so, in predicting $\log{p_{\text{joint}}(y = 1\vert x)} = 1 - \log{p_{\text{joint}}(y = 0\vert x)}$, **we'll be maximizing the likelihood of the data under the model.**

$$
\theta, c = \underset{\theta, c}{\arg\max}\  \mathbb{E}_{x, y\ \sim\ p_{\text{train}}} [\log{p_{\text{joint}}(y \vert x)}]
$$

where:

$$
p_{\text{train}}(x\vert y) =
\begin{cases}
p_{\text{noise}}(x)\quad y = 0\\
p_{\text{data}}(x)\quad y = 1\\
\end{cases}
$$

As a final step, we'll expand our object into something more elegant:

$$
\begin{align*}
p_{\text{joint}}(y = 0\vert x)
&= \frac{p_{\text{joint}}(y = 0)p_{\text{noise}}(x)}{p_{\text{joint}}(y = 0)p_{\text{noise}}(x) + p_{\text{joint}}(y = 1)p_{\text{model}}(x)}\\
&= \frac{1}{1 + \frac{p_{\text{joint}}(y = 1)p_{\text{model}}(x)}{p_{\text{joint}}(y = 0)p_{\text{noise}}(x)}}\\
\end{align*}
$$

Assuming *a priori* that $p_{\text{joint}}(x, y)$ is $k$ times more likely to generate a noise sample, i.e. $\frac{p_{\text{joint}}(y = 1)}{p_{\text{joint}}(y = 0)} = \frac{1}{k}$:

$$
\begin{align*}
p_{\text{joint}}(y = 0\vert x)
&= \frac{1}{1 + \frac{p_{\text{model}}(x)}{p_{\text{noise}}(x)\cdot k}}\\
&= \frac{1}{1 + \exp\big(\log{\frac{p_{\text{model}}(x)}{{p_{\text{noise}}(x)\cdot k}}}\big)}\\
&= \sigma\bigg(-\log{\frac{p_{\text{model}}(x)}{{p_{\text{noise}}(x)\cdot k}}}\bigg)\\
&= \sigma\bigg(\log{k} + \log{p_{\text{noise}}(x)} - \log{p_{\text{model}}(x)}\bigg)\\
p_{\text{joint}}(y = 1\vert x)
&= 1 - \sigma\bigg(\log{k} + \log{p_{\text{noise}}(x)} - \log{p_{\text{model}}(x)}\bigg)
\end{align*}
$$

Given a joint training distribution over $(X_{\text{data}}, y=1)$ and $(X_{\text{noise}}, y=0)$, this is the target we'd like to maximize.

### Implications

For our training data, we require the ability to sample from our noise distribution.

For our target, we require the ability to compute the likelihood of some data under our noise distribution.

Therefore, these criterion do place practical restrictions on the types of noise distributions that we're able to consider.

### Extensions

We briefly alluded to the fact that our noise distribution is non-parametric. However, there is nothing stopping us from evolving this distribution, **i.e. giving it trainable parameters, then updating these parameters such that it generates increasingly "optimal" samples.**

Of course, we would have to design what "optimal" means. One interesting approach is called [Adversarial Contrastive Estimation
](https://arxiv.org/abs/1805.03642), wherein they adapt the noise distribution to generate increasingly "harder negative examples, which forces the main model to learn a better representation of the data."[^2]

## Negative sampling

Negative sampling is the same as NCE except:

1. We consider noise distributions whose likelihood we cannot evaluate
2. To accommodate, we simply set $p_{\text{noise}}(x) = 1$

Therefore:

$$
\begin{align*}
p_{\text{joint}}(y = 0\vert x)
&= \frac{1}{1 + \frac{p_{\text{model}}(x)}{p_{\text{noise}}(x)\cdot k}}\\
&= \frac{1}{1 + \frac{p_{\text{model}}(x)}{ k}}\\
&=\sigma(-\frac{p_{\text{model}}(x)}{ k})\\
&=\sigma(\log{k} - \log{p_{\text{model}}(x)})\\
p_{\text{joint}}(y = 1\vert x)
&= 1 - \sigma(\log{k} - \log{p_{\text{model}}(x)})
\end{align*}
$$

## In code

Since I learn best by implementing things, let's play around.


## Code
The [repository](https://github.com/cavaunpeu/boltzmann-machines) and [rendered notebook](https://nbviewer.jupyter.org/github/cavaunpeu/boltzmann-machines/blob/master/boltzmann-machines-part-1.ipynb) for this project can be found at their respective links.

## References
[^1]: [CSC321 Lecture 19: Boltzmann Machines](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/slides/lec19.pdf)
[^2]: [Derivation: Maximum Likelihood for Boltzmann Machines](https://theclevermachine.wordpress.com/2014/09/23/derivation-maximum-likelihood-for-boltzmann-machines/)
[^3]: [Boltzmann Machines](https://www.cs.toronto.edu/~hinton/csc321/readings/boltz321.pdf)
