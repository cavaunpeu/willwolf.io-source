Title: The Bayesian Formalism: Further Exploring Common Probabilistic Models
Date: 2017-04-07 10:07
Author: Will Wolf
Lang: en
Slug: further-exploring-common-probabilistic-models
Status: draft
Summary:
Image:

The previous post on this blog sought to expose the statistical underpinnings of several machine learning models you know and love. Therein, we made the analogy of a swimming pool: you start on the surface — you know what these models do and how to use them effectively — dive to the bottom — you deconstruct these models into their elementary assumptions and intentions — then finally, work your way back to the surface — reconstructing their functional forms, optimization exigencies and loss functions one step at a time.

In this post, we're going to stay on the surface: instead of deconstructing common models, we're going to further explore the relationships between them — swimming to different corners of the pool itself. Keeping us afloat will be Bayes' theorem — our balanced, dependable yet at times fragile pool tube, so to speak — which we'll take with us wherever we go.

![](http://www.yourfaxlesspaydayloan.com/wp-content/uploads/2014/05/inner_tube.png)

While there are many potential themes of probabilistic models we might explore, we'll herein focus on two: **generative vs. discriminative models**, and **"fully Bayesian" vs. "lowly point estimate" learning**. We will stick to the supervised setting as well.

Finally, our pool tube is not a godhead — we are not nautical missionaries brandishing a divine statistical truth, demanding that each model we encounter interpret this truth in a rigid, bottom-up fashion. Instead, we'll begin with the goals, advantages and limitations of each model type, and fall back on Bayes' theorem to bridge the gaps between. Without it, we'd quickly start sinking.

# Discriminative vs. generative models
The goal of a supervised model is to compute the distribution over outcomes $y$ given an input $x$, written $P(y\vert x)$. If $y$ is discrete, this distribution is a probability mass function, e.g. a multinomial or binomial distribution. If continuous, it is a probability density function, e.g. a Gaussian distribution.

## Discriminative models
In discriminative models, we immediately direct our focus to this output distribution. Taking an example from the previous post, let's assume a softmax regression which receives some data $x$ and predicts a multi-class label `red or green or blue`. The model's output distribution is therefore multinomial; a multinomial distribution requires as a parameter a vector $\pi$ of respective outcome probabilities: `pi = {red: .27, green: .11, blue: .62}`, for example. We can compute these individual probabilities via the softmax function, where:

- $\pi_k = \frac{e^{\eta_k}}{\sum\limits_{k=1}^K e^{\eta_k}}$
- $\eta_k = \theta_k^Tx$
- $\theta$ is a matrix of weights which we must infer, and $x$ is our input.

### Inference
Typically, we perform inference by taking the *maximum likelihood estimate*: "which parameters $\theta$ most likely gave rise to the observed data pairs $D = ((x^{(i)}, y^{(i)}), ..., (x^{(m)}, y^{(m)}))$ via the relationships described above?" We compute this estimate by maximizing the log-likelihood function with respect to $\theta$, or equivalently minimizing the negative log-likelihood in identical fashion — the latter better known as a "loss function" in machine learning parlance.

Unfortunately, the maximum likelihood estimate includes no information about the plausibility of the chosen parameter value itself. As such, we often place a *prior* on our parameter and take the ["argmax"](https://en.wikipedia.org/wiki/Arg_max) over their product. This gives the *maximum a posterior* estimate, or MAP.

$$
\begin{align*}
\theta_{MAP}
&= \underset{\theta}{\arg\max}\ \log \prod\limits_{i=1}^{m} P(y^{(i)}\vert x^{(i)}; \theta)P(\theta)\\
&= \underset{\theta}{\arg\max}\ \sum\limits_{i=1}^{m} \log{P(y^{(i)}\vert x^{(i)}; \theta)} + \log{P(\theta)}\\
\end{align*}
$$

The $\log{P(\theta)}$ term can be easily rearranged into what is better known as a *regularization term* in machine learning, where the type of prior distribution we place on $\theta$ gives the type of regularization term.

The argmax finds the point(s) $\theta$ at which the given function attains its maximum value. As such, the typical discriminative model — softmax regression, logistic regression, linear regression, etc. — returns a single, lowly point estimate for the parameter in question.

#### How do we compute this value?
In the trivial case where $\theta$ is 1-dimensional, we can take the derivative of the function in question with respect to $\theta$, set it equal to 0, then solve for $\theta$. (Additionally, in order to verify that we have indeed obtained a maximum, we should compute a second derivative and assert that its value is negative.)

In the more realistic case where $\theta$ is a high-dimensional vector or matrix, we can compute the argmax by way of an optimization routine like stochastic gradient ascent or, as is more common, the argmin by way of stochastic gradient descent.

#### What if we're uncertain about our parameter estimates?
Consider the following three scenarios — taken from Daphne Koller's [Learning in Probabilistic Graphical Models](https://www.coursera.org/learn/probabilistic-graphical-models-3-learning/home/welcome).

> Two teams play 10 times, and the first wins 7 of the 10 matches.

\> *Estimate that the probability of the first team winning is 0.7.*

Seems reasonable, right?

> A coin is tossed 10 times, and comes out `heads` on 7 of the 10 tosses.

\> *Estimate that the probability of observing `heads` is 0.7.*

Changing only the analogy, this now seems wholly unreasonable — right?

> A coin is tossed 10000 times, and comes out `heads` on 7000 of the 10000 tosses.

\> *Estimate that the probability of observing `heads` is 0.7.*

Finally, increasing the observed counts, the previous scenario seems plausible once more.

I find this a terrific succession of examples with which to convey the notion of *uncertainty* — that the more data we have, the less uncertain we are about what's really going on. This notion is at the heart of Bayesian statistics and is extremely intuitive to us as humans. Unfortunately, when we compute "lowly point estimates," i.e. the argmin of the loss function with respect to our parameters $\theta$, we are discarding this uncertainty entirely. Should our model be fit with $n$ observations where $n$ is not a large number, our estimate would amount to that of Example #2: *a coin is tossed $n$ times, and comes out `heads` on `int(.7n)` of the `n` tosses — estimate that the probability of observing `heads` is squarely, immovably, `0.7`.*

Conversely, what does *including* uncertainty look like? It looks like a *distribution*: a range of possible values for $\theta$. Further, these values are of varying plausibility as dictated by the data we've observed. In Example #2, while we'd still say that $\Pr(\text{heads}) = .7$ is the parameter value *most likely* to have generated our data, we'd additionally maintain that values in $[.3, .9]$ are plausible, albeit less so, as well. Again, this logic should be simple to grasp: it comes easy to us as humans.

### Prediction
With the parameter $\theta$ in hand prediction is simple: just plug back into our original function $P(y\vert x)$. With a point estimate for $\theta$, we'll compute but a lowly point estimate for $y$.

## Generative models
In generative models, we instead compute *component parts* of the desired output distribution $P(y\vert x)$ instead of directly computing $P(y\vert x)$ itself. To examine these parts, we'll turn to Bayes' theorem:

$$
P(y\vert x) = \frac{P(x\vert y)P(y)}{P(x)}
$$

The numerator posits a generative mechanism for the observed data pairs $D = ((x^{(i)}, y^{(i)}), ..., (x^{(m)}, y^{(m)}))$ in idiomatic terms. Specifically, it states that each pair was generated by:

1. Selecting a label $y^{(i)}$ from $P(y)$. If our model is predicting `red or green or blue`, $P(y)$ is likely a multinomial distribution. (If our observed label counts are `{'red': 20, 'green': 50, blue: 30}`, we would retrodictively believe this multinomial distribution to have had a parameter vector near $\pi = [.2, .5, .3]$.)
2. Given a label $y^{(i)}$, select a value $x^{(i)}$ from $P(x\vert y)$. Trivially, this means that we are positing *three distinct distributions* of this form: $P(x\vert y=\text{red}), P(x\vert y=\text{green}), P(x\vert y=\text{blue})$.

### Inference
The inference task is to compute $P(y)$ and each distinct $P(x\vert y_k)$. In a classification setting, the former is likely a multinomial distribution. The latter might be a multinomial distribution in the case of discrete-feature data, a set of binomial distributions in the case of discrete-feature data, or a set of Gaussian distributions in the case of continuous-feature data. In fact, these distributions can be whatever you'd like, dictated by the assumptions you make in your model.

Finally, we can compute these distributions as per normal: via a maximum likelihood estimate, a MAP estimate, etc.

### Prediction
To compute $P(y\vert x)$ we return to the beginning of this section:

$$
P(y\vert x) = \frac{P(x\vert y)P(y)}{P(x)}
$$

We now have the numerator $P(y)$ in hand and three distinct conditional distributions $P(x\vert y=\text{red}), P(x\vert y=\text{green})$ and $P(x\vert y=\text{blue})$. What about the denominator?

#### Conditional probability and marginalization
The axiom of conditional probability allows us to write $P(B\vert A)P(A) = P(B, A)$, i.e. the *joint probability* of $B$ and $A$. This is a simple algebraic manipulation. As such, we can rewrite Bayes' theorem in its more compact form.

$$
P(y\vert x) = \frac{P(x, y)}{P(x)}
$$

Another manipulation of probability distributions is the *marginalization* operator, which allows us to write:

$$
\int P(x, y)dy = P(x)
$$

As such, we can *marginalize $y$ out of the numerator* so as to obtain the denominator we require.

##### Marginalization example
Marginalization took me a while to understand. Imagine we have the following joint probability distribution out of which we'd like to marginalize $A$.

| $A$   | $B$   | $p$   |
|-------|-------|-------|
| $a^1$ | $b^7$ | $.03$ |
| $a^2$ | $b^8$ | $.14$ |
| $a^3$ | $b^7$ | $.09$ |
| $a^1$ | $b^8$ | $.34$ |
| $a^2$ | $b^8$ | $.23$ |
| $a^3$ | $b^8$ | $.17$ |

The result of marginalization is $P(B)$, i.e. "what is the probability of observing each of the distinct values of $B$?" In this example there are two — $b^7$ and $b^8$. To marginalize over $A$, we simply:

1. Delete the $A$ column.
2. "Collapse" the remaining columns — in this case, $B$.

Step 1 gives:

| $B$   | $p$   |
|-------|-------|
| $b^7$ | $.03$ |
| $b^8$ | $.14$ |
| $b^7$ | $.09$ |
| $b^8$ | $.34$ |
| $b^8$ | $.23$ |
| $b^8$ | $.17$ |

Step 2 gives:

| $B$   | $p$              |
|-------|------------------|
| $b^7$ | $.03 + .09 = .12$|
| $b^8$ | $.14 + .34 + .23 + .17 = .88$|

In the context of our generative model with a given input $x$, the result of this marginalization is a *scalar* — not a distribution. To see why, let's construct the joint distribution then marginalize:

$P(x, y)$:

| $y$            | $X$ | $P(y, X)$                |
|----------------|-----|--------------------------|
| $\text{red}$   | $x$ | $P(y = \text{red}, x)$   |
| $\text{green}$ | $x$ | $P(y = \text{green}, x)$ |
| $\text{blue}$  | $x$ | $P(y = \text{blue}, x)$  |

$\int P(x, y)dy = P(x)$:

| $X$ | $P(y, X)$       |
|-----|-----------------|
| $x$ | $P(y = \text{red}, x) + P(y = \text{green}, x) + P(y = \text{blue}, x)$  |

The resulting probability distribution is over a single value: it is a scalar. This scalar *normalizes* the respective numerator terms such that:

$$
\frac{P(y = \text{red}, x)}{P(x)} +
\frac{P(y = \text{green}, x)}{P(x)} +
\frac{P(y = \text{blue}, x)}{P(x)}
= 1
$$

This gives $P(y\vert x)$: a valid probability distribution over the class labels $y$.

#### Partition function
$P(x)$ often takes another name and even another variable: $Z$, the *partition function*. The stated purpose of this function is to normalize the numerator such that the above summation-to-1 holds. This normalization is necessary because the numerators typically will not sum to 1 themselves, which follows logically from the fact that:

$$
\begin{align*}
\sum\limits_{k = 1}^K P(y = k) = 1
\end{align*}
$$

$$
\begin{align*}
P(x\vert y = k) \neq 1
\end{align*}
$$

Since $(1)$ is always true, $(2)$ would also have to hold true such that:

$$
\sum\limits_{k = 1}^K P(y = k)P(x\vert y = k) = \sum\limits_{k = 1}^K P(x, y = k) = 1
$$

Unfortunately, this is rarely the case.

As you'll now note, the $x$-specific partition function gives an equivalent result to the marginalized-over-$y$ joint distribution: a scalar value $P(x)$ with which to normalize the numerator. However, crucially, please keep in mind:

-  *The partition function is a specific component of a probabilistic model which always yields a scalar*.
- *Marginalization is a much more general operation performed on a probability distribution, which yields a scalar only when the remaining variables(s) are homogeneous, i.e. each remaining columns contains a single distinct value.*
  - In the majority of cases it will simply yield a reduced probability distribution over many value configurations, similar to the $P(B)$ example above.

#### In practice, this is superfluous
If we neglect to compute $P(x)$, i.e. if we don't normalize our joint distributions $P(x, y = k)$, we'll be left with an invalid probability distribution $\tilde{P}(y\vert x)$ in which the values do not sum to 1. This distribution might look like `P(y|x) = {'red': .00047, 'green': .0011, 'blue': .0000853}`. *If our goal is to simply compute the most likely label, taking the argmax of this unnormalized distribution works just fine.* This follows trivially from our Bayesian pool ring:

$$
\underset{y}{\arg\max}\ \frac{P(x, y)}{P(x)} = \underset{y}{\arg\max}\ P(x, y)
$$

# "Fully Bayesian learning"
We previously lamented the shortcomings of "lowly point estimates" and sang the praises of computing estimates as full distributions. Unfortunately, this is a computationally-hard thing to do.

To see why, let's revisit Bayes' theorem. Assume we are estimating the parameters $\theta$ of a softmax regression model and have placed a prior on $\theta$. In concrete terms, this estimate can be written as $P(\theta\vert D = ((x^{(i)}, y^{(i)}), ..., (x^{(m)}, y^{(m)})))$: the distribution over our belief in the true value of $\theta$ given the data we've observed. Bayes' theorem allows us to expand this quantity into:

$$
P(\theta\vert D) = \frac{P(D\vert\theta)P(\theta)}{P(D)}
$$

Previously, we computed a "lowly point estimate" for this quantity — the MAP — as:

$$
\begin{align*}
\theta_{MAP}
&= \underset{\theta}{\arg\max}\ \log \prod\limits_{i=1}^{m} P(y^{(i)}\vert x^{(i)}; \theta)P(\theta)\\
&= \underset{\theta}{\arg\max}\ \log \prod\limits_{i=1}^{m} P(\theta\vert (y^{(i)}, x^{(i)}))\\
\end{align*}
$$

While $P(y^{(i)}\vert x^{(i)}; \theta)P(\theta) \neq P(\theta\vert (y^{(i)}, x^{(i)}))$, the argmaxes of the respective products are equal. For this reason, we were able to compute a point estimate for $P(\theta\vert D)$, i.e. a "summarization" of $P(\theta\vert D)$ in a single value, without ever computing the denominator $P(D)$.

(As a brief aside, please note that we could summarize $P(\theta\vert D)$ with *any* single value from this distribution. We often select the maximum likelihood estimate — the single value of $\theta$ that most likely gave rise to our data, or the MAP — the single value of $\theta$ that most likely gave rise to our data and most plausibly occurred itself.)

To compute $P(\theta\vert D)$ itself — trivially, a full distribution as the term suggests — we will need to compute $P(D)$ after all. As before, this can be accomplished via marginalization:

$$
\begin{align*}
P(\theta\vert D)
&= \frac{P(D\vert\theta)P(\theta)}{P(D)}\\
&= \frac{P(D, \theta)}{P(D)}\\
&= \frac{P(D, \theta)}{\int P(D, \theta)d\theta}\\
\end{align*}
$$

As $\theta$ takes continuous values, we can no longer employ the "delete and collapse" method of marginalization in discrete distributions. Furthermore, in all but trivial cases, $\theta$ is a high-dimensional vector or matrix, leaving us to compute a "high-dimensional integral that lacks an analytic (closed-form) solution — the central computational challenge in inference."[^1]

As such, computing the full distribution $P(\theta\vert D)$ becomes *approximating* the full distribution $P(\theta\vert D)$.

## Markov chain monte carlo
In small to medium-sized models, we often take an alternative ideological approach to approximating $P(\theta\vert D)$: instead of computing a distribution — a gory algebraic expression whose shape is dictated by a few canonical parameters — we produce *samples* from this distribution. Roughly speaking, the aggregate of these samples then gives, retrodictively, the distribution itself. The general family of these methods is known as those of Markov chain monte carlo, or MCMC.

In simple terms, MCMC estimation for a given parameter $\phi$ works as follows:

1. Initialize $\phi$ to some value $\phi_{\text{current}}$.
2. Compute the prior probability of $\phi_{\text{current}}$ and the probability of having observed our data under $\phi_{\text{current}}$ — $P(\phi_{\text{current}})$ and $P(D\vert \phi_{\text{current}})$, respectively. Their product gives $P(D, \phi_{\text{current}})$ — the joint probability of having observed the proposed parameter estimate and our observed data.
3. Add $\phi_{\text{current}}$ into a big plastic bucket of "accepted values."
4. Propose moving to a new, nearby value $\phi_{\text{proposal}}$. This value is drawn from an entirely separate *sampling distribution* which bears no influence on our prior $P(\phi)$ nor likelihood function $P(D\vert \phi)$. Repeat Step 2 using $\phi_{\text{proposal}}$ instead of $\phi_{\text{current}}$.
5. Walk the following tree:
    - If $P(D, \phi_{\text{proposal}}) \gt P(D, \phi_{\text{current}})$:
      - Set $\phi_{\text{current}} = \phi_{\text{proposal}}$.
      - Move to Step 3.
    - Else:
      - With some small probability:
        - Set $\phi_{\text{current}} = \phi_{\text{proposal}}$.
        - Move to Step 3.
      - Else:
        - Move to Step 4.

After a few thousand iterations — and discarding the first few hundred, in which we drunkenly amble towards the region of high joint probability — we now have a bucket of samples from our desired posterior distribution. Nota bene: we never had to touch the high-dimensional integral $\int P(D, \theta)d\theta$.

## Variational inference
In large-scale models, one powerful approach to approximating the posterior is to compute this approximation via optimization.
variational inference
mcmc
advi

- approximate inference:
  - variational inference for the ELBO because the denominator is hard
    - faster
  - MCMC to *skip the denominator entirely!*
    - slower
- maybe write out the posterior predictive interval integration because i don't know how to do it
- the partition function:
  - part of the *prediction* step
  - it's the thing we must do if we approach it from the PGM standpoint, and the thing we're already told to do by the GLM standpoint
  - when we're computing the probability of P(y=green|x), we:
    - compute the numerator by an exponentiated wTx
    - normalize by the partition function
    - bayes rule breaks this down further:
      - that probability is P(y=green|x):
        - the joint probability P(y=green, x)/P(x)
      - the denominator could be hard to compute if we have tons of classes:
        - this is mostly because we have to compute numerators for each one -- not for the big bad sum at the end
        - for this we use things like negative sampling
- when it's all said and done, what do full posteriors actually do for us?
  - decision theory
  - comfort
  - fidelity
  - small world
  - the robo post

[^1]: [Edward — Inference of Probabilistic Models](http://edwardlib.org/tutorials/inference)
