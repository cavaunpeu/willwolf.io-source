Title: The Bayesian Formalism: Further Exploring Common Probabilistic Models
Date: 2017-04-07 10:07
Author: Will Wolf
Lang: en
Slug: further-exploring-common-probabilistic-models
Status: draft
Summary:
Image:

The previous post on this blog sought to expose the statistical underpinnings of several machine learning models you know and love. Therein, we made the analogy of a swimming pool: you start on the surface — you know what these models do and how to use them effectively — dive to the bottom — you deconstruct these models into their elementary assumptions and intentions — then finally, work your way back to the surface — reconstructing their functional forms, optimization requirements and loss functions one step at a time.

In this post, we're going to stay on the surface: instead of deconstructing common models, we're going to further explore the relationships between them — swimming to different corners of the pool itself. Keeping us afloat will be Bayes' theorem — our balanced, dependable yet at times fragile pool tube, so to speak — which we'll take with us wherever we go.

![](http://www.yourfaxlesspaydayloan.com/wp-content/uploads/2014/05/inner_tube.png)

While there are many potential themes of probabilistic models we might explore, we'll herein focus on two: **generative vs. discriminative models**, and **"fully Bayesian" vs. "lowly point estimate" learning**. We will stick to the supervised setting as well.

Finally, our pool tube is not a godhead — we are not nautical missionaries brandishing a divine statistical truth, demanding that each model we encounter interpret this truth in a rigid fashion, bottom-up fashion. Instead, we'll begin with the goals, advantages and limitations of each model type, and fall back on Bayes' theorem to bridge the gaps between. Without it, we'd sadly start stinking.

# Discriminative vs. generative models
The goal of a supervised model is to compute the distribution over outcomes $y$ given an input $x$, written $P(y\vert x)$. If $y$ is discrete, this distribution is a probability mass function, e.g. a multinomial or binomial distribution. If continuous, it is a probability density function, e.g. a Gaussian distribution.

## Discriminative models
In discriminative models, we direct our full focus to this output distribution. Taking an example from the previous post, let's assume a softmax regression which receives some data $x$ and predicts a multi-class label `red or green or blue`. The model's output distribution is therefore multinomial; a multinomial distribution requires as a parameter a vector $\pi$ of respective outcome probabilities: `pi = {red: .27, green: .11, blue: .62}`, for example. We can compute these individual probabilities via the softmax function, where:

- $\pi_k = \frac{e^{\eta_k}}{\sum\limits_{k=1}^K e^{\eta_k}}$
- $\eta_k = \theta_k^Tx$
- $\theta$ is a matrix of weights which we must infer, and $x$ is our input.

### Inference
Typically, we perform inference by taking the *maximum likelihood estimate*: "which parameters $\theta$ most likely gave rise to the observed data pairs $D = ((x^{(i)}, y^{(i)}), ..., (x^{(m)}, y^{(m)}))$ via the relationships described above?" We compute this estimate by maximizing the log-likelihood function with respect to $\theta$, or equivalently minimizing the negative log-likelihood in identical fashion — the latter better known as a "loss function" in machine learning parlance.

Unfortunately, the maximum likelihood estimate includes no information about the plausibility of the chosen parameter value itself. As such, we often place a *prior* on our parameter and take the argmax over their product. This gives the *maximum a posterior* estimate, or MAP.

$$
\begin{align*}
\theta_{MAP}
&= \underset{\theta}{\arg\max}\ \log \prod\limits_{i=1}^{m} P(y^{(i)}\vert x^{(i)}; \theta)P(\theta)\\
&= \underset{\theta}{\arg\max}\ \sum\limits_{i=1}^{m} \log{P(y^{(i)}\vert x^{(i)}; \theta)} + \log{P(\theta)}\\
\end{align*}
$$

The $\log{P(\theta)}$ term can be easily rearranged into what is better known as a *regularization term* in machine learning parlance, where the type of prior distribution we place on $\theta$ gives the type of regularization term.

The ["argmax"](https://en.wikipedia.org/wiki/Arg_max) finds the point(s) $\theta$ at which the given function attains its maximum value. As such, the typical discriminative model — softmax regression, logistic regression, linear regression, etc. — returns a single, lowly point estimate for the parameter in question.

#### How do we compute this value?
In the trivial case where $\theta$ is 1-dimensional, we can take the derivative of the function in question with respect to $\theta$, set it equal to 0, then solve for $\theta$. (Additionally, in order to verify that we have indeed obtained a maximum, we should compute a second derivative and verify that its value is negative.)

In the more realistic case where $\theta$ is a high-dimensional vector or matrix, we can compute the argmax by way of an optimization routine like stochastic gradient ascent or, as is more common, the argmin by way of stochastic gradient descent.

#### What if we're uncertain about our parameter estimates?
Consider the following three scenarios — taken from Daphne Koller's [Learning in Probabilistic Graphical Models](https://www.coursera.org/learn/probabilistic-graphical-models-3-learning/home/welcome).

> Two teams play 10 times, and the first wins 7 of the 10 matches.

\> *Estimate that the probability of the first team winning is 0.7.*

Seems reasonable, right?

> A coin is tossed 10 times, and comes out `heads` on 7 of the 10 tosses.

\> *Estimate that the probability of observing `heads` is 0.7.*

Changing only the analogy, this now seems wholly unreasonable, right?

> A coin is tossed 10000 times, and comes out `heads` on 7000 of the 10000 tosses.

\> *Estimate that the probability of observing `heads` is 0.7.*

Finally, increasing the observed counts, the previous scenario seems plausible once more.

I find this a terrific succession of examples with which to convey the notion of *uncertainty* — that the more data we have, the less uncertain we are about what's really going on. This notion is at the heart of Bayesian statistics and is extremely intuitive to us as humans. Unfortunately, when we compute "lowly point estimates," i.e. the argmin of the loss function with respect to our parameters $\theta$, we are discarding this uncertainty entirely. Should our model be fit with $n$ observations where $n$ is not a large number, our estimate would amount to that of Example #2: *a coin is tossed $n$ times, and comes out `heads` on `int(.7n)` of the `n` tosses — estimate that the probability of observing `heads` is `0.7`.*

### Prediction
With the parameter $\theta$ in hand prediction is simple: just plug back into our original function $P(y\vert x)$. With a point estimate for $\theta$, we'll compute but a lowly point estimate for $y$.

## Generative models

# inference, then prediction


- discriminative:
  - let's look at y first
  - posit our PDF/PMF
  - compute a point estimate for theta
  - if we want an estimate for theta that:
    - makes our observed data likely
    - was likely to occur itself
    - ... we add a prior, which gives the argmax of theta over the joint distribution of P(y|x, 0)
- to get full distributions, we need Bayes' denominator
  - it's often intractable -- we therefore approximate
  - approximation methods include:
  - mcmc, variational inference, advi
- generative:
  - posit the data generating process
  - estimate each distribution separately
    - MLE, or bayesian i guess...
  - MAP estimation
    - how to solve by hand
    - why we use SGD
- models:
  - naive bayes
  - multinomial regression
  - HMM's?
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
