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

While there are many potential themes of probabilistic models we might explore, we'll herein focus on two: **generative vs. discriminative models**, and **"fully Bayesian" vs. "lowly point estimate" learning**.

Finally, our pool tube is not a godhead — we are not nautical missionaries brandishing a divine statistical truth, demanding that each model we encounter interpret this truth in a rigid fashion, bottom-up fashion. Instead, we'll begin with the goals, advantages and limitations of each model type, and fall back on Bayes' theorem to bridge the gaps between. Without it, we'd sadly start stinking.

emboldened by richard mcelreath

# inference, then prediction

# discriminative vs. generative
- keep bayes' theorem to the side as a helper, not as a standing rock in the middle to work around
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
