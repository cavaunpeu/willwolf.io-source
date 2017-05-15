Title: Minimizing the Negative Log-Likelihood, in English
Date: 2017-04-07 10:07
Author: Will Wolf
Lang: en
Slug: minimizing_the_negative_log_likelihood_in_english
Status: draft
Summary:
Image:

Roughly speaking, my machine learning journey began on [Kaggle](http://kaggle.com). "There's data, a model and a loss function to optimize," I learned. "Regression models predict continuous-valued real numbers; classification models predict 'red,' 'green,' 'blue.' Typically, the former employs the mean squared error or mean absolute error; the latter, the cross-entropy loss. Stochastic gradient descent updates the model's parameters to drive these losses down." Furthermore, to build these models, just `import sklearn`.

A dexterity with the above is often sufficient for -- at least from a technical stance -- both employment and impact as a Data Scientist. In industry, commonplace prediction and inference problems -- binary churn prediction, credit scoring, product recommendation and A/B testing, for example -- are easily matched with an off-the-shelf algorithm plus proficient Data Scientist for a measurable boost to the company's bottom line. In a vacuum, I think this is fine: the winning driver does not *need* to know how to build the car. Surely, I've been this driver before.

Once fluid with "scikit-learn fit and predict," I turned to statistics. I was always aware that the two were related, yet figured them ultimately parallel sub-fields of my job. With the former, I build classification models; with the latter, I infer signup rates with the exponential distribution and MCMC -- right?

Before long, I dove deeper into machine learning -- reading textbooks, studying documentation and writing this blog. Therein, I began to come across *terms I didn't understand used to describe the things that I did.* "I understand what the categorical cross-entropy loss is, what it does and how it's defined," for example; **"why are you calling it the negative log-likelihood?"**

Marginally wiser, I now know two truths about the above:
1. Techniques we anoint as "machine learning" - classification and regression models, notably - have their underpinnings almost entirely in statistics. For this reason, terminology can often flow between.
2. None of this stuff is new.

The goal of this post is to take three models we know, love, and know how to use and explain what's really going on underneath the hood. Notwithstanding, it typically *is* in our best interest to use an off-the-shelf model in a production setting just the same -- something we already know how to do. As such, this post will start and end here: your head is currently above water; we're going to dive into the pool, touch the bottom, then come back up to the surface.

![bottom of pool](http://img2.hungertv.com/wp-content/uploads/2014/09/SP_Kanawaza-616x957.jpg)

This post will have three protagonists.

The goal of this post...

3 characters.
  - their canonical loss functions
    - mean, median
  - their canonical loss functions plus regularization
  - their functional forms
  - their output distributions themselves, i.e. max entropy

Like many dove deeper into the field. I started reading textbooks,

are relatively straightforward

In industry, most prediction and inference problems are relatively straightforward

Most prediction and inference problems in industry are relative

to be an impactful Data Scientist.

motivate

! start with linear regression and go all the way through. then repeat (quickly enough) for logistic regression.

- 'its easy to start with sklearn docs, y_true, y_pred, etc. and never think about think in terms of probabilistic models'

- mean for l2, median for l1

## what is marginalization/integration

## MLE, MAP
- MLE on a PMF for classification?
  - yeah this is how you back out into the KL divergence/cross entropy loss
- MLE ...with a uniform prior on theta
  - it can run off to huge values!
- overfitting tradeoffs
  - regularization
  - shrinkage!
- MAP tends to MLE as n goes to infin
  - convex combination of prior and sample mean
- regularization

To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding the maximum likelihood esti- mate of θ. This is thus one set of assumptions under which least-squares re- gression can be justified as a very natural method that’s just doing maximum likelihood estimation.

..check out that camdp post on regularization mle blah linear regression

## exponential families

posterior predictive distribution

## fully bayesian approaches
- theta is usually very high-dimensional and we can't integrate over the full thing
- so, we take the MAP and get a point-estimate

## obtaining the MLE, MAP via various inference algorithms

## exponential families

## maximum entropy distributions
- why do we actually choose normal? max entrp!
## regularization terms

## bayesian models
- absolute loss implies we pick the posterior median
- squared loss implies we pick the posterior mean

## samplers

## references
-
