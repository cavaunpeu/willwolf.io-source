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

The goal of this post is to take three models we know, love, and know how to use and explain what's really going on underneath the hood. I assume the reader is familiar with concepts in both machine learning and statistics, and comes in search of a deeper understanding of the connections therein. There will be math -- but only as much as necessary.

When deploying a predictive model in a production setting, it is generally in our best interest to `import sklearn`, i.e. use a model that someone else has built. This is something we already know how to do. As such, this post will start and end here: your head is currently above water; we're going to dive into the pool, touch the bottom, then come back up to the surface. Lemmas will be written in _**bold**_.

![bottom of pool](http://img2.hungertv.com/wp-content/uploads/2014/09/SP_Kanawaza-616x957.jpg)

First, let's meet our three protagonists. We'll define them in Keras for the illustrative purpose of a unified and idiomatic API.

## [Linear regression](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/) with mean squared error

```python
input = Input(shape=(10,))
output = Dense(1)(input)

model = Model(input, output)
model.compile(optimizer=optimizer, loss='mean_squared_error')
```

## [Logistic regression](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/) with binary cross-entropy loss
```python
input = Input(shape=(10,))
output = Dense(1, activation='sigmoid')(input)

model = Model(input, output)
model.compile(optimizer=optimizer, loss='binary_crossentropy')
```

## [Softmax regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/) with categorical cross-entropy loss
```python
input = Input(shape=(10,))
output = Dense(3, activation='softmax')(input)

model = Model(input, output)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

Next, we'll select four components key to each: its response variable, its functional form, its loss function and its loss function plus regularization. For each model, we'll describe the statistical underpinnings of each component -- the steps on the ladder towards the bottom of the pool.

Before diving in, we'll need to define a few important concepts.

## Random variable
I define a random variable as "a thing that can take on a bunch of different values."
- "The tenure of despotic rulers in Central Africa" is a random variable. It could take on values of 25.73 years, 14.12 years, 8.99 years, ad infinitum; it could not take on values of 1.12 million years, nor -5 years.
- "The height of the next person to leave the supermarket" is a random variable.
- "The color of shirt I wear on Mondays" is a random variable. (Incidentally, this one only has ~3 unique values.)

## Probability distribution
A probability distribution is a lookup table for the likelihood of observing each unique value of a random variable. Assuming a given variable can take on values in $\{\text{rain, snow, sleet, hail}\}$, the following is a valid probability distribution:

```python
p = {'rain': .14, 'snow': .37, 'sleet': .03, 'hail': .46}
```

Trivially, these values must sum to 1.

- A *probability mass function* is a probability distribution for a discrete-valued random variable.
- A *probability density function* _**gives**_ a probability distribution for a continuous-valued random variable.
  - *Gives*, because this function itself is not a lookup table. Given a random variable that takes on values in $[0, 1]$, we do not and cannot define $p(0.01)$, $p(0.001)$, $p(0.0001)$, etc.
  - Instead, we define a function that tells us the probability of observing a value within a certain *range*, i.e. $p(0.01 < X < .4)$.
  - This is the probability density function, where $p(0 \leq X \leq 1) = 1$.

## Entropy
Entropy quantifies the number of ways we can reach a given outcome. Imagine 8 friends are splitting into 2 taxis en route to a Broadway show. Consider the following two scenarios:
  - *4 friends climb into each taxi.* We could accomplish this with the following assignments:

  ```python
  # fill the first, then the second
  assignment_1 = [1, 1, 1, 1, 2, 2, 2, 2]

  # alternate assignments
  assignment_2 = [1, 2, 1, 2, 1, 2, 1, 2]

  # alternate assignments in batches of two
  assignment_1 = [1, 1, 2, 2, 1, 1, 2, 2]

  # etc.
  ```

  - *All friends climb into the first taxi.* There is only one possible assignment.

  ```python
  assignment_1 = [1, 1, 1, 1, 1, 1, 1, 1]
  ```

 (In this case, the Broadway show is probably in [West Africa](http://willtravellife.com/2013/04/how-does-a-west-african-bush-taxi-work/) or a similar part of the world.)

Since there are more ways to reach the first outcome than there are the second, the first outcome has a higher entropy.

### More explicitly
We compute entropy for probability distributions. This computation is given as:

$$
H(p) = -E\log{P_i} = -\sum\limits_{i=1}^{n} p_i \log{p_i}
$$

where:
- There are $n$ unique, possible events.
- Each event $i$ has probability $p_i$.

Entropy is the *weighted-average log probability* over possible events, which measures the *uncertainty inherent in their probability distribution.* The higher the entropy, the less certain we are about the value we're going to get.

Let's calculate the entropy of our distribution above.

```python
p = {'rain': .14, 'snow': .37, 'sleet': .03, 'hail': .46}

def entropy(prob_dist):
    return -sum([ p*log(p) for p in prob_dist.values() ])

In [1]: entropy(p)
Out[1]: 1.1055291211185652
```

For comparison, let's assume two more distributions and calculate their respective entropies.

```python
p_2 = {'rain': .01, 'snow': .37, 'sleet': .03, 'hail': .59}

p_3 = {'rain': .01, 'snow': .01, 'sleet': .03, 'hail': .95}

In [2]: entropy(p_2)
Out[2]: 0.8304250977453105

In [3]: entropy(p_3)
Out[3]: 0.2460287703075343
```

In the first distribution, we are least certain as to what tomorrow's weather will bring. As such, this has the highest entropy. In the third distribution, we are almost certain it's going to hail. As such, this has the lowest entropy.

Now, we can start our descent into the pool.

# Response variable

Roughly speaking, each model looks as follows. It is a diamond that receives an input and produces an output.

![simple input/output model](../images/simple_input_output_model.png)

The models differ in the type of response variable they predict, i.e. the $y$.
- Linear regression predicts a continuous-valued real number. Let's call it `temperature`.
- Logistic regression predicts a binary label. Let's call it `cat or dog`
- Softmax regression predicts a multi-class label. Let's call it `red or green or blue`.

In each model, the response variable can take on a bunch of different values. In other words, they are *random variables.* Which probability distributions are associated with each?

Unfortunately, we don't know. All we do know, in fact, is the following:
- `temperature` has an underlying true mean $\mu \in (-\infty, \infty)$ and variance $\sigma^2 \in (-\infty, \infty)$.
- `cat or dog` takes on the value `cat` or `dog`. The likelihood of observing each outcome does not change over time, in the same way that $p(\text{heads})$ for a fair coin is always $0.5$.
- `red or green or blue` takes on the value `red` or `green` or `blue`. The likelihood of observing each outcome does not change over time, in the same way that the probability of rolling a given number from fair die is always $\frac{1}{6}$.

For clarity, each one of these assumptions is utterly banal. *Can we use them nonetheless to select probability distributions for our random variables?*

lagrange multipliers

### Maximum entropy distributions
Consider another continuous-valued random variable: "Uber's yearly profit." Like `temperature`, it also has an underlying true mean $\mu \in (-\infty, \infty)$ and variance $\sigma^2 \in (-\infty, \infty)$ Trivially, the respective means and variances will be different. Assume we observe 10 values of each that look as follows:

| uber | temperature |
|------|-------------|
| -100 | -50         |
| -80  | 5           |
| -20  | 56          |
| 5    | 65          |
| 15   | 62          |
| -10  | 63          |
| 22   | 60          |
| 12   | 78          |
| 70   | 100         |
| 100  | -43         |

Plotting, we get the following:

![](../figures/temperature_random_variable.png?)
![](../figures/uber_random_variable.png?)

definition: “the distribution that can happen the most ways is also the distribution with the biggest information entropy. the distribution with the biggest entropy is the most conservative distribution that obeys its constraints”
the maximum entropy distribution

choosing the distribution with the largest entropy means spreading probability as evenly as possible, while still remaining consistent with anything we think we know about a process

the distribution that can happen the most number of ways is the most plausible distribution, a.k.a. the maximum entropy distribution


## Functional form

exponential family distributions
my last softmax post


## Loss function



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
