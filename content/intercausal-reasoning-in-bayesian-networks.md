Title: Intercausal Reasoning in Bayesian Networks
Date: 2017-03-13 15:14
Author: Will Wolf
Category: Uncategorized
Slug: intercausal-reasoning-in-bayesian-networks
Status: published

The work for this post is contained in the following [Jupyter notebook](http://nbviewer.jupyter.org/github/cavaunpeu/intercausal-reasoning/blob/master/intercausal_reasoning.ipynb). Below is a brief introduction of what's inside.

I'm currently taking a course on [probabilistic graphical models](https://www.coursera.org/learn/probabilistic-graphical-models) in which we algebraically compute conditional probability estimates for simple Bayesian networks given ground-truth probabilities of the component parts. Since this is rather unrealistic in the real world, I sought to explore the same question with observational data instead.

This work contains an example of intercausal reasoning in Bayesian networks. Given the following toy network, where both the "president being in town" and a "car accident on the highway" exert influence over whether a traffic jam occurs, we'd like to compute the following probabilities: 

$$P(\text{Accident = 1}\ |\ \text{Traffic = 1})$$

$$P(\text{Accident = 1}\ |\ \text{Traffic = 1}, \text{President = 1})$$

Answering these questions with given, ground-truth probabilities is an exercise in the factorization of graphical models. Conversely, using observational data asks us to first estimate these probabilities themselves - as distributions, ideally - before pressing forward with the final plug-and-play.

Once more, this work can be found in the notebook linked above. The accompanying repository can be found [here](https://github.com/cavaunpeu/intercausal-reasoning).

![]({filename}figures/simple_bayesian_network.png)

Your thoughts are welcome. Many thanks for reading.
