Title:
Date: 2018-10-29 22:00
Author: Will Wolf
Lang: en
Slug: additional-strategies-partition-function
Status: published
Summary:
Image: figures/em-for-lda/.png

# we want to maximize

$$
\log{p(\mathbf{X}\vert\theta)} = \log{\sum_{\mathbf{Z}}p(\mathbf{X, Z}\vert\theta)}
$$

# however, this is hard, so we go for

$$
\begin{align*}
\log{p(\mathbf{X}\vert\theta)}
&= \log{\sum_{\mathbf{Z}}p(\mathbf{X, Z}\vert\theta)}\\
&= \log{\sum_{\mathbf{Z}}q(\mathbf{Z})\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\\
&= \log{\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\bigg]\
\end{align*}
$$

## in the convex case, jensen's inequality says that the expectation of a function is greater than the function of the expectation. in concave case, it's the opposite.

$$
\begin{align*} \log{p(\mathbf{X}\vert\theta)} = \log{\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}\bigg]}
&\geq \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\bigg] + R
\end{align*}
$$

# so, what's R?

$$
\begin{align*}
R
&= \log{p(\mathbf{X}\vert\theta)} -  \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\bigg]\\
&= \log{p(\mathbf{X}\vert\theta)} -  \sum_{\mathbf{Z}}q(\mathbf{Z})\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}
\end{align*}
$$

# as an aside

$$
\begin{align*}
p(\mathbf{Z}\vert\mathbf{X}, \theta) &= \frac{p(\mathbf{X}, \mathbf{Z}\vert\theta)}{p(\mathbf{X}\vert\theta)}\\
\log{p(\mathbf{Z}\vert\mathbf{X}, \theta)}
&= \log{p(\mathbf{X}, \mathbf{Z}\vert\theta)} - \log{p(\mathbf{X}\vert\theta)}\\
\log{p(\mathbf{X}, \mathbf{Z}\vert\theta)} &= \log{p(\mathbf{Z}\vert\mathbf{X}, \theta)} + \log{p(\mathbf{X}\vert\theta)}
\end{align*}
$$

# continuing

$$
\begin{align*}
R
&= \log{p(\mathbf{X}\vert\theta)} -  \sum_{\mathbf{Z}}q(\mathbf{Z})\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\\
&= \log{p(\mathbf{X}\vert\theta)} -  \sum_{\mathbf{Z}}q(\mathbf{Z})\bigg(\log{p(\mathbf{X, Z}\vert\theta) - \log{q(\mathbf{Z})}}\bigg)\\
&= \log{p(\mathbf{X}\vert\theta)} -  \sum_{\mathbf{Z}}q(\mathbf{Z})\bigg(\log{p(\mathbf{Z}\vert\mathbf{X}, \theta)} + \log{p(\mathbf{X}\vert\theta)} - \log{q(\mathbf{Z})}\bigg)\\
&= \sum_{\mathbf{Z}}q(\mathbf{Z})\log{p(\mathbf{X}\vert\theta)} -  \sum_{\mathbf{Z}}q(\mathbf{Z})\bigg(\log{p(\mathbf{Z}\vert\mathbf{X}, \theta)} + \log{p(\mathbf{X}\vert\theta)} - \log{q(\mathbf{Z})}\bigg)\\
&= \sum_{\mathbf{Z}}q(\mathbf{Z})\bigg(\log{p(\mathbf{X}\vert\theta)} -  \big(\log{p(\mathbf{Z}\vert\mathbf{X}, \theta)} + \log{p(\mathbf{X}\vert\theta)} - \log{q(\mathbf{Z})}\big)\bigg)\\
&= \sum_{\mathbf{Z}}q(\mathbf{Z})-  \big(\log{p(\mathbf{Z}\vert\mathbf{X}, \theta)} - \log{q(\mathbf{Z})}\big)\\
&=
-\sum_{\mathbf{Z}}q(\mathbf{Z})\log{\frac{p(\mathbf{Z}\vert\mathbf{X}, \theta)}{q(\mathbf{Z})}}\\
&= \text{KL}\big(p(\mathbf{Z}\vert\mathbf{X}, \theta) \Vert q(\mathbf{Z})\big)\\
\end{align*}
$$

# putting this back together

$$
\log{p(\mathbf{X}\vert\theta)}
\geq \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\bigg] + \text{KL}\big(p(\mathbf{Z}\vert\mathbf{X}, \theta) \Vert q(\mathbf{Z})\big)
$$

# from here, EM is quite simple

show the pictures

Thanks for reading.

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
