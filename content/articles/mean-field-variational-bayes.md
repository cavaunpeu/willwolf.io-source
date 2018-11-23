Title: Foo Bar
Date: 2018-11-23 10:00
Author: Will Wolf
Lang: en
Slug: mean-field-variational-bayes
Status: published
Summary:
Image: figures/.png

# mean field algorithm

$$
\log{p(\mathbf{X}\vert\theta)} = \mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z}\vert\theta)}{q(\mathbf{Z})}}\bigg] + \text{KL}\big(q(\mathbf{Z})\Vert p(\mathbf{Z}\vert\mathbf{X}, \theta)\big)
$$

## thoughts
- is the marginal log-likelihood of X fixed? is all we can do just minimize the KL divergence (by maximizing the ELBO?)
- we have to introduce factorized distributions first

## derivation

dropping the dependence on theta...

$$
\begin{align*}
\mathop{\mathbb{E}}_{q(\mathbf{Z})}\bigg[\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}\bigg]
&= \int{q(\mathbf{Z})\log{\frac{p(\mathbf{X, Z})}{q(\mathbf{Z})}}}d\mathbf{Z}\\
&= \int{q(\mathbf{Z})\log{p(\mathbf{X, Z})}}d\mathbf{Z} - \int{q(\mathbf{Z})\log{q(\mathbf{Z})}}d\mathbf{Z}\\
&= A + B
\end{align*}
$$

we'd like to extract the j'th factor

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
