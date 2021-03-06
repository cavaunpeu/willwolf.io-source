Title: Recurrent Neural Network Gradients, and Lessons Learned Therein
Date: 2016-10-18 14:00
Author: Will Wolf
Lang: en
Slug: recurrent-neural-network-gradients-and-lessons-learned-therein
Status: published
Summary: Recurrent neural network gradients by hand.
Image: images/rnn_gradient.png

I've spent the last week hand-rolling recurrent neural networks. I'm currently taking Udacity's Deep Learning [course](https://www.udacity.com/course/deep-learning--ud730), and arriving at the section on RNN's and LSTM's, I decided to build a few for myself.

### What are RNN's?

On the outside, recurrent neural networks differ from typical, feedforward neural networks in that they take a *sequence* of input instead of an input of fixed length. Concretely, imagine we are training a sentiment classifier on a bunch of tweets. To embed these tweets in vector space, we create a bag-of-words model with vocabulary size 3. In a typical neural network, this implies an input layer of size 3; an input could be $[4, 9, 3]$, or $[1, 0, 5]$, or $[0, 0, 6]$, for example. In a recurrent neural network, our input layer has the same size 3, but instead of just a single size-3 input, we can feed it a sequence of size-3 inputs of any length. For example, an input could be $[[1, 8, 5], [2, 2, 4]]$, or $[[6, 7, 3], [6, 2, 4], [9, 17, 5]]$, or $[[2, 3, 0], [1, 1, 7], [5, 5, 3], [8, 18, 4]]$.

On the inside, recurrent neural networks have a different feedforward mechanism than typical neural networks. In addition, each input in our sequence of inputs is processed individually and chronologically: the first input is fed forward, then the second, and so on. Finally, after all inputs have been fed forward, we compute some gradients and update our weights. Like in feedforward networks, we also use backpropagation. However, we must now backpropagate errors to our parameters at every step in time. In other words, we must compute gradients with respect to: the state of the world when we fed our first input forward, the state of the world when we fed our second input forward, and up until the state of the world when we fed our last input forward. This algorithm is called [Backpropagation Through Time](https://en.wikipedia.org/wiki/Backpropagation_through_time).

### Other Resources, My Frustrations

There are many resources for understanding how to compute gradients using Backpropagation Through Time. In my view, [Recurrent Neural Networks Maths](https://www.existor.com/en/ml-rnn.html) is the most mathematically comprehensive, while [Recurrent Neural Networks Tutorial Part 3](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) is more concise yet equally clear. Finally, there exists Andrej Karpathy's [Minimal character-level language model](https://gist.github.com/karpathy/d4dee566867f8291f086), accompanying his excellent [blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) on the general theory and use of RNN's, which I initially found convoluted and hard to understand.

In all posts, I think the authors unfortunately blur the line between the derivation of the gradients and their (efficient) implementation in code, or at the very least jump too quickly from one to another. They define variables like `dbnext`,  `delta_t`, and $e_{hi}^{2f3}$ without thoroughly explaining their place in the analytical gradients themselves. As one example, the first post includes the snippet:

> $$
\frac{\partial J^{t=2}}{\partial w^{xh}_{mi}} =
e^{t=2f2}_{hi} \frac{\partial h^{t=2}_i}{\partial z^{t=2}_{hi}} \frac{\partial z^{t=2}_{hi}}{\partial w^{xh}_{mi}} +
e^{t=1f2}_{hi} \frac{\partial h^{t=1}_i} {\partial z^{t=1}_{hi}} \frac{\partial z^{t=1}_{hi}}{\partial w^{xh}_{mi}}
$$

So far, he's just talking about analytical gradients. Next, he gives hint to the implementation-in-code that follows.

> So the thing to note is that we can delay adding in the backward propagated errors until we get further into the loop. In other words, we can initially compute the derivatives of *J* with respect to the third unrolled network with only the first term:
>
> $$
\frac{\partial J^{t=3}}{\partial w^{xh}_{mi}} =
e^{t=3f3}_{hi} \frac{\partial h^{t=3}_i}{\partial z^{t=3}_{hi}} \frac{\partial z^{t=3}_{hi}}{\partial w^{xh}_{mi}}
$$
>
> And then add in the other term only when we get to the second unrolled
network:
>
> $$
\frac{\partial J^{t=2}}{\partial w^{xh}_{mi}} =
(e^{t=2f3}_{hi} + e^{t=2f2}_{hi}) \frac{\partial h^{t=2}_i}{\partial z^{t=2}_{hi}}
\frac{\partial z^{t=2}_{hi}}
{\partial w^{xh}_{mi}}
$$

Note the opposing definitions of the variable $\frac{\partial J^{t=2}}{\partial w^{xh}_{mi}}$. As far as I know, the latter is, in a vacuum, categorically false. This said, I believe the author is simply providing an alternative definition of this quantity in line with a computational shortcut he later takes.

Of course, these ambiguities become very emotional, very quickly. I myself was confused for two days. As such, the aim of this post is to derive recurrent neural network gradients from scratch, and emphatically clarify that all implementation "shortcuts" thereafter are nothing more than just that, with no real bearing on the analytical gradients themselves. In other words, if you can derive the gradients, you win. Write a unit test, code these gradients in the crudest way you can, watch your test pass, and then immediately realize that your code can be made more efficient. At this point, all "shortcuts" that the above authors (and myself, now, as well) take in their code will make perfect sense.

### Backpropagation Through Time

In the simplest case, let's assume our network has 3 layers, and just 3 parameters to optimize: $\mathbf{W^{xh}}$, $\mathbf{W^{hh}}$ and $\mathbf{W^{hy}}$. The foundational equations of this network are as follows:

- $\mathbf{z_t} = \mathbf{W^{xh}}\mathbf{x} + \mathbf{W^{hh}}\mathbf{h_{t-1}}$
- $\mathbf{h_t} = \tanh(\mathbf{z_t})$
- $\mathbf{y_t} = \mathbf{W^{hy}}\mathbf{h_t}$
- $\mathbf{p_t} = \text{softmax}(\mathbf{y_t})$
- $\mathbf{J_t} = \text{crossentropy}(\mathbf{p_t},
    \mathbf{\text{labels}_t})$

I've written "softmax" and "cross-entropy" for clarity: before tackling the math below, it is important to understand what they do, and how to derive their gradients by hand.

Before moving forward, let's restate the definition of a partial derivative itself.

> A partial derivative, for example $\frac{\partial y}{\partial x}$, measures how much $y$ increases with every 1-unit increase in $x$.

Our cost $\mathbf{J_t}$ is the *total* *cost* (i.e., not the average cost) of a given sequence of inputs. As such, a 1-unit increase in $\mathbf{W^{hy}}$ will impact each of $\mathbf{J_1}$, $\mathbf{J_2}$ and $\mathbf{J_3}$ individually. Therefore, our gradient is equal to the sum of the respective gradients at each time step $t$:

$$
\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hy}}} =
\sum\limits_t \frac{\partial \mathbf{J_t}}{\partial
\mathbf{W^{hy}}} = \frac{\partial \mathbf{J_3}}{\partial
\mathbf{W^{hy}}} + \frac{\partial \mathbf{J_2}}{\partial
\mathbf{W^{hy}}} + \frac{\partial \mathbf{J_1}}{\partial
\mathbf{W^{hy}}}\\
\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hh}}} =
\sum\limits_t \frac{\partial \mathbf{J_t}}{\partial
\mathbf{W^{hh}}} = \frac{\partial \mathbf{J_3}}{\partial
\mathbf{W^{hh}}} + \frac{\partial \mathbf{J_2}}{\partial
\mathbf{W^{hh}}} + \frac{\partial \mathbf{J_1}}{\partial
\mathbf{W^{hh}}}\\
\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{xh}}} =
\sum\limits_t \frac{\partial \mathbf{J_t}}{\partial
\mathbf{W^{xh}}} = \frac{\partial \mathbf{J_3}}{\partial
\mathbf{W^{xh}}} + \frac{\partial \mathbf{J_2}}{\partial
\mathbf{W^{xh}}} + \frac{\partial \mathbf{J_1}}{\partial
\mathbf{W^{xh}}}
$$

Let's take this piece by piece.

### Algebraic Derivatives
<br>
#### $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hy}}}$:

Starting with $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hy}}}$, we note that a change in $\mathbf{W^{hy}}$ will only impact $\mathbf{J_3}$ at time $t=3$: $\mathbf{W^{hy}}$ plays no role in computing the value of anything other than $\mathbf{y_3}$. Therefore:

$$
\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hy}}} =
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{W^{hy}}}\\
\frac{\partial \mathbf{J_2}}{\partial \mathbf{W^{hy}}} =
\frac{\partial \mathbf{J_2}}{\partial \mathbf{p_2}}
\frac{\partial \mathbf{p_2}}{\partial \mathbf{y_2}}
\frac{\partial \mathbf{y_2}}{\partial \mathbf{W^{hy}}}\\
\frac{\partial \mathbf{J_1}}{\partial \mathbf{W^{hy}}} =
\frac{\partial \mathbf{J_1}}{\partial \mathbf{p_1}}
\frac{\partial \mathbf{p_1}}{\partial \mathbf{y_1}}
\frac{\partial \mathbf{y_1}}{\partial \mathbf{W^{hy}}}\\
$$

#### $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hh}}}$:

Starting with $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hh}}}$, a change in $\mathbf{W^{hh}}$ will impact our cost $\mathbf{J_3}$ in *3 separate ways:* once, when computing the value of $\mathbf{h_1}$; once, when computing the value of $\mathbf{h_2}$, which depends on $\mathbf{h_1}$; once, when computing the value of $\mathbf{h_3}$, which depends on $\mathbf{h_2}$, which depends on $\mathbf{h_1}$.

More generally, a change in $\mathbf{W^{hh}}$ will impact our cost $\mathbf{J_t}$ on $t$ separate occasions. Therefore:

$$
\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hh}}} =
\sum\limits_{k=0}^{t} \frac{\partial \mathbf{J_t}}{\partial
\mathbf{h_t}} \frac{\partial \mathbf{h_t}}{\partial
\mathbf{h_k}} \frac{\partial \mathbf{h_k}}{\partial
\mathbf{z_k}} \frac{\partial \mathbf{z_k}}{\partial
\mathbf{W^{hh}}}
$$

Then, with this definition, we compute our individual gradients as:

$$
\begin{align*}
\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hh}}} &=
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{h_3}}
\frac{\partial \mathbf{h_3}}{\partial \mathbf{z_3}}
\frac{\partial \mathbf{z_3}}{\partial \mathbf{W^{hh}}}\\ &+
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{h_3}}
\frac{\partial \mathbf{h_3}}{\partial \mathbf{z_3}}
\frac{\partial \mathbf{z_3}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{W^{hh}}}\\ &+
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{h_3}}
\frac{\partial \mathbf{h_3}}{\partial \mathbf{z_3}}
\frac{\partial \mathbf{z_3}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{h_1}}
\frac{\partial \mathbf{h_1}}{\partial \mathbf{z_1}}
\frac{\partial \mathbf{z_1}}{\partial \mathbf{W^{hh}}}\\
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathbf{J_2}}{\partial \mathbf{W^{hh}}} &=
\frac{\partial \mathbf{J_2}}{\partial \mathbf{p_2}}
\frac{\partial \mathbf{p_2}}{\partial \mathbf{y_2}}
\frac{\partial \mathbf{y_2}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{W^{hh}}}\\ &+
\frac{\partial \mathbf{J_2}}{\partial \mathbf{p_2}}
\frac{\partial \mathbf{p_2}}{\partial \mathbf{y_2}}
\frac{\partial \mathbf{y_2}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{h_1}}
\frac{\partial \mathbf{h_1}}{\partial \mathbf{z_1}}
\frac{\partial \mathbf{z_1}}{\partial \mathbf{W^{hh}}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathbf{J_1}}{\partial \mathbf{W^{hh}}} &=
\frac{\partial \mathbf{J_1}}{\partial \mathbf{p_1}}
\frac{\partial \mathbf{p_1}}{\partial \mathbf{y_1}}
\frac{\partial \mathbf{y_1}}{\partial \mathbf{h_1}}
\frac{\partial \mathbf{h_1}}{\partial \mathbf{z_1}}
\frac{\partial \mathbf{z_1}}{\partial \mathbf{W^{hh}}}
\end{align*}
$$

#### $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{xh}}}$:

Similarly:

$$
\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{xh}}} =
\sum\limits_{k=0}^{t} \frac{\partial \mathbf{J_t}}{\partial
\mathbf{h_t}} \frac{\partial \mathbf{h_t}}{\partial
\mathbf{h_k}} \frac{\partial \mathbf{h_k}}{\partial
\mathbf{z_k}} \frac{\partial \mathbf{z_k}}{\partial
\mathbf{W^{xh}}}
$$

Therefore:

$$
\begin{align*}
\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{xh}}} &=
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{h_3}}
\frac{\partial \mathbf{h_3}}{\partial \mathbf{z_3}}
\frac{\partial \mathbf{z_3}}{\partial \mathbf{W^{xh}}}\\ &+
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{h_3}}
\frac{\partial \mathbf{h_3}}{\partial \mathbf{z_3}}
\frac{\partial \mathbf{z_3}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{W^{xh}}}\\ &+
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}
\frac{\partial \mathbf{y_3}}{\partial \mathbf{h_3}}
\frac{\partial \mathbf{h_3}}{\partial \mathbf{z_3}}
\frac{\partial \mathbf{z_3}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{h_1}}
\frac{\partial \mathbf{h_1}}{\partial \mathbf{z_1}}
\frac{\partial \mathbf{z_1}}{\partial \mathbf{W^{xh}}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathbf{J_2}}{\partial \mathbf{W^{xh}}} &=
\frac{\partial \mathbf{J_2}}{\partial \mathbf{p_2}}
\frac{\partial \mathbf{p_2}}{\partial \mathbf{y_2}}
\frac{\partial \mathbf{y_2}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{W^{xh}}}\\ &+
\frac{\partial \mathbf{J_2}}{\partial \mathbf{p_2}}
\frac{\partial \mathbf{p_2}}{\partial \mathbf{y_2}}
\frac{\partial \mathbf{y_2}}{\partial \mathbf{h_2}}
\frac{\partial \mathbf{h_2}}{\partial \mathbf{z_2}}
\frac{\partial \mathbf{z_2}}{\partial \mathbf{h_1}}
\frac{\partial \mathbf{h_1}}{\partial \mathbf{z_1}}
\frac{\partial \mathbf{z_1}}{\partial \mathbf{W^{xh}}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathbf{J_1}}{\partial \mathbf{W^{xh}}} &=
\frac{\partial \mathbf{J_1}}{\partial \mathbf{p_1}}
\frac{\partial \mathbf{p_1}}{\partial \mathbf{y_1}}
\frac{\partial \mathbf{y_1}}{\partial \mathbf{h_1}}
\frac{\partial \mathbf{h_1}}{\partial \mathbf{z_1}}
\frac{\partial \mathbf{z_1}}{\partial \mathbf{W^{xh}}}
\end{align*}
$$

### Analytical Derivatives

Finally, we plug in the individual partial derivates to compute our final gradients, where:

- $\frac{\partial \mathbf{J_t}}{\partial \mathbf{y_t}} = \mathbf{p_t} - \mathbf{\text{labels}_t}$, where $\mathbf{\text{labels}_t}$ is a one-hot vector of the correct answer at a given time-step $t$
- $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hy}}} = (\mathbf{p_t} - \mathbf{\text{labels}_t})\mathbf{h_t}$
- $\frac{\partial \mathbf{J_t}}{\partial \mathbf{h_t}} = (\mathbf{p_t} - \mathbf{\text{labels}_t})\mathbf{W^{hy}}$
- $\frac{\partial \mathbf{h_t}}{\partial \mathbf{z_t}} = 1 - \tanh^2(\mathbf{z_t}) = 1 - \mathbf{h_t}^2$, as $\mathbf{h_t} = \tanh(\mathbf{z_t})$
- $\frac{\partial \mathbf{z_t}}{\mathbf{h_{t-1}}} = \mathbf{W^{hh}}$
- $\frac{\partial \mathbf{z_t}}{\partial \mathbf{W^{xh}}} = \mathbf{x_t}$
- $\frac{z_t}{\partial \mathbf{W^{hh}}} = \mathbf{h_{t-1}}$

At this point, you're done: you've computed your gradients, and you understand Backpropagation Through Time. From this point forward, all that's left is writing some for-loops.

### Implementation Shortcuts

As you'll readily note, when computing the gradient for, for example, $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{xh}}}$, we'll need access to our labels at time-steps $t=3$, $t=2$ and $t=1$. For $\frac{\partial \mathbf{J_2}}{\partial \mathbf{W^{xh}}}$, we'll need our labels at time-steps $t=2$ and $t=1$. Finally, for $\frac{\partial \mathbf{J_1}}{\partial \mathbf{W^{xh}}}$, we'll need our labels at just $t=1$. Naturally, we look to make this efficient: for, for example, $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{xh}}}$, how about just compute the $t=3$ parts at $t=3$, and add in the rest at $t=2$? Instead of explaining further, I leave this step to you: it is ultimately trivial, a good exercise, and when you're finished, you'll find that your code readily resembles much of that written in the above resources.

### Lessons Learned

Throughout this process, I learned a few lessons.

1. When implementing neural networks from scratch, derive gradients by hand at the outset. *This makes thing so much easier.*
2. Turn more readily to your pencil and paper before writing a single line of code. They are not scary and they absolutely have their place.
3. The chain rule remains simple and clear. If a derivative seems to "supercede" the general difficulty of the chain rule, there's probably something else you're missing.

Happy RNN's.

---

Key references for this article include:

- [Recurrent Neural Networks Tutorial Part 2 Implementing A Rnn With Python Numpy And Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
- [Recurrent Neural Networks Tutorial Part 3 Backpropagation Through Time And Vanishing Gradients](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086)
- [Machine Learning - Recurrent Neural Networks Maths](https://www.existor.com/en/ml-rnn.html)
