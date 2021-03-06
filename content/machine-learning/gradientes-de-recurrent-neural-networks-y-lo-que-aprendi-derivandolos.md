Title: Gradientes de Recurrent Neural Networks y Lo Que Aprendí Derivándolos
Date: 2016-10-18 14:00
Author: Will Wolf
Lang: es
Url: 2016/10/18/gradientes-de-recurrent-neural-networks-y-lo-que-aprendi-derivandolos/
Save_as: 2016/10/18/gradientes-de-recurrent-neural-networks-y-lo-que-aprendi-derivandolos/index.html
Slug: gradientes-de-recurrent-neural-networks-y-lo-que-aprendi-derivandolos
Status: published
Summary: Gradientes de recurrent neural networks a mano.
Image: ../images/rnn_gradient.png

He pasado la mayoría de la última semana construyendo recurrent neural networks a mano. Estoy tomando el curso de [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730), y, llegando al contenido sobre RNN's y LSTM's, decidí construir algunos de ellos desde cero yo mismo.

### ¿Qué es un RNN?

Por afuera, las recurrent neural networks se diferencian del feedforward neural network típico por el hecho de que pueden ingerir *una secuencia* de input en lugar de un sólo input de largo fijo. Concretamente, imagínense que estamos entrenando un modelo de clasificación con un puñado de tuits. Para codificar dichos tuits en el espacio vectorial, creamos un modelo de bag-of-words con un vocabulario de 3 palabras distintas. En el neural network clásico, esto implica un "input layer" con un tamaño de 3; un input podría ser $[4, 9, 3]$, o $[1, 0, 5]$, o $[0, 0, 6]$, por ejemplo. En el caso del recurrent neural network, nuestro input layer tiene el mismo tamaño de 3, pero en lugar de un sólo input, le podemos alimentar una secuencia de inputs de tamaño 3 de cualquier largo. Como ejemplo, un input podría ser $[[1, 8, 5], [2, 2, 4]]$, o $[[6, 7, 3], [6, 2, 4], [9, 17, 5]]$, o $[[2, 3, 0], [1, 1, 7], [5, 5, 3], [8, 18, 4]]$.

En su interior, las recurrent neural networks tienen un mecanismo feedforward diferente del neural network típico. Además, cada input en nuestra secuencia se procesa individual y cronológicamente: el primer input es procesado, luego el segundo, hasta procesar el último. Por fin, después de procesar todos los inputs, computamos algunos gradientes y actualizamos los parámetros (weights) de la red. Tal como en los feedforward networks, lo hacemos con backpropagation. Al contrario, ya nos toca propagarles los errores a cada parámetro en cada etapa del tiempo. Dicho de otra manera, nos toca calcular gradientes con respecto a: el estado del mundo al procesar nuestro primer input, el estado del mundo al procesar nuestro segundo input, hasta el en el que procesamos nuestro último input. Este algoritmo se llama [Backpropagation Through Time](https://en.wikipedia.org/wiki/Backpropagation_through_time).

### Otros Recursos, Mis Frustraciones

Existen bastantes recursos para entender cómo calcular los gradientes usando el Backpropagation Through Time. En mi opinión, [Recurrent Neural Networks Maths](https://www.existor.com/en/ml-rnn.html) es el más comprehensivo en un sentido matemático, mientras [Recurrent Neural Networks Tutorial Part 3](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) es más conciso pero igual de claro. Finalmente, está [Minimal character-level language model](https://gist.github.com/karpathy/d4dee566867f8291f086) por Andrej Karpathy, acompañando su [blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) excelente sobre la teoría y el uso general de los RNN's, que al inicio me costó mucho entender.

En todos los posts, pienso que los autores desafortunadamente no aclaran muy bien la línea divisoria entre la derivación de los gradientes y su implementación (eficiente) en código, o por lo menos brincan demasiado rápido del primero al segundo. Definen variables como  `dbnext`,  `delta_t`, y $e_{hi}^{2f3}$ sin explicar cabalmente su significado en los gradientes analíticos mismos. Como ejemplo, el primer post incluye la siguiente sección:

> $$
\frac{\partial J^{t=2}}{\partial w^{xh}_{mi}} =
e^{t=2f2}_{hi} \frac{\partial h^{t=2}_i}{\partial z^{t=2}_{hi}} \frac{\partial z^{t=2}_{hi}}{\partial w^{xh}_{mi}} +
e^{t=1f2}_{hi} \frac{\partial h^{t=1}_i} {\partial z^{t=1}_{hi}} \frac{\partial z^{t=1}_{hi}}{\partial w^{xh}_{mi}}
$$

Hasta ahora, no está hablando sino de los gradientes analíticos. A continuación, alude a la implementación-en-código que sigue.

> So the thing to note is that we can delay adding in the backward propagated errors until we get further into the loop. In other words, we can initially compute the derivatives of *J* with respect to the third unrolled network with only the first term:
>
> $$
\frac{\partial J^{t=3}}{\partial w^{xh}_{mi}} =
e^{t=3f3}_{hi} \frac{\partial h^{t=3}_i}{\partial z^{t=3}_{hi}} \frac{\partial z^{t=3}_{hi}}{\partial w^{xh}_{mi}}
$$
>
> And then add in the other term only when we get to the second unrolled network:
>
> $$
\frac{\partial J^{t=2}}{\partial w^{xh}_{mi}} =
(e^{t=2f3}_{hi} + e^{t=2f2}_{hi}) \frac{\partial h^{t=2}_i}{\partial z^{t=2}_{hi}}
\frac{\partial z^{t=2}_{hi}}
{\partial w^{xh}_{mi}}
$$

Noten las definiciones opuestas de la variable $\frac{\partial J^{t=2}}{\partial w^{xh}_{mi}}$. Hasta donde yo sé, la segunda es, sin hacerle caso a algún posible código, categóricamente falsa. Dicho eso, creo que el autor está simplemente ofreciendo una definición alternativa para esta cantidad en cuanto a un atajo pequeño que luego tome.

Sobre decir que estas ambigüedades hacen que todo se vuelva muy emocional, muy rápido. Me dejaron confundido durante dos días. Por lo tanto, el objetivo de este post es derivar los gradientes de un recurrent neural network desde cero, y clarificar enfáticamente que cualquier atajo de implementación que siga no es nada más que ese mismo, y que no tiene nada que ver con la definición analítica del gradiente correspondiente. En otras palabras, si puedes derivar los gradientes, has ganado. Escribe un test unitario, implementa dichos gradientes de la manera más cruda posible, velo pasar, y enseguida te darás cuenta de que puedes hacer tu código mucho más eficiente con muy poco esfuerzo. A esa altura, todos los "atajos" que tomen los autores ya mencionados te van a parecer absolutamente obvios.

### Backpropagation Through Time

En el caso más simple, asumamos que nuestra red tiene 3 capas, y tan sólo 3 parámetros para optimizar: $\mathbf{W^{xh}}$, $\mathbf{W^{hh}}$ y $\mathbf{W^{hy}}$. Las ecuaciones principales son las siguientes:

- $\mathbf{z_t} = \mathbf{W^{xh}}\mathbf{x} + \mathbf{W^{hh}}\mathbf{h_{t-1}}$
- $\mathbf{h_t} = \tanh(\mathbf{z_t})$
- $\mathbf{y_t} = \mathbf{W^{hy}}\mathbf{h_t}$
- $\mathbf{p_t} = \text{softmax}(\mathbf{y_t})$
- $\mathbf{J_t} = \text{crossentropy}(\mathbf{p_t}, \mathbf{\text{labels}})$

He escrito "softmax" y "cross-entropy" por cuestiones de claridad: antes de emprender lo siguiente, es crucial entender lo que hacen y cómo calcular sus gradientes a mano.

Antes de avanzar, damos la definición de una derivada parcial misma:

> Una derivada parcial, $\frac{\partial y}{\partial x}$ por ejemplo, nos dice cuánto crece $y$ a consecuencia de un cambio en $x$.

Nuestro costo $\mathbf{J_t}$ es el costo *total* (no el costo promedio) de una cierta secuencia de inputs. Por eso, un cambio de una unidad en $\mathbf{W^{hy}}$ impacta a $\mathbf{J_1}$, $\mathbf{J_2}$ y $\mathbf{J_3}$ por separado. En consecuencia, nuestro gradiente equivale a la suma de los gradientes respectivos en cada etapa de tiempo $t$:

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
\mathbf{W^{xh}}}$$

Tomémoslo pasa a paso.

### Derivadas Algebraicas
<br>
#### $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hy}}}$:

Empezando con$\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hy}}}$, notamos que un cambio en $\mathbf{W^{hy}}$ impacta a $\mathbf{J_3}$ sólo cuando $t=3$, y no a ninguna otra cantidad. Sigue que:

$$
\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hy}}} =
\frac{\partial \mathbf{J_3}}{\partial \mathbf{p_3}}
\frac{\partial \mathbf{p_3}}{\partial \mathbf{y_3}}\frac{\partial \mathbf{y_3}}{\partial \mathbf{W^{hy}}}\\
\frac{\partial \mathbf{J_2}}{\partial \mathbf{W^{hy}}} =
\frac{\partial \mathbf{J_2}}{\partial \mathbf{p_2}}
\frac{\partial \mathbf{p_2}}{\partial \mathbf{y_2}}\frac{\partial \mathbf{y_2}}{\partial \mathbf{W^{hy}}}\\
\frac{\partial \mathbf{J_1}}{\partial \mathbf{W^{hy}}} =
\frac{\partial \mathbf{J_1}}{\partial \mathbf{p_1}}
\frac{\partial \mathbf{p_1}}{\partial \mathbf{y_1}}\frac{\partial \mathbf{y_1}}{\partial \mathbf{W^{hy}}}
$$

#### $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hh}}}$:

Empezando con $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{hh}}}$, un cambio en $\mathbf{W^{hh}}$ impacta a nuestro costo en *3 momentos distintos:* por primera vez al calcular el valor de $\mathbf{h_1}$; por segunda vez al calcular el valor de $\mathbf{h_2}$, que está condicionado a $\mathbf{h_1}$; por tercera vez al calcular $\mathbf{h_3}$, que está condicionado a $\mathbf{h_2}$, que está condicionado a $\mathbf{h_1}$.

En términos más generales, un cambio en $\mathbf{W^{hh}}$ impacta al costo $\mathbf{J_t}$ en $t$ momentos distintos. Sigue que:

$$
\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hh}}} =
\sum\limits_{k=0}^{t} \frac{\partial \mathbf{J_t}}{\partial
\mathbf{h_t}} \frac{\partial \mathbf{h_t}}{\partial
\mathbf{h_k}} \frac{\partial \mathbf{h_k}}{\partial
\mathbf{z_k}} \frac{\partial \mathbf{z_k}}{\partial
\mathbf{W^{hh}}}
$$

Con esta definición, calculamos nuestras gradientes como:

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

Análogamente:

$$\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{xh}}} =
\sum\limits_{k=0}^{t} \frac{\partial \mathbf{J_t}}{\partial
\mathbf{h_t}} \frac{\partial \mathbf{h_t}}{\partial
\mathbf{h_k}} \frac{\partial \mathbf{h_k}}{\partial
\mathbf{z_k}} \frac{\partial \mathbf{z_k}}{\partial
\mathbf{W^{xh}}}$$

Así que:

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

### Derivadas Analíticas

Finalmente, insertamos las derivadas parciales individuales para llegar a los gradientes finales con lo siguiente en mano:

- $\frac{\partial \mathbf{J_t}}{\partial y} = \mathbf{p_t} - \mathbf{\text{labels}_t}$, where $\mathbf{\text{labels}_t}$ is a one-hot vector of the correct answer at a given time-step $t$
- $\frac{\partial \mathbf{J_t}}{\partial \mathbf{W^{hy}}} = (\mathbf{p_t} - \mathbf{\text{labels}_t})\mathbf{h_t}$
- $\frac{\partial \mathbf{J_t}}{\partial \mathbf{h_t}} = (\mathbf{p_t} - \mathbf{\text{labels}_t})\mathbf{W^{hy}}$
- $\frac{\partial \mathbf{h_t}}{\partial \mathbf{z_t}} = 1 - \tanh^2(\mathbf{z_t}) = 1 - \mathbf{h_t}^2$, as $\mathbf{h_t} = \tanh(\mathbf{z_t})$
- $\frac{\partial \mathbf{z_t}}{\mathbf{h_{t-1}}} = \mathbf{W^{hh}}$
- $\frac{\partial \mathbf{z_t}}{\partial \mathbf{W^{xh}}} = \mathbf{x_t}$
- $\frac{z_t}{\partial \mathbf{W^{hh}}} = \mathbf{h_{t-1}}$

A esta altura, has terminado: has calculado tus gradientes, y entiendes bien el algoritmo de Backpropagation Through Time. De aquí en adelante, lo único que queda es escribir algunos for-loops.

### Atajos de Implementación

Al calcular le gradiente de, por ejemplo, $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{xh}}}$, se nota de inmediato que necesitamos acceso a los labels de $t=3$, $t=2$ y $t=1$. Para $\frac{\partial \mathbf{J_2}}{\partial \mathbf{W^{xh}}}$, necesitamos acceso a los labels de $t=2$ y $t=1$. Por fin, para $\frac{\partial \mathbf{J_1}}{\partial \mathbf{W^{xh}}}$, necesitamos los labels de $t=1$. Naturalmente, nos preguntamos cómo podemos hacer este proceso más eficiente: por ejemplo, para calcular $\frac{\partial \mathbf{J_3}}{\partial \mathbf{W^{xh}}}$, ¿qué tal sólo calcular las partes de $t=3$ a $t=3$, y agregarle el resto en los pasos del tiempo que siguen? En lugar de profundizar, te los dejo a ustedes: esta parte es trivial en el fondo, un buen ejercicio para el practicante, y al acabar vas a descubrir de repente que tu código se parece bastante al de los recursos arriba.

### Aprendizajes del Proceso

Mediante este proceso, aprendí varias cosas claves:

1. Al querer implementar un neural network desde cero, deriva las gradientes a mano al inicio. *Esto hace que todo salga mucho más fácil.*
2. Usa más el lápiz y papel antes de siquiera escribir una sola linea de código. No dan miedo y tienen absolutamente su función.
3. El "chain rule" queda simple y claro. Si una derivada parece estar fuera de esta dificultad general, es probable que haya otro detalle importante que te falta reconocer.

Felices RNN's.

---
Referencias claves para este artículo se nombran:

- [Recurrent Neural Networks Tutorial Part 2 Implementing A Rnn With Python Numpy And Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
- [Recurrent Neural Networks Tutorial Part 3 Backpropagation Through Time And Vanishing Gradients](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086)
- [Machine Learning - Recurrent Neural Networks Maths](https://www.existor.com/en/ml-rnn.html)
