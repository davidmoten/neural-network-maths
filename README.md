## The mathematics of feed-forward neural networks

With all the fuss about ChatGPT-3/4+ I decided it was time to get on top of what a (simple) neural network looks like, and work through the training calculations as well. I found backpropagation explanations to be somewhat vague and imprecise so I wrote this to sharpen things up a bit.

The neural network consists of an input vector and 1 or more layers. Each layer contains multiple nodes (a vector of calculated values), a matrix $W^l$ and a bias vector $b^l$ used to transform the nodes in the previous layer (or the input vector if the first layer) into the nodes of the current layer. The exact transformation from layer to layer of the vectors is described later.

Each layer $l$ has 
* a vector $a^l$ of node values to contain calculations as data flows forward through the network. $a^{0}$ is the input vector. ($a$ is short for activation).
* a matrix $W^l$ of weights (initially between -1 and 1) where the $(i,j)$ entry contains the weight for
 the edge between $a_i^{l-1}$ and $a_j^l$
* a vector $b^l$ that is the bias for the calculation with that layer
* an activation function $\sigma_l$ that is the last transformation of the node values vector

Common activation functions are:
* sigmoid: $sigmoid(x) = \frac{1}{1 + e^{-x}}$
* LeRU:  $leru(x) = x \text { if } x \gt 0 \text { else } 0$
* swish, or SiLU: $swish(x) = x * sigmoid(x)$
* GELU: $gelu(x) = \frac{x}{2}(1 + erf(\frac{x}{\sqrt{2}}))$ (about [erf](https://mathworld.wolfram.com/Erf.html))

A different activation function may be used for the final layer, especially for multiple classification (label) outputs:
* softmax: $\sigma(v) = \frac{e^{v_i}}{\Sigma_j e^{v_j}},\text{ v a vector}$
 
Note that when you see an activation function applied to a matrix (also a vector which is a matrix with one column) then the function is applied to every element of the matrix.
 
## Calculating output from input
Here's how a layer vector $a^l$ is pushed through the neural network to produce the next layer vector $a^{l+1}$:
 
 &emsp; $a^{l+1} = \sigma_{l+1}(W^{l+1}a^l + b^{l+1})$
 
So the next vector is just the sum of the product of weights with the previous vector values which is then fed into the activation function. One purpose of the activation function is to introduce *non-linearity* in the hope that intrinsically non-linear problems can be solved with the network.

## Training the network
We start by assigning random values between -1 and 1 to the weight matrices. 

A training set is a set of input and expected result vector pairs $(v_1, e)$. We use those pairs (repeatedly even) and differential calculus to move the weights and biases so that the network outputs move closer and closer to the expected results. We will use a method called Gradient Descent to do this. 
 
We define a function that we will minimize that when minimized represents a close match of output and desired output. That function is called the *cost function*. Here is one possibility which is the 1/2 of the square of the distance between the actual and expected vectors:
 
&emsp; $C = \frac{1}{2}(a^{L}-e)\cdot(a^{L}-e)$
 
where $a^L$ is the final output vector and $e$ is the expected result. 
 
Now we are going to measure the gradient of the cost function with respect to all the weights of the layers and the biases.
 
Let $p$ be one of the weights or biases in the layers. Then
 
&emsp; $\frac{\partial C}{\partial p} = \frac{\partial a^L}{\partial p}\cdot(v-e)$
 
For one training iteration (one input and expected output) we make repeated adjustments of the weights and biases (this is the Gradient Descent method):
 
&emsp; $p_{next} = p - r\frac{\partial C}{\partial p}$
 
where $r$ is the *learning rate*, and is usually a smallish number like 0.01.
 
Our next task is to come up with a formula for $\frac{\partial C}{\partial p}$.

## Back propagation
We are going to make use of some tricks with calculus to ensure that we reduce the amount of computation required to calculate partial derivatives.

We want to find partial derivatives of the cost function C relative to each of the weights in all the layers and all the biases so we know how to adjust the weights and biases to reduce the cost.

An important step is to define notation that works for the many different variables in the neural network. Using the layer number as a superscript on variables is very helpful.

Definitions:

1. $L$ is the number of layers in the network not including the input vector
1. $a^l$ is the activation vector (the values on the nodes of a layer in the network) for layer $l$. $a^{0}$ is the input vector and $a^{L}$ is the output vector.
2. $\sigma_l$ is the activation function for layer $l$, $\sigma_{0}$ is the identity function.
1. $b^l$ is the bias vector for layer $l$, $b^{0}$ is a zero vector.
1. $z^l$ is the intermediate vector (pre-activation) for layer $l$, $z^{0}$ is the input vector.
1. $z_{i}^l = \sum_{j} w_{ij}^la_{j}^{l-1} + b_{i}^l$
1. $a_{i}^l =  \sigma_l(z_{i}^l)$
1. $y$ is the vector of the expected values of the output ($a^{L}$ is the actual output vector)
1. Cost function $C = \frac{1}{2}\sum_i (a_{i}^{L} - y_{i})^{2}$. Note that the cost function can be arbitrary, we'll run with this one as an example.

For each weight $w_{ij}^l$ and bias $b_{i}^l$ we want to calculate the partial derivative of the cost function.

That is, we want to calculate $\frac{\partial C}{\partial w_{ij}^l}$ and $\frac{\partial C}{\partial b_{i}^l}$ for all $i, j, l$ 

Define vector $\delta^l$ with elements $\delta_{i}^l = \frac{\partial C}{\partial z_{i}^l}$. We are going to find this value very useful for simplifying computation.

Now

&emsp; $\frac{\partial C}{\partial w_{ij}^l} = \frac{\partial C}{\partial z_i^l} . \frac{\partial z_i^l}{\partial w_{ij}^l} = \delta_i^l \frac{\partial}{\partial w_{ij}^l}( \sum_k w_{ik}^l a_k^{l-1} + b_i^l ) = \delta_{i}^l . a_j^{l-1}$

Using the **point-distance-squared** cost function the last layer's $\delta$ value is:

&emsp; $\delta_{i}^{L} = a_{i}^{L} y_{i} . \sigma_{L}^{\prime}(z_{i}^{L})$

Using the **dot-product** cost function, $C = a^L . y$ :

&emsp; $\delta_{i}^{L} = y_{i} . \sigma_{L}^{\prime}(z_{i}^{L})$

Using the **cosine-similarity** cost function, $C = \frac{a^L . y}{|a||y|}$ :

&emsp; $\delta_i^L = \frac{y_i \sigma_L^{\prime}(z_i^L)}{|y|} (\frac{1}{|a|} - \frac{a^L . y}{{|a|}^3}) $

Let's now find an expression for $\delta_{i}^l$ in terms of the $\delta$ values in the $l+1$ layer:

&emsp; $\delta_{i}^l = \frac{\partial C}{\partial z_{i}^l} = \sum_{k} \frac{\partial C}{\partial z_{k}^{l+1}} . \frac{\partial z_{k}^{l+1}}{\partial z_{i}^l} = \sum_{k} \delta_{k}^{l+1} . \frac{\partial z_{k}^{l+1}}{\partial z_{i}^l}$

Why is a summation of partials happening here? The answer is that each $z_k^{l+1}$ is a function of all the members of $z^l$ so the [total derivative](https://en.wikipedia.org/wiki/Total_derivative) will be the sum of all the partial contributions.

Now 

&emsp; $z_{k}^{l+1} = \sum_{j} w_{kj}^{l+1}a_{j}^l + b_{k}^{l+1} = \sum_{j} w_{kj}^{l+1}\sigma_l(z_{j}^l) + b_{k}^{l+1}$

Therefore

&emsp; $\frac{\partial z_{k}^{l+1}}{\partial z_{i}^l} = w_{ki}^{l+1}\sigma_l^{\prime}(z_{i}^l)$

and thus

&emsp; $\delta_{i}^l = \sigma_l(z_{i}^l) \sum_{k} w_{ki}^{l+1}\delta_{k}^{l+1}$

In matrix notation:

&emsp; $\delta^l = (W^{l+1})^{T}\delta^{l+1} \circ \sigma_l^{\prime}(z^l)$ where $\circ$ is the Hadamard operator.

The formula for the cost function partial derivative with respect to a bias value is:

&emsp; $\frac{\partial C}{\partial b_{i}^l} = \delta_{i}^l$

With these formulae we now are in a position to calculate $z^l$, $a^l$ (and cache them for reuse) for all layers $l$ using a forward pass. We then calculate and cache
$\delta^{L}, \delta^{L-1}, .. , \delta^{1}$ in a _backwards_ pass (hence _back propagation_) through the layers. At that point we have all we need to calculate all partial derivatives of $C$ with respect to the weights and biases in all layers.

### Multi-classification (Softmax)
Especially when word embeddings are being computed it may be desirable to limit the output vector element values to the 0-1 range (for example probabilities). A function is applied to get the final neural network output and feeds the cost function. *Softmax* and *Standard Normalization* are examples of such a transformation.

Let's generalize this a bit. We are going to assume that the final activation function is the identity function and that we have a final transformation function $T$ that takes the output vector in the last layer and transforms it into a vector of the same size (for use in the cost function and as network output). An activation function is like this but only operates pointwise on the vector elements, without creating dependencies on other elements in the input vector that need to be considered in differentiation particularly.

We have this cost function for the system without this extra transformation:

&emsp; $C = f(x)$ where $x$ is the output vector (note that the expected vector for that output vector $y$ is a constant)

Define 

&emsp; $f^{\prime}(x) = \frac{\partial f}{\partial x}(x) = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..]^T$

Then define the cost taking into account the transformation as:

&emsp; $C_T = f(T(x))$

Define the Jacobian matrix (derivative) of T as

```math
\begin{aligned}
\nabla^{\mathrm T} = \begin{bmatrix}
    \dfrac{\partial T_1}{\partial x_1} & \cdots & \dfrac{\partial T_1}{\partial x_n}\\
    \vdots                             & \ddots & \vdots\\
    \dfrac{\partial T_m}{\partial x_1} & \cdots & \dfrac{\partial T_m}{\partial x_n}
\end{bmatrix}
\end{aligned}
```

Then, bearing in mind that $a^L = z^L$

&emsp; $\frac{\partial C_T}{\partial z^L_i} = \sum_j \frac{\partial T_i}{\partial x_i}(z^L)_j \frac{\partial f}{\partial x_j}(T(z^L))_j$

That is

&emsp; $\frac{\partial C_T}{\partial z^L} = \nabla^{\mathrm T}(z^L) f^{\prime}(T(z^L))$ 

In fact every layer honours a similar equation but the Jacobian matrix for an activation function only has entries on the diagonal so can be represented as a vector (and applied using the Hadamard element-wise product).

So our cost function changes, the neural network is changed and the only change to the $\delta$ values will be that $\delta_T^L = \frac{\partial C_T}{\partial z^L}$ from above.

The softmax function takes a vector and returns a vector:

&emsp; $S(x)_i = \frac{e^{x_i}}{\sum_k e^{x_k}}$

and its derivative $\nabla^{\mathrm S}$ is a square Jacobian matrix (unlike an activation function there are interdependencies to consider) with these entries:

&emsp; $\frac{\partial S_i}{\partial x_j} = S_i (kron_{ij} - S_j)$ where $kron$ is the Kronecker delta function: $kron_{ij} = 1\text{ if }i = j,\text{ otherwise }\ 0$

### Regularization
Using the above basic approach to minimize the cost function may end up with runaway weight sizes. To help with this scenario (in some circumstances) let's use a new cost function that helps minimize weights. 

The regularized cost function is 

&emsp; $C_R = C + R$

where $R$ is the *regularization function*. $R$ is a function of the weight values from all layers.

One example is the L2 regularization function:

&emsp; $R = \frac{\alpha}{2}\sum_{i,j,l} (w_{ij}^l)^2$ where $\alpha$ is the *regularization constant*

Now 

&emsp; $\frac{\partial C_R}{w_{ij}^l}  = \frac{\partial C}{\partial w_{ij}^l} + \frac{\partial R}{\partial w_{ij}^l} = \delta_i^l . a_j^{l-1} + \frac{\partial R}{\partial w_{ij}^l}$

When R is the L2 regularization function, 

&emsp; $\frac{\partial R}{\partial w_{ij}^l} = \alpha w_{ij}^l$

The gradient descent formula becomes:

&emsp; ${w_{ij}^l}^{\prime} = w_{ij}^l - r \frac{\partial C}{\partial w_{ij}^l} - \alpha r w_{ij}^l$
