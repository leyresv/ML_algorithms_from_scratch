# ML_algorithms_from_scratch

In this repo I implement from scratch different Machine Learning algorithms commonly used in Natural Language Processing tasks.

# Models

[Here](models) you can find the different model implementations.
## Logistic Regression

### Logistic Regression equation
---

$$ h(x_i, θ) = \frac{1}{1 + e^{-θ^Tx_i}}  $$

---

In Linear Regresssion, the output is the weighted sum of inputs.
*   Linear Regression Equation: $h(x) = θ^T x =\theta_0 x_0 + \theta_1 x_1 +
\theta_2 x_2 + ... \theta_m x_m$
*   Input featues vector for a single sequence: $x_i$
*   Weights vector: $θ$

In Logistic Regression, we pass the Linear Regression output through a sigmoid
function, that can map any real value between 0 and 1.
*   Sigmoid function: $σ(x) = \frac{1}{1 + e^{-x}} $
*   Logistic Regression Equation: $h(x_i) = σ(θ^T x_i)=$
    * $\geq 0.5   \text{    if     }   \theta^Tx_i > 0$
    * $< 0.5 \text{    if    }    \theta^Tx_i < 0$


```python
import numpy as np

def sigmoid(X, theta):
    """
    Sigmoid function for logistic regression

    :param X: input features. Numpy array of shape (m, n+1) (m=number of input sequences, n=number of features + 1 for bias)
    :param theta: weights vector. Numpy array of shape (n+1, 1)
    :return: logistic regression over X. Numpy array of shape (m, 1)
    """
    return 1.0 / (1.0 + np.exp(-np.dot(X, theta)))
```

### Cost function: log loss

Log loss for a single training example:
$$Loss = -1\cdot[{\color{DarkGreen} {y_i \cdot log(p_i)}} + {\color{Red}{(1-y_i)\cdot log(1-p_i)}}]$$
* Gold label: $y_i$
* Predicted probability for seq $x_i$ (between 0 and 1): $h(x_i, θ)$ => The logs
will be negative (we add a small value to avoid $log(0)$):
    * $log(0.0001)=-4$ 
    * $log(0.5)=-0.3$
    * $log(1)=0$


* First term: probability of 1: ${\color{DarkGreen} {p_i = h(x_i, \theta)}}$
* Second term: probability of 0: ${\color{Red} {1-p_i=1-h(x_i, \theta)}}$


 $y_i$|${\color{DarkGreen}{p_i}}$|$log({\color{DarkGreen}{p_i}})$|$y_i⋅log({\color{DarkGreen}{p_i}})$|${\color{Red} {1-p_i}}$|$log({\color{Red} {1-p_i}})$|$(1-y_i)⋅log({\color{Red} {1-p_i}})$|$y_i⋅log({\color{DarkGreen}{p_i}}) + (1-y_i)⋅log({\color{Red} {1-p_i}})$|loss
 :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
${\color{DarkGreen}1}$|${\color{DarkGreen}{0.9999}}$|$\sim0$|$\sim0$|$0.0001$|$-4$|$0$|$\sim0$|$\sim0$
${\color{DarkGreen}1}$|${\color{Red}{0.0001}}$|$-4$|$-4$|$0.9999$|$\sim0$|$0$|$-4$|$4$
${\color{Red}0}$|${\color{DarkGreen}{0.9999}}$|$\sim0$|$\sim0$|$0.0001$|$-4$|$-4$|$-4$|$4$
${\color{Red}0}$|${\color{Red}{0.0001}}$|$-4$|$0$|$0.9999$|$\sim0$|$\sim0$|$\sim0$|$\sim0$


The cost function used for logistic regression is the average of the log loss
across all training examples:

---
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[{\color{DarkGreen} {y_i \cdot
log(h(x_i, \theta))}} + {\color{Red} {(1-y_i)\cdot log(1-h(x_i, \theta))}}] $$

---

* Number of training examples: $m$

```python
def cost(Y_prob, Y):
    """
    Compute the log loss between predicted labels and gold labels

    :param Y_prob: predicted probabilities. Numpy array of shape (m, 1) (m= number of labels)
    :param Y: true labels. Numpy array of shape (m, 1)
    :return: cost value (float)
    """
    return float(-(np.dot(Y.T, np.log(Y_prob)) + np.dot((1-Y).T, np.log(1 - Y_prob)))/Y.shape[0])

```

### Weights update: Gradient descent

We update the weight $\theta_j$ by subtracting a fraction of the gradient
determined by $\alpha$: 
 $$\theta_j = \theta_j - \alpha \times \nabla_{\theta_j}J(\theta)$$
 $$\nabla_{\theta_j}J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(h_i-y_i)x_j$$

* $i$ is the index across all $m$ training examples.
* $j$ is the index of the weight $\theta_j$, so $x_j$ is the feature associated
with weight $\theta_j$

---
$$\mathbf{\theta} = \mathbf{\theta} - \frac{\alpha}{m} \times \left(
\mathbf{X}^T \cdot \left( \mathbf{Y_{prob}-Y} \right) \right)$$

---

* weights vector: $θ$, with dimensions $(n+1, 1)$
* lerning rate: $α$
* number of training examples: $m$
* training examples: $X$, with dimensions $(m, n+1)$
* training labels: $Y$, with dimensions $(m, 1)$
* predicted labels (probabilities): $Y_{prob}$

```python
def gradient_descent(X, Y, Y_prob, theta, alpha):
    """
    Update the weights vector

    :param X: training features. Numpy array of shape (m, n+1) (m=number of training sequences, n=number of features + 1 for bias)
    :param Y: training labels. Numpy array of shape (m, 1)
    :param Y_prob: predicted probabilities. Numpy array of shape (m, 1)
    :param theta: weights vector. Numpy array of shape (n+1, 1)
    :param alpha: learning rate. Float
    :return: updated weight vector
    """
    return theta - alpha/ X.shape[0] * np.dot(X.T, Y_prob-Y)
```

## Support Vector Machine

## Naïve Bayes

# Data

# Usage

# Visualization
