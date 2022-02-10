# JAGS Cheat Sheet

This cheat sheet covers all the models and their implementations in ``JAGS`` introduced in the course. 

Each section provides a prototypical implementation of each type of model. Note that the particular choice of probability model and priors used are only examples, and so will have to have to be adapted to suit the requirements of an applied problem. 

An example call of ``rjags`` for each model is also provided. Note that the choice of ``JAGS`` parameters (e.g. ``n.chains``, ``n.iter`` and ``thin``) used are only examples and will likely have to be changed to suit the demands of an applied problem.

Where applicable, the ``JAGS`` code for sampling from the predictive distribution has also been included. These parts may be removed if you don't want to perform prediction.

## Table of Contents
1. [Distributions](#distributions)
2. [Independent, Identically Distributed Data](#iid)
3. [Simple Linear Regression](#simplelinear)
4. [Multiple Linear Regression](#multilinear)
5. [Logistic Regression](#logistic)
6. [Binomial Logistic Regression](#binomial)
7. [Poisson Regression](#poisson)
8. [Hierarchical Regression](#hiearchichal)
   - [Varying Intercept no Group Predictors](#varying_intercept_no_group)
9.  [Mixture Model](#mixture)
10. [Autoregressive Model](#ar)
    - [$\text{AR}(1)$ Model](#ar1)
    - [$\text{AR}(2)$ Model](#ar2)
11. [Hidden Markov Model](#hmm)
12. [Gaussian Process](#gp)
13. [Gaussian Process Prediction](#gp_pred)

## Distributions: <a name="distributions"></a>

The following is a list of distributions available in base ``JAGS``. These distributions will be used either as a probability model for our observations, or as a prior for the parameters of our probability model.

1. [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution), $\text{Bernoulli}(p)$: ``dbern``. Discrete distribution, takes values $0$ or $1$.
2. [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), $\text{Beta}(\alpha,\beta)$: ``dbeta``. Univariate continuous distribution, takes values in the interval $[0,1]$.
3. [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution), $\text{Bin}(n,p)$: ``dbin``. Discrete distribution, takes values $0,\ldots,n$.
4. [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution), $\text{Categorical}(p)$: ``dcat``. Discrete distribution. Takes values in $\{1,\ldots,k\}$, where $k$ is the number of categories. So $p = (p_1,\ldots,p_k)$.
5. [Chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution), $\chi^2(k)$: ``dchisqr``. Univariate continuous distribution, takes values in $\mathbb{R}_{\geq 0} = [0, \infty)$.
6. [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution), $\text{Dirichlet}(\alpha_1,\ldots,\alpha_k)$: ``ddirch``. Multivariate continuous distribution, takes values in the standard $k-1$ simplex (the set of numbers $x = (x_1,\ldots,x_k) \in \mathbb{R}^k$ such that $\sum_{i=1}^k x_i = 1$ and each $x_i \geq 0$).
7. [Double Exponential distribution](https://en.wikipedia.org/wiki/Laplace_distribution), $\text{Laplace}(\mu, b)$: ``ddexp``. Univariate continuous distribution, takes values in $\mathbb{R}$.
8. [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution), $\text{Exp}(\lambda)$: ``dexp``. Univariate continuous distribution, takes values in $\mathbb{R}_{\geq 0}$.
9. [F distribution](https://en.wikipedia.org/wiki/F-distribution), $\text{F}(d_1,d_2)$: ``df``. Univariate continuous distribution, takes values in $\mathbb{R}_{\geq 0}$.
10. [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution), $\text{Gamma}(\alpha, \beta)$: ``dgamma``. Univariate continuous distribution, takes values in $\mathbb{R}_{\geq 0}$.
11. [Generalised Gamma distribution](https://en.wikipedia.org/wiki/Generalized_gamma_distribution), $\text{GeneralisedGamma}(r, \lambda, b)$: ``dgen.gamma``. Univariate continuous distribution, takes values in $\mathbb{R}_{\geq 0}$.
12. [Hypergeometric](https://en.wikipedia.org/wiki/Hypergeometric_distribution), $\text{HyperGeometric}(n_1,n_2,m,\psi)$: ``dhyper``. Discrete distribution, takes integer values $y$ in $\max(0, n_1 + n_2 − m) \leq y \leq \min(n_1,m).$
13. [Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution), $\text{Logistic}(\mu, s)$: ``dlogis``. Univariate continuous distribution, takes values in $\mathbb{R}$.
14. [Log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution), $\text{LogNormal}(\mu, \sigma^2)$: ``dlnorm``. Univariate continuous distribution, takes values in $\mathbb{R}$. In ``JAGS``, it is parameterised by the precision $\tau = 1/\sigma^{2}$ instead.
15. [Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution), $\text{Multinomial}(n,p_1,\ldots,p_k)$: ``dmulti``. "Multivariate" discrete distribution, takes values $y = (y_1,\ldots,y_k)$ satisfying $\sum_{i=1}^n y_i = n$ and each $y_i \in \{0,\ldots,n\}$.
16. [Multivariate Normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), $\text{MultivariateNormal}(\underline{\mu}, \Sigma)$: ``dmnorm`` and ``dmnorm.vcov``. Multivariate continuous distribution, takes values in $\mathbb{R}^d$, where $d$ is the dimension of mean vector $\underline{\mu}$. The ``JAGS`` distribution ``dmnorm.cov`` is parameterised by a covariance matrix $\Sigma$, whereas ``dmnorm`` is parameterised by a precision matrix $\Sigma^{-1}$.
17. [Multivarite Student-t distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution), $\text{Multivariate-t}(\underline{\mu}, \Sigma, \eta)$: ``dmt``. Multivariate continuous distribution, takes values in $\mathbb{R}^d$, where $d$ is the dimension of the location vector $\underline{\mu}$.
18. [Negative Binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution), $\text{NegBin}(p, r)$: ``dnegbin``. Discrete distribution, takes values in $\mathbb{N} = \{0,1,2,3,\ldots\}$.
19. [Noncentral Chi-squared distribution](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution), $\text{NonCentralChiSquare}(k, \delta)$: ``dnchisqr``. Univariate continuous distribution, takes values in $\mathbb{R}_{\geq 0}$.
20. [Noncentral Student-t](https://en.wikipedia.org/wiki/Noncentral_t-distribution), $\text{NonCentral-t}(\mu, \tau, d)$: ``dnt``. Univariate continuous distribution, takes values in $\mathbb{R}$.
21. [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution), $\mathcal{N}(\mu,\sigma^2)$: ``dnorm``. Univariate continuous distribution, takes values in $\mathbb{R}$. In ``JAGS``, it is parameterised by the precision $\tau = 1/\sigma^{2}$ instead.
22. [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution), $\text{Pareto}(\alpha,c)$: ``dpar``. Univariate continuous distribution, takes values in $[c, \infty)$.
23. [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution), $\text{Poisson}(\lambda)$: ``dpois``. Discrete distribution taking values in $\mathbb{N} = \{0,1,2,3,\ldots\}$.
24. Sampling with replacement: ``dsample``.
25. [Student-t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution), $\text{Student-t}(\mu,\tau,d)$: ``dt``. Univariate continuous distribution taking values in $\mathbb{R}$. Equivalent to non-central Student-t (typically the Student-t distribution is parameterised by just the degrees of freedom $d$, the introduction of location and precision parameters $\mu$ and $\tau$ turn it into a non-central distribution). A [heavier tailed](https://en.wikipedia.org/wiki/Heavy-tailed_distribution) Normal distribution.
26. [Uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution), $\text{Unif}(a,b)$: ``dunif``. Univariate continuous distribution, taking values in $[a,b]$.
27. [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution), $\text{Weibull}(v,\lambda)$: ``dweib``. Univariate continuous distribution, taking values in $\mathbb{R}_{\geq 0}$.
28. [Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution), $\text{Wishart}(R, d)$: ``dwish``. Multivariate continuous distribution, taking values in the space of [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix), [positive definite](https://en.wikipedia.org/wiki/Definite_matrix) matrices.

## Independent, Identically Distributed Data <a name="iid"></a>

#### Bayesian Model:

- Probability Model: $$Y_i | \alpha, \beta \sim \text{Beta}(\alpha, \beta).$$
- Prior for $\alpha$: $\alpha\sim\text{LogNormal}(\mu_a, \tau_a)$.
- Prior for $\beta$: $\beta \sim \text{LogNormal}(\mu_b, \tau_b)$.
-  Defining Prior Hyperparameters: $\mu_a = 1$, $\tau_a = 1$, $\mu_b = 2$ and $\tau_b = 3$.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is number of data):
            for (i in 1:N) {
                Y[i] ~ dbeta(alpha, beta)
            }

            # Priors for alpha and beta:
            alpha ~ dlnorm(mu_a, tau_a)
            beta ~ dlnorm(mu_b, tau_b)

            # Prediction:
            Y_pred ~ dbeta(alpha, beta)

            # Defining Prior Hyperparameters:
            mu_a <- 1
            tau_a <- 1
            mu_b <- 2
            tau_b <- 3
        }
        "
#### Running Code: 
In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of observations.
2. ``N``: The number of data points.

Example code:

    jags_data <- list(Y = Y, N = length(Y), L = length(x_pred))
    model <- jags.model(textConnection(model_string), data = jags_data, n.chains = 4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("beta_1", "beta_2", "tau", "Y_pred"), n.iter = 1000, thin = 5)


    
## Simple Linear Regression <a name="simplelinear"></a>

#### Bayesian Model:

- Probability Model: Each $Y_i| x_i, \beta_1, \beta_2, \tau \sim \mathcal{N}(\beta_1 + \beta_2x_i, \tau).$
- Prior for $\beta_1$: $\beta_1 \sim \mathcal{N}(\mu_1, \tau_1)$.
-  Prior for $\beta_2$: $\beta_2 \sim \mathcal{N}(\mu_2, \tau_2)$.
-  Prior for $\tau$: $\tau \sim \text{Gamma}(\alpha, \beta)$.
-  Specifying Prior Hyperparameters: $\mu_1 = 1$, $\tau_1 = 2$, $\mu_2 = -1$, $\tau_2 = 5$, $\alpha = 1$ and $\beta = 1$.


#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is number of data):
            for (i in 1:N) {
                mean[i] <- beta_1 + beta_2 * x[i]
                Y[i] ~ dnorm(mean[i], tau)
            }

            # Priors (M is the number predictors used):
            beta_1 ~ dnorm(mu_1, tau_1)
            beta_2 ~ dnorm(mu_2, tau_2)
            tau ~ dgamma(alpha, beta)

            # Prediction (L is the number of predictions to make):
            for (k in 1:L) {
                mean_pred[k] <- beta_1 + beta_2 * x_pred[k] 
                Y_pred[k] ~ dnorm(mean_pred[k], tau)
            }

            # Specifying Prior Hyperparameters:
            mu_1 <- 1
            tau_1 <- 2
            mu_2 <- -1
            tau_2 <- 5
            alpha <- 1
            beta <- 1
        }
        "

#### Running Code: 
In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses.
2. ``x``: the length ``N`` vector of the values of the predictor.
3. ``x_pred``: the length ``L`` vector of new the values to predict with.
4. ``N``: The number of data points.
5. ``L``: The number of predictions we want to make. The length of ``x_pred``.

Example code:

    jags_data <- list(Y = Y, x = x, x_pred = x_pred, N = length(Y), L = length(x_pred))
    model <- jags.model(textConnection(model_string), data = jags_data, n.chains = 4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("beta_1", "beta_2", "tau", "Y_pred"), n.iter = 1000, thin = 5)

## Standard Multilinear Regression <a name="multilinear"></a>

#### Bayesian Model:

In the following, each $x_i = (x_{i1}, x_{i2},\ldots,x_{iM})^\top$ is the (column) vector of predictors and $\beta = (\beta_1,\ldots,\beta_M)^\top$ is the (column) vector of coefficients. If we are including an intercept term, $x_{i1} = 1$ and $\beta_1$ would be the intercept.

- Probability Model: $$Y_i| x_i,  \beta, \tau \sim \mathcal{N}(x_i^\top \beta, \tau).$$
- Prior for $\beta = (\beta_1,\ldots,\beta_M)^\top$: Each $\beta_j \sim \mathcal{N}(\mu_0, \tau_0)$.
-  Prior for $\tau$: $\tau \sim \text{Gamma}(\alpha, \beta)$.
-  Specifying Prior Hyperparameters: $\mu_0 = 0$, $\tau_0 = 10^{-2}$, $\alpha = 0.1$ and $\beta = 0.1$.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is number of data):
            for (i in 1:N) {
                mean[i] <- inprod(X[i, ], beta) # X is design matrix
                Y[i] ~ dnorm(mean[i], tau)
            }

            # Priors (M is the number predictors used):
            for (j in 1:M) {
                beta[j] ~ dnorm(mu_0, tau_0)
            }
            tau ~ dgamma(alpha, beta)

            # Prediction (L is the number of predictions to make):
            for (k in 1:L) {
                mean_pred[k] <- inprod(X_pred[k, ], beta)
                Y_pred[k] ~ dnorm(mean_pred[k], tau)
            }

            # Specifying Prior Hyperparameters:
            mu_0 <- 0
            tau_0 <- 1E-2
            alpha <- 0.1
            beta <- 0.1
        }
        "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses. 
2. ``X``: the design matrix of predictors corresponding to the responses ``Y``. The $i$th row of ``X`` are the predictor values for the $i$th response ``Y[i]``. A matrix with ``N`` rows and ``M`` columns. In order to use an intercept term in our regression model, the first column of the design matrix ``X[,1]`` must be a column of ones.
3. ``X_pred``: matrix of new predictor values. A matrix with ``N`` rows and ``L`` columns.
4. ``N``: The number of data points (the number of rows of your data frame).
5. ``M``: The number of predictors used $x_i = (x_{i1}, \ldots, x_{iM})^\top$. The number of columns of ``X`` and ``X_pred``.
6. ``L``: The number of predictions we want to make. The number of rows of ``X_pred``

Example code:

    jags_data <- list(Y = Y, X = X, X_pred = X_pred,
                       N = length(Y), M = ncol(X), L = nrow(X_pred))
    model <- jags.model(textConnection(model_string),
                        data=jags_data, n.chains=4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("beta", 
                            "tau", "Y_pred"), n.iter = 1000, thin = 5)

## Logistic Regression <a name="logistic"></a>

#### Bayesian Model:

In the following, each $x_i = (x_{i1}, x_{i2},\ldots,x_{iM})^\top$ is the (column) vector of predictors and $\beta = (\beta_1,\ldots,\beta_M)^\top$ is the (column) vector of coefficients. If we are including an intercept term, $x_{i1} = 1$ and $\beta_1$ would be the intercept.

- Probability Model: $$Y_i | x_i, \beta \sim \text{Bernoulli}\left(\text{Logistic}(x_i^\top\beta)\right).$$
- Prior for $\beta = (\beta_1,\ldots,\beta_M)^\top$: Each $\beta_j \sim \mathcal{N}(\mu_0, \tau_0)$.
- Defining Prior Hyperparameters: $\mu_0 = 1$, $\tau_0 = 2$.
  
Here,

$$ \text{Logistic}(x) = \frac{1}{1 + e^{-x}}$$

is the [*logistic function*](https://en.wikipedia.org/wiki/Logistic_function). In ``JAGS`` this is the function ``ilogit``.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is number of data):
            for (i in 1:N) {
                linear[i] <- inprod(X[i, ], beta)
                prob[i] <- ilogit(linear[i])
                Y[i] ~ dbern(prob[i])
            }

            # Priors (M is the number predictors used):
            for (j in 1:M) {
                beta[j] ~ dnorm(mu_0, tau_0)
            }

            # Prediction (L is the number of predictions to make):
            for (k in 1:L) {
                linear_pred[k] <- inprod(X_pred[k, ], beta)
                prob_pred[k] <- ilogit(linear_pred[k])
                Y_pred[k] ~ dbern(prod_pred[k])
            }

            # Defining Prior Hyperparameters:
            mu_0 <- 1
            tau_0 <- 2
        }
        "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses. 
2. ``X``: the design matrix of predictors corresponding to the responses ``Y``. The $i$th row of ``X`` are the predictor values for the $i$th response ``Y[i]``. A matrix with ``N`` rows and ``M`` columns. In order to use an intercept term in our regression model, the first column of the design matrix ``X[,1]`` must be a column of ones.
3. ``X_pred``: matrix of new predictor values. A matrix with ``N`` rows and ``L`` columns.
4. ``N``: The number of data points (the number of rows of your data frame).
5. ``M``: The number of predictors used $x_i = (x_{i1}, \ldots, x_{iM})^\top$. The number of columns of ``X`` and ``X_pred``.
6. ``L``: The number of predictions we want to make. The number of rows of ``X_pred``

Example code:

    jags_data <- list(Y = Y, X = X, X_pred = X_pred,
                       N = length(Y), M = ncol(X), L = nrow(X_pred))
    model <- jags.model(textConnection(model_string),
                        data=jags_data, n.chains=4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("beta", 
                            "Y_pred"), n.iter = 1000, thin = 5)

## Poisson Regression <a name="poisson"></a>

#### Bayesian Model:

In the following, each $x_i = (x_{i1}, x_{i2},\ldots,x_{iM})^\top$ is the (column) vector of predictors and $\beta = (\beta_1,\ldots,\beta_M)^\top$ is the (column) vector of coefficients. If we are including an intercept term, $x_{i1} = 1$ and $\beta_1$ would be the intercept.

- Probability Model: $$Y_i | x_i, \beta \sim \text{Poisson}\left(\exp(x_i^\top\beta)\right).$$
- Prior for $\beta = (\beta_1,\ldots,\beta_M)^\top$: Each $\beta_j \sim \mathcal{N}(\mu_0, \tau_0)$.
-  Defining Prior Hyperparameters: $\mu_0 = 1$, $\tau_0 = 2$.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is number of data):
            for (i in 1:N) {
                linear[i] <- inprod(X[i, ], beta)
                rate[i] <- exp(linear[i])
                Y[i] ~ dpoisson(rate[i])
            }

            # Priors (M is the number predictors used):
            for (j in 1:M) {
                beta[j] ~ dnorm(mu_0, tau_0)
            }

            # Prediction (L is the number of predictions to make):
            for (k in 1:L) {
                linear_pred[k] <- inprod(X_pred[k, ], beta)
                rate_pred[k] <- exp(linear_pred[k])
                Y_pred[k] ~ dpoisson(rate_pred[k])
            }

            # Defining Prior Hyperparameters:
            mu_0 <- 1
            tau_0 <- 2
        }
        "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses. 
2. ``X``: the design matrix of predictors corresponding to the responses ``Y``. The $i$th row of ``X`` are the predictor values for the $i$th response ``Y[i]``. A matrix with ``N`` rows and ``M`` columns. In order to use an intercept term in our regression model, the first column of the design matrix ``X[,1]`` must be a column of ones.
3. ``X_pred``: matrix of new predictor values. A matrix with ``N`` rows and ``L`` columns.
4. ``N``: The number of data points (the number of rows of your data frame).
5. ``M``: The number of predictors used $x_i = (x_{i1}, \ldots, x_{iM})^\top$. The number of columns of ``X`` and ``X_pred``.
6. ``L``: The number of predictions we want to make. The number of rows of ``X_pred``.

Example code:

    jags_data <- list(Y = Y, X = X, X_pred = X_pred, N = length(Y), M = ncol(X), L = nrow(X_pred))
    model <- jags.model(textConnection(model_string), data = jags_data, n.chains=4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("beta", "Y_pred"), n.iter = 1000, thin = 5)

## Hiearchichal Regression <a name="hierarchical"></a>

###  Varying Intercept with No Group Level Predictors <a name="varying_intercept_no_group"></a>

#### Bayesian Model:

In the following, each $x_i = (x_{i1}, x_{i2},\ldots,x_{iM})^\top$ is the (column) vector of predictors and $\beta = (\beta_1,\ldots,\beta_M)^\top$ is the (column) vector of coefficients. The intercept term is provided by $c_j$ for $j =1,\ldots, J$ implying there are $J$ total groups.

- Individual Probability Model: $$Y_i | x_i, c_j, \beta, \tau \sim \mathcal{N}(c_j + x_i^\top \beta, \tau ).$$
- Group Probability Model: $$ c_j | \mu_c, \tau_c \sim \mathcal{N}(\mu_c, \tau_c).$$
- Prior for $\beta$: Each $\beta_k \sim \mathcal{N}(0, 10^{-2})$.
- Prior for $\tau$: $\tau \sim \text{Gamma}(10^{-3}, 10^{-3}) .$
- Prior for $\mu_c$: $ \mu_c \sim \mathcal{N}(0, 10^{-2}). $
- Prior for $\tau_c$: $ \tau_c \sim \text{Gamma}(10^{-3}, 10^{-3}). $

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Individual Probability Model:
            for (i in 1:N) {
                unit_mean <- inprod(X[i, ], beta)
                group_mean <- c[group_number[i]] 
                Y[i] ~ dnorm(unit_mean + group_mean, tau)
            }
            # Group Probability Model:
            for (j in 1:J) {
                c[j] ~ dnorm(mu_c, tau_c)
            }

            # Priors:
            for (k in 1:M) {
                beta[k] ~ dnorm(0, 1E-2)
            }
            tau ~ dgamma(1E-3, 1E-3)

            mu_c ~ dnorm(0, 1E-2)
            tau_c ~ dgamma(1E-3, 1E-3)
        }

    "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses. 
2. ``X``: the design matrix of individual level predictors corresponding to the responses ``Y``. The $i$th row of ``X`` are the predictor values for the $i$th response ``Y[i]``. A matrix with ``N`` rows and ``M`` columns.
3. ``group_number``: A length ``N`` vector with $i\text{th}$ entry corresponding to the group of number of reponse $i$.
4. ``N``: The number of data points (the number of rows of your data frame).
5. ``J``: The number of groups.


Example code:

    jags_data <- list(Y = Y, X = X, X_pred = X_pred, N = length(Y), M = ncol(X), L = nrow(X_pred))
    model <- jags.model(textConnection(model_string), data = jags_data, n.chains=4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("beta", "Y_pred"), n.iter = 1000, thin = 5)

## Mixture Model <a name="mixture"></a>

### Normal Mixture Model

#### Bayesian Model:

In the following, $p = (p_1,\ldots, p_J)$ is the vector of group probabilities: each $p_j$ is the probability of belonging to group $j$ and satisfies $\sum_{j=1}^J p_j = 1$. $J$ is the total number of groups and there are $J$ different means and precisions: $\mu_1, \ldots, \mu_J$ and $\tau_1,\ldots,\tau_J$. Recall that a categorical variable takes values in $\{1,\ldots, J\}$ and so $\mu_{c_i}$ and $\tau_{c_i}$ makes sense.

- Probability Model: $$ Y_i | c_i \sim \mathcal{N}(\mu_{c_i}, \tau_{c_i}), $$ $$ c_i|p \sim \text{Categorical}(p). $$
- Prior for $p$: $p \sim \text{Dirichlet}(d_1, \ldots, d_J)$.
- Prior for $\mu_1,\ldots, \mu_J$: Each $\mu_j \sim \mathcal{N}(0, 10^{-6})$.
- Prior for $\tau_1,\ldots,\tau_J$: Each $\tau_j \sim \text{Gamma}(10^{-2}, 10^{-2})$.
- Specification of Prior Hyperparameters: Each $d_j = 1$.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model:
            for (i in 1:N) {
                Y[i] ~ dnorm(mu[c[i]], tau[c[i]])
                c[i] ~ dcat(p_vec)
            }
            # Priors:
            for (j in 1:J) {
                d[j] <- 1 # constructing d vector
                mu_unsorted[j] ~ dnorm(0, 1E-6)
                tau[j] ~ dgamma(1E-2, 1E-2)
            }
            p_vec ~ ddirich(d) # dirichlet prior
            mu <- sort(mu_unsorted) # tries to stop label switching
        }
    "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of observations. 
2. ``N``: The number of data points (the number of rows of your data frame).
3. ``J``: The number of groups.

Example code:

    jags_data <- list(Y = Y, N = length(Y), J = 3)
    model <- jags.model(textConnection(model_string), data = jags_data, n.chains=4)
    update(model, n.iter = 10000)
    samples <- coda.samples(model = model, variable.names = c("mu", "p_vec", "tau"), n.iter = 10000, thin = 10)


## Autoregressive Model <a name="ar"></a>

Autoregressive models are suitable for time series data.

### $\text{AR}(1)$ model <a name="ar1"></a>

#### Bayesian Model:

- Probability Model: $$Y_t | y_{t-1}, \mu, 
  \alpha, \tau \sim \mathcal{N}(\mu + \alpha(y_{t-1}-\mu), \tau).$$
- Prior for $\mu$: $\mu \sim \mathcal{N}(0, 10^{-4})$.
- Prior for $\alpha$: $\alpha \sim \mathcal{N}(0, 10^{-4})$.
- Prior for $\tau:$ $\tau\sim \text{Gamma}(10^{-3},10^{-3})$.

If we know a-priori (beforehand) that our time series is *stationary*, then we would specify a prior for $\alpha$ that ensures that $-1\leq \alpha \leq 1$. For example, $\alpha\sim \text{Unif}(-1,1)$, using the ``JAGS`` code  ``alpha ~ dunif(-1,1)`` or a transformed Beta distribution:

        t_alpha ~ dbeta(a, b)
        alpha <- 2 * t_alpha - 1
        a <- 1
        b <- 1 # these hyperparameters can be changed

The uniform prior may be preferred if we lack any more prior information about $\alpha$.

#### ``JAGS`` Code:

THe following ``JAGS`` code defines the prior hyperparameters implicitly:

    model_string <- "
        model{
            # Probability Model (N is number of data):
            for (t in 2:N) {
                Y[t] ~ dnorm(mu + alpha * (Y[t-1] - mu), tau)
            }

            # Priors:
            mu ~ dnorm(0, 1E-4)
            alpha ~ dnorm(0, 1E-4)
            tau ~ dgamma(0.001, 0.001)

            # Prediction (L is the number of predictions to make into the future):
            Y_pred[1] ~ dnorm(mu + alpha * (Y[N] - mu), tau)
            for (t in 2:L)) {
                Y_pred[t] ~ dnorm(mu + alpha * (Y_pred[t-1] - mu), tau)
            }
        }
        "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses. 
2. ``N``: The number of data points (the number of rows of your data frame).
1. ``L``: The number of predictions we want to make. The number of rows of ``X_pred``.

Example code:

    jags_data <- list(Y = Y, N = length(Y), L = 5)
    model <- jags.model(textConnection(model_string), data = jags_data, n.chains=4)
    update(model, n.iter = 4000)
    samples <- coda.samples(model = model, variable.names = c("mu", "alpha", "tau", "Y_pred"), n.iter = 4000, thin = 3)

### $\text{AR}(2)$ model <a name="ar2"></a>

(Under Construction.)

### $\text{AR}(2)$ model with linear trend <a name="arn_linear"></a>

(Under Construction.)

## Hidden Markov Model <a name="hmm"></a>

(Under Construction)

## Gaussian Process <a name="gp"></a>

We only use ``JAGS`` to perform Bayesian inference on the mean parameters (if given), kernel parameters (typically only the lengthscale $\ell > 0$ and amplitude $\sigma^2 > 0$) and error variance $\delta > 0$. If our parameters must be positive, we must use a prior which takes values in $(0,\infty)$. This can be achieved in two ways in ``JAGS``:
1. Placing a prior directly on a parameter which takes values in $(0,\infty)$. For example, the following ``JAGS`` code specifies the lengthscale $\ell$ as a [*truncated Normal distribution*](https://en.wikipedia.org/wiki/Truncated_normal_distribution) (restricted to be greater $0$):

        lengthscale ~ dnorm(10, 1 / 4)  T(0,)
2.  Placing a prior on a parameter which takes values in $(-\infty,\infty)$ on the transformed parameter and performing a transformation that ensures positivity. For example, the following ``JAGS`` code, we specify the lengthscale $\ell$ as the square of a Student-t variable:
   
        t_lengthscale ~ dt(0, 1)
        lengthscale <- t_lengthscale ** 2

 We perform prediction outside of ``JAGS`` (refer to [Gaussian Process Prediction](#gp_pred) for how to perform prediction).

### Zero-Mean Gaussian Process

#### Bayesian Model:

The following assumes a squared-exponential kernel with a lengthscale parameter $\ell > 0$ and amplitude parameter $\sigma^2 > 0$. The ${\bf X} = (x_1, \ldots, x_n)^\top$ is the vector of all predictor values.

- $(Y_1, \ldots, Y_N) | {\bf X}, \sigma^2, \ell, \delta\sim \text{MultivariateNormal}({\bf 0}, \Sigma)$, where $$\Sigma = \begin{pmatrix} k(x_1, x_1) + \delta & k(x_1, x_2) & \ldots & k(x_1, x_N) \\ k(x_2, x_1) & k(x_2, x_2)+ \delta & \ldots & k(x_2, x_N) \\ \vdots  & \vdots & \ddots & \vdots \\ k(x_N, x_1) & k(x_N, x_2) & \ldots & k(x_N, x_N)+ \delta \\\end{pmatrix} $$
  and $k(x_i, x_j) = \sigma^2 \exp\left(-\frac{|x_i - x_j|^2}{2\ell}\right)$ is the squared-exponential kernel.
- Prior for $\sigma^2$: $\sigma^2 \sim \text{Gamma}(\alpha, \beta) $.
- Prior for $\ell$:  $\ell\sim \text{LogNormal}(\mu_0, \tau_0)$.
- Prior for $\delta$: $\delta\sim\text{Exp}(\lambda)$.
- Specification of Prior Hyperparameters: $\alpha = 1$, $\beta = 1$, $\mu_0 = 1$, $\tau_0 = 1/10$, $\lambda = 4$.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is the number of data):
            Y ~ dmvnorm.vcov(mean_vec, cov_mat)

            for (i in 1:N) {
                mean_vec[i] <- 0 # Zero-Mean
                # diagonal terms:
                cov_mat[i, i] <- amplitude + delta
                # off-diagonal terms:
                for (j in (i+1):N){
                    cov_mat[i, j] <- amplitude * exp(-0.5 * (X[i] - X[j]) ** 2 / lengthscale)
                    cov_mat[j, i] <- cov_mat[i, j]
                }
            }

            # Priors:
            amplitude ~ dgamma(alpha, beta)
            length_scale ~ dlnorm(mu_0, tau_0)
            delta ~ dexp(lambda)

            # Defining Prior Hyperparameters:
            alpha <- 1
            beta <- 1
            mu_0 <- 1
            tau_0 <- 1/10
            lambda <- 4
        }
        "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses.
2. ``X``: the length ``N`` vector of the values of the predictor.
3. ``N``: The number of data points.

Example code:

    jags_data <- list(Y = Y, X = X, N = length(Y))
    model <- jags.model(textConnection(model_string), data=jags_data, n.chains=4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("amplitude", "lengthscale", "delta"), n.iter = 1000, thin = 5)

### Linear Mean

The following assumes model assumes the mean of the Gaussian process is linear:

$$ \mu(x) = a + bx, $$

with unknown coefficients to be estimated $a$ and $b$. It is also common to assume the mean is just a constant:

$$ \mu(x) = a. $$

The following ``JAGS`` code can be easily adapted to posit this instead.

#### Bayesian Model:

The following assumes a squared-exponential kernel with a lengthscale parameter $\ell > 0$ and amplitude parameter $\sigma^2 > 0$.

- $(Y_1, \ldots, Y_N) | {\bf X}, \sigma^2, \ell, \delta\sim \text{MultivariateNormal}(\mu, \Sigma)$, where $$ \mu = \left(a + bx_1, a + bx_2, \ldots, a+bx_N \right)^\top $$ and
  $$\Sigma = \begin{pmatrix} k(x_1, x_1) + \delta^2 & k(x_1, x_2) & \ldots & k(x_1, x_N) \\ k(x_2, x_1) & k(x_2, x_2)+ \delta^2 & \ldots & k(x_2, x_N) \\ \vdots  & \vdots & \ddots & \vdots \\ k(x_N, x_1) & k(x_N, x_2) & \ldots & k(x_N, x_N)+ \delta^2 \\\end{pmatrix} $$
  and $k(x_i, x_j) = \sigma^2 \exp\left(-\frac{|x_i - x_j|^2}{2\ell}\right)$ is the squared-exponential kernel.
- Prior for $a$ and $b$: $a\sim \mathcal{N}(0, 1/10)$ and $b\sim \mathcal{N}(0, 1/10)$.
- Prior for $\sigma^2$: $\sigma^2 \sim \text{Gamma}(\alpha, \beta) $.
- Prior for $\ell$:  $\ell\sim \text{LogNormal}(\mu_0, \tau_0)$.
- Prior for $\delta$: $\delta\sim\text{Exp}(\lambda)$.

#### ``JAGS`` Code:

    model_string <- "
        model{
            # Probability Model (N is the number of data):
            Y ~ dmvnorm.vcov(mean_vec, cov_mat)

            for (i in 1:N) {
                mean_vec[i] <- a + b * X[i] # Zero-Mean
                # diagonal terms:
                cov_mat[i, i] <- amplitude + delta
                # off-diagonal terms:
                for (j in (i+1):N){
                    cov_mat[i, j] <- amplitude * exp(-0.5 * (X[i] - X[j]) ** 2 / lengthscale)
                    cov_mat[j, i] <- cov_mat[i, j]
                }
            }

            # Priors:
            a ~ dnorm(0, 1/10)
            b ~ dnorm(0, 1/10)
            amplitude ~ dgamma(alpha, beta)
            length_scale ~ dlnorm(mu_0, tau_0)
            delta ~ dexp(lambda)

            # Defining Prior Hyperparameters:
            alpha <- 1
            beta <- 1
            mu_0 <- 1
            tau_0 <- 1/10
            lambda <- 4
        }
        "

#### Running Code:

In order to run the ``JAGS`` program with data, you need to provide ``rjags`` an ``R`` list with the following attribute names:

1. ``Y``: the length ``N`` vector of responses.
2. ``X``: the length ``N`` vector of the values of the predictor.
3. ``N``: The number of data points.

Example code:

    jags_data <- list(Y = Y, X = X, N = length(Y))
    model <- jags.model(textConnection(model_string), data=jags_data, n.chains=4)
    update(model, n.iter = 1000)
    samples <- coda.samples(model = model, variable.names = c("a", "b", "amplitude", "lengthscale", "delta"), n.iter = 1000, thin = 5)

## Gaussian Process Prediction <a name="gp_pred"></a>

### Gaussian Process Class:

To perform prediction, use the following Gaussian Process class (i.e. copy and paste into ``R``):

    # install.packages(c("MASS, "expm"))
    library(MASS) # this is used to compute multivariate normal samples
    library(expm) # this is used for matrix square root

    setClass("GaussianProcess", 
        slots = c(
        mean = "function", 
        kernel = "function",
        delta = "numeric"
        ), 
        prototype = list(
        delta = 0
        )
    )

    setGeneric("condition", function(object, ...) standardGeneric("condition"))
    setGeneric("sample", function(object, n_samples, ...) standardGeneric("sample"))

    setMethod("condition", "GaussianProcess", function(object, x_pred, x_cond, y_cond, ...) {
        prior_mean <- object@mean(x_pred, ...)
        prior_covariance_matrix <- object@kernel(x_pred, x_pred, ...) 

        cond_mean <- object@mean(x_cond, ...)
        cond_covariance_matrix <- object@kernel(x_cond, x_cond, ...) + diag(length(x_cond)) * object@delta ** 2

        cross_covariance_matrix <- object@kernel(x_pred, x_cond, ...)
        inverse_prior_covariance_matrix <- solve(cond_covariance_matrix)

        posterior_mean <- prior_mean + cross_covariance_matrix %*% inverse_prior_covariance_matrix %*% matrix(y_cond - cond_mean, ncol = 1)
        posterior_covariance_matrix <- prior_covariance_matrix -
                                    cross_covariance_matrix %*% inverse_prior_covariance_matrix %*% t(cross_covariance_matrix) +
                                    diag(object@delta ** 2, nrow = length(x_pred))

        return(list("mean_vector" = c(posterior_mean), "covariance_matrix" = posterior_covariance_matrix, "variance" = diag(posterior_covariance_matrix)))
    }
    )

    setMethod("sample", "GaussianProcess", function(object, n_samples, mean_vector, covariance_matrix) {
        n <- length(mean_vector) # length of normal vector
        normal_sample <- rnorm(n * n_samples, mean = 0, sd = 1)
        Z_sample <- matrix(normal_sample, nrow = n)

        matrix_sqrt <- Re(sqrtm(covariance_matrix)) # perform square root
        mvn_sample <- matrix_sqrt %*% Z_sample + mean_vector

        return(t(mvn_sample))
    }
    )

### Mean Functions and Covariance Functions:

Your mean function and covariance function must match the one used in your ``JAGS`` code. For example, in the Marathon example, the mean function used was:

    marathon_mean_func <- function(x, a, b, ...) {
        output <- a + b * (x - 1940)
        return(output)
    }

That is, a linear mean function with shifted to the right by $1940$. This was used because predictor values were years (and so were centered around $1940$) and so helped standardise the data for MCMC mixing.

If you need to define your own mean function or covariance function, remember to include ``...`` as a final argument (this is to allow for optional arguments).

#### Zero-Mean function:

    zero_mean_func <- function(x, ...) {
        return(numeric(length(x)))
    }

#### Constant Mean function:

    constant_mean_func <- function(x, a, ...) {
        return(numeric(length(x)) + a)
    }

#### Linear Mean function:

    linear_mean_func <- function(x, a, b, ...) {
        output <- a + b * x
        return(output)
    }

#### Squared Exponential Kernel:

    sqexp_kernel <- function(x, y, amplitude, lengthscale) {
        output <- amplitude * exp(- (x - y) ** 2 / (2 * lengthscale ** 2))
        return(output)
    }

    squared_exponential <- function(x, y, amplitude, lengthscale, ...) {
        covariance_matrix <- outer(x, y, sqexp_kernel, amplitude, lengthscale)
        return(covariance_matrix)
    }

### Visualising and Predicting with Gaussian Processes:

In order to perform prediction, we need to instantiate a Gaussian process object. In order to do this, we must provide a choice of mean function, a choice of kernel and a value of the error variance parameter $\delta$ (default value is $\delta = 0$).  As an example, lets pick the ``zero_mean_func`` and the ``squared_exponential`` kernel and the value $0$ for the ``delta`` parameter. This is achieved with the following code:

    gp = new("GaussianProcess", mean = zero_mean_func, kernel = squared_exponential, delta = 0)

In order to perform prediction we need to provide a Gaussian process object ``gp`` a vector of $x$ values to predict using and the vectors $x$ and $y$ we are conditioning upon:

    x_pred <- seq(0, 5, length.out = 400)
    x_cond <- c(1.2, 2.3, 3.4)
    y_cond <-  c(0.6, -2, 1.6)

    output <- condition(gp, x_pred, x_cond, y_cond, amplitude = 1,lengthscale = 0.5)
    posterior_mean_vector <- output$mean_vector
    posterior_covariance_matrix <- output$covariance_matrix
    posterior_variance <- output$variance # vector of variances at each x_pred location

---
**NOTE**

Use the posterior means of your ``JAGS`` obtained samples as the values for ``delta``, ``amplitude``, ``lengthscale`` and further mean parameters (if applicable) when using the ``condition`` function. These should all inputted with the equals sign (e.g. ``amplitude = 2.3``). You can obtain the posterior means using ``summary(samples)$statistics[ ,1]`` or simply compute the means of each of the columns of samples using ``colMeans(as.matrix(samples))``.

---

There are multiple ways to visualise a Gaussian process. The way expected in this course is to plot the posterior mean, and the $95\%$ prediction interval. This is achieved by first plotting the posterior mean vector and then plotting the posterior mean vector plus and minus $1.96$ times the posterior standard deviation vector. The ``ggplot`` code to do this is as follows (for the previous example):

    library(ggplot2)

    posterior_sd <- sqrt(posterior_variance)

    gp_df <- data.frame(x = x_pred, y = posterior_mean_vector,
                        y_max = posterior_mean_vector + 1.96 * posterior_sd,
                        y_min = posterior_mean_vector - 1.96 * posterior_sd)

    point_df <- data.frame(x = x_cond, y = y_cond)

    gp_posterior_plot <- ggplot(gp_df) +
                    geom_line(aes(x = x, y = y), colour = "red") +
                    geom_ribbon(aes(x = x, ymin = y_min, ymax = y_max), alpha = 0.1,
                                fill = "red", color = "red", linetype = "dotted") +
                    geom_point(data = point_df, aes(x = x, y = y))
    gp_posterior_plot

This generates the following figure:


![Gaussian Process Posterior](https://github.com/MatthewAlexanderFisher/MAS8405-Bayesian-Data-Analysis/blob/main/Seminar%202/figures/posterior_gp.png?raw=true)


Alternatively, you can visualise samples from the predictive distribution. In order to do this, we can sample from the posterior predictive as follows:

    N_samples <- 200
    posterior_samples <- sample(gp, N_samples, posterior_mean_vector, posterior_covariance_matrix)

The output ``posterior_samples`` will be a large matrix with each row corresponding to a single sample of the posterior Gaussian process. To plot these samples, we can use ``ggplot`` as follows:

    # install.packages("reshape2")
    library(reshape2)

    gp_samples_df <- melt(data.frame(x_pred, t(posterior_samples)), id = "x_pred")
    point_df <- data.frame(x = x_cond, y = y_cond)

    gp_sample_plot <- ggplot(gp_samples_df) +
                    geom_line(aes(x = x_pred, y = value, group = variable), alpha = 0.1, colour = "green") +
                    geom_point(data = point_df, aes(x = x, y = y)) + xlab("x") + ylab("y")
    gp_sample_plot

This generates the following figure:

![Gaussian Process Posterior](https://github.com/MatthewAlexanderFisher/MAS8405-Bayesian-Data-Analysis/blob/main/Seminar%202/figures/posterior_gp_samples.png?raw=true)