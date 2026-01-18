# Mathematics Roadmap for Machine Learning & Data Science – TODO List

Track your progress from **Basic → Intermediate → Advanced**.  
Focus on **understanding + application** (not just memorization).  
Recommended: pair theory with Python/NumPy implementations, visualizations (3Blue1Brown, etc.), and small ML-relevant exercises.

## Basic Understanding (Entry-Level – Build Foundations)

### Linear Algebra Basics
- [ X ] Vectors: definition, addition, scalar multiplication, dot product
- [ X ] Matrices: definition, addition, scalar multiplication, matrix multiplication
- [ X ] Key matrix operations: transpose, identity matrix, inverse (when exists)
- [ X ] Vector norms (L1, L2/Euclidean)
- [ X ] Euclidean distance and Manhattan distance
- [ X ] Geometric intuition: vector as arrow/point, matrix as transformation

### Statistics Basics
- [ X ] Descriptive statistics: mean, median, mode
- [ X ] Variance and standard deviation
- [ X ] Covariance
- [ X ] Correlation (Pearson) and different methods to measure correlation

### Probability Basics
- [ X ] Basic probability rules (addition, multiplication)
- [ ] Joint, marginal, and conditional probability
- [ ] Independence of events
- [ X ] Bayes' theorem (understand intuition + formula)

### Calculus Basics
- [ X ] Limits and continuity (intuitive understanding)
- [ X ] Derivatives: definition, rules (power, product, quotient, chain)
- [ X ] Partial derivatives
- [ X ] Higher-order derivatives (second derivative → concavity)
- [ X ] Gradient (multivariable case)

## Intermediate Understanding (Practical Implementation – ML-Relevant)

### Linear Algebra Intermediate
- [ ] Eigenvalues and eigenvectors (intuition + computation)
- [ ] LU Decomposition (basics and purpose)
- [ ] Singular Value Decomposition (SVD) – very important!
- [ ] Applications: PCA intuition via SVD, low-rank approximation

### Probability & Statistics Intermediate
- [ ] Central Limit Theorem (why it matters in ML)
- [ ] Discrete probability distributions (Bernoulli, Binomial, Poisson)
- [ ] Continuous probability distributions (Uniform, Normal/Gaussian, Exponential)
- [ ] Hypothesis testing: null/alternative, p-value, significance level
- [ ] Confidence intervals (construction and interpretation)

### Calculus Intermediate
- [ ] Chain rule – deep understanding for backpropagation
- [ ] Partial derivatives in multivariable functions
- [ ] Gradient and its role in optimization

### Optimization
- [ ] Gradient Descent: intuition, math, learning rate
- [ ] Variations: Stochastic Gradient Descent (SGD), Mini-batch GD
- [ ] Momentum, learning rate schedules (basic awareness)

## Advanced Understanding (Research / Custom Algorithms – Deeper Insight)

### Vector Calculus
- [ ] Jacobian matrix (multivariable derivative)
- [ ] Hessian matrix (second-order derivatives, curvature)

### Probability Distributions & Statistics Advanced
- [ ] Sampling distributions
- [ ] Chi-Square distribution and tests
- [ ] t-Distribution and t-tests
- [ ] Parametric vs Non-Parametric tests
- [ ] Bias-Variance tradeoff (deep intuition)
- [ ] Bootstrap method (resampling)

### Geometry in High Dimensions
- [ ] Cosine similarity
- [ ] Jaccard similarity
- [ ] Orthogonality
- [ ] Projections (onto subspace/line)

### Regression Analysis & Estimation
- [ ] Maximum Likelihood Estimation (MLE) – core concept
- [ ] Mean Squared Error (derivation + why we minimize it)

---

## Quick Progress Milestones & Verification Checklist

- [ ] Can explain why matrix multiplication is central to neural networks / linear models
- [ ] Can derive the gradient for simple linear regression from scratch
- [ ] Understand how backpropagation uses the chain rule
- [ ] Can interpret eigenvalues/eigenvectors in context of PCA
- [ ] Know when to use cosine similarity vs Euclidean distance
- [ ] Can explain Bias-Variance tradeoff with a real example
- [ ] Comfortable reading math notation in ML papers (vectors, gradients, expectations)

## Recommended Learning Habits (Add Your Own Sub-tasks)

- [ ] Watch 3Blue1Brown Linear Algebra series (at least first 5–6 videos)
- [ ] Watch 3Blue1Brown Essence of Calculus series
- [ ] Implement vector/matrix operations from scratch in NumPy
- [ ] Code gradient descent for linear regression (no sklearn)
- [ ] Solve Bayes' theorem problems (spam filter example)
- [ ] Visualize distributions (normal, binomial) in Python
- [ ] Read one ML paper section and identify the math concepts used

Slow and steady wins — aim for deep intuition over rushing through topics.