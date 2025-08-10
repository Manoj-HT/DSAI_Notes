# Machine learning - 1
------------------------------------------

# C1: BASICS OF MACHINE LEARNING & DATA PREPROCESSING

## What is Machine Learning?

- **Machine Learning** : Machine Learning (ML) isz a field of AI where computers learn patterns from data and make predictions/decisions without being explicitly programmed.
- **Goal**             : Create models that improve automatically as they get more data.

## Traditional Programming vs Machine Learning

| **Traditional Programming**           | **Machine Learning**                       |
| ------------------------------------- | ------------------------------------------ |
| Rules are explicitly coded by humans. | Model learns rules from data.              |
| Input + Rules $\to$ Output                | Input + Output $\to$ Model (rules)             |
| Used when logic is well-known.        | Used when logic is complex or unknown.     |

Example:
- Traditional: Writing rules to detect spam emails manually.
- ML: Feed thousands of labeled emails $\to$ model learns spam patterns automatically.

## Understanding a ML Problem

Before starting any ML project:

1. Define the Problem – What do we want to predict?
2. Understand Data – What features are available? What’s the target variable?
3. Identify ML Type – Regression, Classification, Clustering, etc.
4. Decide Success Metric – Accuracy, RMSE, etc.

## Steps in a ML Project

1. Data Collection
2. Data Preprocessing (cleaning, encoding, scaling, etc.)
3. Splitting into Train & Test sets
4. Choosing Model (Linear Regression, Decision Tree, etc.)
5. Training the Model
6. Evaluating Performance
7. Hyperparameter Tuning
8. Deploying Model

## Basic Terms in ML

- Feature / Variable / Input (X) – Information used for prediction.
- Target / Label / Output (y) – What we want to predict.
- Model – Mathematical function that maps input $\to$ output.
- Training – Model learns from data.
- Testing – Evaluating model on unseen data.
- Overfitting – Model memorizes training data but fails on new data.
- Underfitting – Model is too simple, performs poorly on both train & test.

## Types of Machine Learning

1. **Supervised Learning** – Model learns with labeled data (X $\to$ y).
    - Regression: Predict continuous values (house price).
    - Classification: Predict categories (spam/ham).

2. **Unsupervised Learning** – No labels, model finds patterns.
    - Clustering: Group similar items (customer segmentation).
    - Dimensionality Reduction: Compress data (PCA).

3. **Reinforcement Learning** – Model learns by interacting with environment (like a game agent learning by trial & error).

##  What is Data Preprocessing?

- Raw data is often incomplete, noisy, inconsistent.
- Preprocessing cleans and transforms data so ML models can learn better.

## General steps

1. **Handling Missing Values**
    - Types of Missing Data:
        1. **Standard missing values**     : Represented as `Nan`, `Null`, empty space
        2. **Non standard missing values** : Represented as `?`, `-`, `Not available`
    - Methods of missing values: 
        1. Remove rows/columns
        2. Impute values
            - Mean/median (for numeric)
            - Mode (for categorical)
            - Predict missing value using another model (advanced).

2. **Handling Non-Numeric Data**
    - ML models cannot handle text directly, so convert to numbers.
        - One-Hot Encoding: Creates new columns (0/1) for each category.
        - Label Encoding: Assigns integer codes (0,1,2...) to categories.
        - Ordinal Encoding: Similar to label encoding but preserves order (e.g., low < medium < high).

3. **Normalization & Transformation**
    - Normalization: Scale values between 0 and 1 $\to$ `x' = (x-min)/(max-min)`
    - Standardization: Transform to mean=0, std=1 $\to$ `z = (x-mean)/std`
    - Why? Models like regression and SVM are sensitive to scale.

4. **Outlier Detection & Removal**
    - Outliers can badly affect models (esp. linear regression).
        - Boxplot Method: Values beyond whiskers.
        - IQR Method:
            - IQR = Q3 - Q1
            - Outliers: < Q1 - 1.5*IQR or > Q3 + 1.5*IQR.
        - Z-Score Method: |z| > 3.
        - Scatterplot: Visual inspection.

5. **Feature Engineering (Intro)**
    - Creating new features from existing ones. Example: BMI = weight / height².
    - Includes:
        - Feature extraction (PCA, text embeddings)
        - Feature transformation (log, square root)
        - Combining features (ratios, interactions).

6. **Train-Test Split**
    - **Purpose**: To evaluate model on unseen data.
    - Common split: 70% Train / 30% Test or 80/20.
    - Use train_test_split() in scikit-learn.

## Applications of ML

ML is used everywhere, some examples:
- Healthcare: Disease prediction, Image analysis
- Finance: Credit scoring, fraud detection
- Retail: Recommendation systems, inventory forecasting
- Transportation: Rote optimisation, self driving cars
- Manufacturing: Predictive maintanence, defect detection

---

# C2: BASICS OF LINEAR REGRESSION

## What is linear regression

- **Definition :** A supervised learning algorithm used to predict a continous value by finding the best fitting straught line through the data
- **Equation :**<br>
**Simple Linear Regression :** y = $b_0 + b_1x$
     - y = Predicted value
     - x = input feature
     - $b_0$ = Intercept
     - $b_1$ = Slope (Change in y when x increases by 1)

## Simple linear regression

- One input feature 
- Eg: Predicting house price based only on house size

## Multiple linear regression

- More than one input feature
- Equation: y = $b_0 + b_1x_1 + b_2x_2, ..., b_nx_n$
- Eg: Predicting house price based on number of bedrooms, house size, location score etc.

## Covariance and correlation

- **Covariance :** Shows direction of relationship(positive/negative) between two variables
    - Formula:<br> 
    $ \mathrm{Cov}(X,Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1} $
    - Positive: both increase together
    - Negative: one increases and other decreases

- **Correlation :** Standardized form of covariance, range from -1 to +1
    - Formula: <br>
    $\mathrm{r} = \dfrac{Cov(X,Y)}{\sigma_x\sigma_y
    }$
    - r = 1 $\to$ perfect positive
    - r = -1 $\to$ perfect negative
    - r = 0 $\to$ No linear relationship

## Regression analysis

- Goal: Find coefficients $b_0, b_1, ..., b_n$ that minimize the prediction error

## Ordinary least squares

- Method tp find the best-fitting line by minimizing sum of squared errors: <br>
$\mathrm{SSE} = \sum (y_i - \hat{y_i})^2$
- The line with minimium SSE is chosen

## $R^2$ and Adjusted $R^2$

- $R^2$ (Coeffiecient of determination):
    - Measures how much variance in y is explained by the model
    - Range: 0 to 1 (higher is better)
    - Formula: <br>
    $\mathrm{R^2} = 1- \dfrac{SSE}{SST}$
- Adjusted $R^2$: Corrects $R^2$ for multiple predictors.
    - Avoids artificial inflation when adding irrelevant variables

## Inferences and slope
- Hypothesis testing for slope: 
    - $H_0: b_1 = 0 \to$ no relationship between x and y
    - $H_a: b_1 \neq 0 \to$relationship exists
- If p-value < significance level (eg: 0.05), reject $H_0$.

## Linear regression with time series (Autoregression)

- Predicts a variable using its past values
- Example: Predict today's stock price based on last 5 days
