# Customer Churn Prediction

## 1. Project Overview

This project predicts whether a telecom customer is likely to churn (leave the company) using machine learning.

The project uses the IBM Telco Customer Churn dataset and a Decision Tree Classifier to identify customers who are more likely to leave.

The project covers:

* Data understanding and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Data preprocessing
* Decision Tree model training
* Hyperparameter tuning
* Model evaluation
* Feature importance analysis
* Model visualization
* Saving the trained model
* REST API using FastAPI
* Churn prediction and churn probability

## Project Links

* **GitHub Repository:** https://github.com/dhpadeveloper/customer_churn_project
* **Demo Video:** https://nagarro-my.sharepoint.com/:v:/p/harsh_bhagwani/IQA4gZC_dOFrQ5DWabsSgd4KASMTr3bIxqy9F4JjKDN9KvE?e=bJ8kda

---

## 2. Data Understanding & Preparation

The project uses the IBM Telco Customer Churn Dataset.

The dataset was inspected to understand its structure, columns, data types, and sample records.

### Observations

The dataset contains customer-level information including:

* Contract type
* Customer tenure
* Internet services
* Additional services
* Payment method
* Monthly charges
* Total charges
* Churn status

### Target Variable

The target variable is `Churn`.

It contains two values:

* **Yes:** Customer churned
* **No:** Customer did not churn

For machine learning, these values are converted into numerical classes:

* **Yes:** `1`
* **No:** `0`

### Missing Values

The `TotalCharges` column contained blank whitespace entries that were converted to numeric format so they could be properly identified as missing data.

These rows were then dropped during data cleaning to ensure the dataset remained clean and reliable before model training.

### Duplicate Analysis

Duplicate records were checked during data preparation.

### Customer ID

`customerID` is a unique identifier and does not provide useful predictive information, so it is excluded from model training.

### Numerical and Categorical Features

The dataset contains two types of information.

#### Numerical Features

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`
* `NumberOfServices`

#### Categorical Features

* `gender`
* `Partner`
* `Dependents`
* `InternetService`
* `Contract`
* `PaymentMethod`
* `tenure_grp`

### Train / Test Split

To build and test the model fairly, the dataset is split into two parts:

* **70% Training:** Used by the model to learn patterns.
* **30% Testing:** Kept separate to evaluate how well the model predicts on new, unseen customers.

The split uses:

```python
train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=42,
    stratify=y
)
```

* `random_state=42`: Guarantees reproducibility so the same split is obtained every time.
* `stratify=y`: Maintains a similar proportion of churned and non-churned customers in both training and testing datasets.

### Data Preprocessing

Different features need different treatment. A Scikit-Learn `ColumnTransformer` is used to automatically apply the appropriate preprocessing steps to numerical and categorical columns.

### Converting Text to Numbers — One-Hot Encoding

Machine learning models work with numerical values rather than text categories. One-Hot Encoding is used to convert categorical columns into numerical columns.

For example, the `Contract` column contains:

* Month-to-month
* One year
* Two year

These are converted into separate binary columns:

* `Contract_Month-to-month`
* `Contract_One year`
* `Contract_Two year`

This allows the Decision Tree to use categorical information without incorrectly treating categories as numerical rankings.

### Preventing Data Leakage

Data leakage happens when a model accidentally uses information from the test data during training. This can make the model appear artificially accurate and perform poorly on real-world data.

To prevent this, the preprocessing steps and model are placed inside a Scikit-Learn `Pipeline`.

This ensures that:

* Preprocessing rules are learned only from the training data.
* The same preprocessing rules are applied to the test data and future customers.

---

## 3. Exploratory Data Analysis

EDA was performed to understand customer behaviour and identify patterns associated with churn.

### Churn Distribution

Shows the number of customers who churned and did not churn.

**Business Insight:**

The churn distribution provides an initial understanding of the size of the customer group that the retention team needs to focus on.

If churn represents a smaller proportion of the customer base, the company should pay particular attention to Precision, Recall, and F1-score, rather than relying only on accuracy.

### Internet Service vs Churn

Shows the distribution of customers across different internet service types.

**Business Insight:**

If a particular internet service category has a higher proportion of churn, the company could investigate potential reasons such as:

* Pricing
* Service quality
* Customer experience
* Technical issues

The company could then design targeted retention strategies for customers using that service.

### Contract Type vs Churn

The visualization compares churned and non-churned customers across different contract types.

**Business Insight:**

If month-to-month customers show substantially higher churn, the company could focus retention campaigns on these customers.

Possible business strategies could include:

* Contract upgrade offers
* Discounts for longer commitments
* Personalized retention offers

### Customer Tenure Distribution

This visualization shows how customers are distributed across different tenure values.

**Business Insight:**

Understanding the tenure distribution helps the company identify whether its customer base contains a large population of new customers who may require stronger onboarding and early-retention strategies.

### Monthly Charges by Churn

The visualization examines the distribution of monthly charges for churned and non-churned customers.

**Business Insight:**

If churned customers show higher monthly charges, the company could investigate whether pricing is contributing to customer dissatisfaction.

---

## 4. Feature Engineering

Two additional features were created.

### 4.1 Tenure Group

#### How was it created?

The original dataset contains `tenure`, which represents the number of months a customer has stayed with the company.

Customers were grouped into meaningful tenure categories:

```python
def tenure_group(tenure):
    if tenure <= 12:
        return "New"
    elif tenure <= 36:
        return "Medium"
    else:
        return "Long-term"
```

#### Why it may be useful

Customer behaviour may differ depending on how long a customer has stayed with the company.

* **New Customers:** May require additional onboarding and retention attention.
* **Medium-Tenure Customers:** May have different retention needs as their relationship with the company develops.
* **Long-Term Customers:** May have stronger relationships with the company.

The `tenure_grp` feature gives the Decision Tree a broader representation of customer tenure rather than relying only on individual tenure values.

### 4.2 Number of Services

#### How was it created?

The following additional service columns were used:

* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`

The number of services for which the customer has `"Yes"` is counted.

For example:

```text
NumberOfServices = 4
```

means the customer uses four of these additional services.

#### Why it may be useful

The original dataset contains multiple individual service columns.

`NumberOfServices` provides a simple summary of the customer's overall service adoption.

This may help represent the customer's level of engagement with the company's services.

---

## 5. Model Development

A Decision Tree Classifier was trained on the training dataset.

* **Model 1:** Used `criterion="gini"` and `max_depth=5`.
* **Model 2:** Used `GridSearchCV` to tune `max_depth`, `criterion`, and `class_weight`.
* **`class_weight`:** Included because the dataset is imbalanced. Without balancing, the Decision Tree may favor the majority `No Churn` class and achieve high accuracy while missing actual churn customers. `class_weight="balanced"` gives more importance to the `Churn` class, helping the model identify more churn cases.
* **GridSearch:** Tested `max_depth = [3, 4, 5, 6, 8, 10]`, `criterion = ["gini", "entropy"]`, and `class_weight = [None, "balanced"]` using 4-fold cross-validation. F1-score was used by GridSearch to select the best hyperparameter combination.
* **Recall:** Given particular attention when comparing Model 1 and Model 2 because high recall means identifying a larger proportion of customers who actually churn. This can help the company take retention actions for more at-risk customers and potentially reduce customer and revenue loss.

---

## 6. Model Evaluation

The final model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Since the dataset is imbalanced, accuracy alone is not sufficient.

Model 1 achieved higher accuracy, but its Recall was lower, meaning it missed a larger number of actual churn customers.

After using `class_weight="balanced"` in Model 2, Recall improved significantly, allowing the model to identify more customers who are actually likely to churn.

From a business perspective, Recall is particularly important for a telecom company.

High Recall means detecting a larger proportion of actual churn customers, giving the company more opportunities to take retention actions before customers leave and the company loses their future revenue.

Therefore, Recall was given greater importance when evaluating the models, while Precision, F1-score, Accuracy, and the Confusion Matrix were also considered.

### Precision vs. Recall

For this churn prediction problem, Recall is prioritized over Precision.

* **False Negative (Missed Churner):** The model predicts `No Churn`, but the customer actually leaves. The company may lose the customer and their future revenue.
* **False Positive (False Alarm):** The model predicts `Churn`, but the customer would have stayed. The company may spend money on an unnecessary retention offer.
* **Business Consideration:** Missing an actual churn customer can have a greater business impact than contacting a customer who was not going to churn.

---

## 7. Model Interpretation

Feature importance was used to understand which features contributed most to the Decision Tree's churn predictions.

### Top Features Influencing Churn

| Feature                        | Importance |
| ------------------------------ | ---------: |
| Contract – Month-to-month      |     63.86% |
| Tenure                         |     10.99% |
| Internet Service – Fiber optic |     10.99% |
| Monthly Charges                |      4.10% |
| Online Security – No           |      3.61% |

### Key Findings

* **Contract (Month-to-month)** was the most influential feature, contributing approximately 63.86% of the total feature importance.
* **Tenure** and **Internet Service – Fiber optic** were the next major features, each contributing approximately 11%.
* **Monthly Charges** and **Online Security – No** also contributed to the model's predictions.
* The Decision Tree visualization showed similar patterns, with contract type, tenure, and internet service appearing in important decision splits.
* Overall, the model indicates that contract type, customer tenure, and internet service are the main factors associated with its churn predictions.

> These are model associations and should not be interpreted as proof that these factors directly cause churn.

---

## 8. Setup and Installation

### 8.1 Clone the Project

```bash
git clone https://github.com/dhpadeveloper/customer_churn_project.git
cd customer_churn_project
```

### 8.2 Create and Activate Virtual Environment

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 8.3 Install Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 8.4 Run the FastAPI Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

The API will start locally and can be accessed at:

http://127.0.0.1:8000

### 8.5 API Documentation

FastAPI automatically provides interactive API documentation at:

http://127.0.0.1:8000/docs

The Swagger UI can be used to test the `/predict` endpoint by providing customer information as JSON input.
