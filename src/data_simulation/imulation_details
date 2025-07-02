# Simulated Experiments for Model Behavior Analysis

This repository contains a series of controlled experiments using simulated datasets designed to evaluate and compare the behavior of linear and nonlinear regression models under different conditions. The goal is to better understand how the choice of model and feature set influences predictive performance and generalization.

## 📁 Structure of Experiments

Each experiment varies in terms of the data-generating process (linear or nonlinear), number of features, and proportion of relevant features used. Below is a detailed description of each simulation scenario:

---

### 1️⃣ Linear Regression (All Features Used)
- **Type**: Linear model
- **Features**: `x` features selected randomly
- **Target**: All `x` features contribute to the target
- **Purpose**: Understand model performance when all features are relevant

---

### 2️⃣ Linear Regression (Subset of Features Used)
- **Type**: Linear model
- **Features**: `x` features selected randomly
- **Target**: Only `y < x` features contribute to the target
- **Purpose**: Simulate scenarios with irrelevant/noisy features

---

### 3️⃣ Nonlinear Multiplicative Model (All Features Used)
- **Type**: Nonlinear multiplicative model
- **Features**: `x` features selected randomly
- **Target**: All features interact multiplicatively (e.g., `y = x1 * x2 * ... * xn + noise`)
- **Purpose**: Evaluate performance on complex feature interactions

---

### 4️⃣ Nonlinear Multiplicative Model (Subset of Features Used)
- **Type**: Nonlinear multiplicative model
- **Features**: `x` features selected randomly
- **Target**: Only `y < x` features interact multiplicatively
- **Purpose**: Introduce irrelevant features into complex nonlinear models

---

### 5️⃣ Logarithmic Model
- **Type**: Nonlinear log model
- **Form**: `y = log(a1 * x1 + a2 * x2 + ... + noise)`
- **Purpose**: Analyze model behavior when the target grows logarithmically with feature input

---

### 6️⃣ Polynomial Model
- **Type**: Nonlinear polynomial model
- **Form**: `y = a1 * x1^2 + a2 * x2^2 + ... + noise`
- **Purpose**: Observe how well models adapt to polynomial relationships in the data

---

### 7️⃣ Suggested Future Model: Piecewise Linear Model
- **Type**: Hybrid model
- **Form**: Combines linearity in segments (e.g., different slopes for feature ranges)
- **Purpose**: Simulate real-world scenarios where relationships change based on feature thresholds

---

## 🚧 Notes
- All experiments include injected noise to simulate realistic conditions.
- Features are standardized prior to modeling.
- Models are evaluated on both training and test data to assess generalization.

## 📌 Requirements
- Python 3.8+
- `numpy`, `scikit-learn`, `pandas`, `matplotlib`

## ✅ Getting Started
To run the experiments:

```bash
python simulate_experiment.py --experiment 1  # Replace with desired experiment number
```

---

Happy experimenting!
