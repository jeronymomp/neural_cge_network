# 🚀 Solow Growth Model Experiments

This repository contains implementations and experiments with a **Solow growth model** using data from the **Penn World Table (PWT)**. The goal is to study the determinants of **GDP per capita growth** across countries.

## 📊 Variables Tested
We tested the model using the following explanatory variables against GDP per capita growth:
- 🌱 **Population growth rate** (`pop_growth`)
- 💰 **Investment rate** (`inv_rate`)
- 🏦 **Initial income** (`initial_income`)

## 🧪 Experiment Design
A total of **three sets of experiments** were conducted:
1. ✅ Using all three variables: population growth, investment rate, and initial income.
2. ⚡ Using population growth and initial income, excluding investment rate.
3. 🌟 Using population growth and investment rate, excluding initial income.

Each experiment was **repeated three times** to ensure robustness.

## 📥 Data Source
- **Penn World Table (PWT)** – version X.X
- Variables: GDP per capita growth, population growth, investment rate, initial income.

## ⚙️ Usage
1. Load the dataset from PWT.
2. Specify which combination of variables to test.
3. Fit the Solow growth model.
4. Evaluate model performance and compare results across different specifications.

## 📝 Notes
This setup allows evaluating the **marginal contribution** of each variable to economic growth and understanding the robustness of the Solow model predictions under different variable combinations.


