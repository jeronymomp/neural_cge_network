# Neural Estimation via Computable General Equilibrium (CGE) Inspired Model

This project explores an innovative method to estimate the internal weights of a neural network using the conceptual structure of a Computable General Equilibrium (CGE) model. Instead of relying solely on backpropagation, we treat each neuron (or layer) as an economic sector, where inputs are transformed into outputs through chained linear equations. This structure enables direct analytical estimation of parameters before comparing them with traditional neural network training.

## 🧠 Concept

The neural network is represented as a series of linear transformations:

```
x (input) → z1 (sector 1) → z2 (sector 2) → y (final output)
```

Each sector is modeled as:

```
z1 = a1 + b1 * x  
z2 = a2 + b2 * z1  
y  = a3 + b3 * z2
```

These equations are optimized by minimizing the mean squared error between predicted and observed outputs, giving us an economic-inspired way to approximate neural weights.

## 📊 Comparison with Backpropagation

After estimating the parameters via this CGE-style model, we compare it with a neural network trained with PyTorch using backpropagation. Both models are trained and evaluated on the same synthetic data.

## 📈 Visual Outputs

The project includes side-by-side visualizations of:
- Real vs. predicted outputs for both models
- Residuals (errors)
- Accuracy metrics (R² and RMSE)

You can expect plots like the following (to be added):
- ![Prediction Comparison](images/prediction_comparison.png)
- ![Residual Plot](images/residuals.png)

## 📁 Project Structure

```
/cge_neural_estimation
├── cge_model.py         # CGE-based estimation logic
├── neural_net.py        # PyTorch or sklearn-based model
├── metrics.py           # Accuracy measures like R², RMSE
├── generate_data.py     # Synthetic data generation
├── plots/               # Output images and comparisons
└── main.py              # Script to run the full pipeline
```

## 🧪 Evaluation Metrics

We evaluate both models using the following:

- **R² (Coefficient of Determination)**: How well the model explains the variance in the data.
- **RMSE (Root Mean Squared Error)**: Average deviation between predicted and true values.

```python
def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
```

## 📦 Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- PyTorch *(or alternatively scikit-learn)*

Install requirements via:
```bash
pip install -r requirements.txt
```

## 🧭 Motivation

This project bridges machine learning and economic modeling, proposing a new way to **analytically initialize** or estimate neural parameters. It may reduce the need for expensive training and offer interpretability to neural networks using economic intuition.

## 👨‍💻 Author

**Jeronymo Marcondes Pinto, PhD*  
**Martha Regina Meira Bianchi, PhD* 

---

*This project is a conceptual prototype combining CGE theory and neural networks. It aims to inspire interdisciplinary innovation between Economics and AI.*
