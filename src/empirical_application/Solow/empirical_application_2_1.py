#--------
# Working Directory
import os
wkdir = "/home/jeronymo/Downloads/neural_cge_network"
os.chdir(wkdir)

# =============================================================================
# 1 - Packages
# =============================================================================
import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

from src.functions.feature_impact import compute_linear_regression_feature_impact_percent, topk_pairwise_rank_agreement, topk_rank_correlation
from src.functions.feature_impact import cobb_douglas_log_impact, model_alphas, compute_shap_impact_percent
from src.functions.feature_impact import compare_importance_cosine, compare_importance_spearman
from src.functions.neural_network_function import FlexibleNeuralNetwork
from src.functions.log_cobb_douglas_act_function import CobbDouglasLogActivationFunction
from src.functions.log_cobb_doulgas_neural_network_function import LogCobbDouglasNeuralNetwork
from src.functions.result_analysis import generate_average_metrics_with_ci

# =============================================================================
# 2 - Parameters
# =============================================================================
result_list = []
random_state = 0
n_samples = 30
g = 0.02
delta = 0.03

start_year = 1971
end_year = 2019

# Neural Network Parameters
epochs_nas = 500
epochs = 30000
learning_rate = 0.0001
learning_rate_cobb = 0.0001
tries_before_stopping = 1000   # FIX: era 10; com early stopping "de verdade" agora
                              # (valid != train), 10 é agressivo demais e pode
                              # parar antes da rede convergir.

# FIX: normalized=0 — sem restrição de soma=1/sinal positivo via softmax.
# A teoria (Solow) exige coeficiente de n negativo; softmax proibia isso.
sum_equal_one = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# 3 - Import PWT Data
# =============================================================================
wkdir = "/home/jeronymo/Downloads/neural_cge_network/src/empirical_application"
os.chdir(wkdir)

# FIX: csh_i (participação do investimento no PIB) no lugar de rnna (estoque
# de capital). rnna não é a variável "s" do modelo de Solow.
variables = ["countrycode", "year", "rgdpna", "pop", "csh_i"]

pwt_data = pd.read_excel('pwt_data.xlsx')
df = pwt_data[variables]
df = df[df['year'] >= start_year]
df["Y_per_L"] = df["rgdpna"] / df["pop"]

df_period = pwt_data[(pwt_data["year"] >= 1960) & (pwt_data["year"] <= 1985)].copy()

pop_growth = (
    df_period.groupby("countrycode")["pop"]
    .agg(["first", "last"])
    .reset_index()
)
pop_growth["n"] = (np.log(pop_growth["last"]) - np.log(pop_growth["first"])) / (1985 - 1960)

# FIX: média de csh_i, não de rnna
inv_mean = (
    df_period.groupby("countrycode")["csh_i"]
    .mean()
    .reset_index()
    .rename(columns={"csh_i": "s"})
)

reg_df = df[["countrycode", "Y_per_L"]].merge(pop_growth[["countrycode", "n"]], on="countrycode")
reg_df = reg_df.merge(inv_mean, on="countrycode")

reg_df = reg_df[(reg_df["Y_per_L"] > 0) & (reg_df["s"] > 0) & (reg_df["n"] > 0)]
reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna()

reg_df['n'] = reg_df['n'] + g + delta

reg_df["ln_Y_per_L"] = np.log(reg_df[["Y_per_L"]])
reg_df["ln_s"] = np.log(reg_df[["s"]])
reg_df["ln_n"] = np.log(reg_df[["n"]])

reg_df = reg_df.groupby('countrycode', as_index=False).mean()

X = reg_df[["s", "n"]]
ln_X = reg_df[["ln_s", "ln_n"]]
ln_y = reg_df["ln_Y_per_L"]

n_obs = len(reg_df)
number_of_features = len(X.columns)
print(f"N países após merge/dropna: {n_obs}")

# =============================================================================
# 3.5 - Train / Valid / Test Split (real, não mais o mesmo array 3x)
# =============================================================================
# FIX: split de verdade. Com N pequeno, considere trocar depois por
# K-Fold/Leave-One-Out — aqui mantemos hold-out simples para não fugir
# da estrutura original do experimento (random search + treino final).
idx = np.arange(n_obs)
idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=random_state)
idx_valid, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_state)
print(f"Split -> train: {len(idx_train)}, valid: {len(idx_valid)}, test: {len(idx_test)}")

X_full = X.values.astype(np.float32)              # níveis (s, n) — para Cobb-Douglas
ln_X_full = ln_X.values.astype(np.float32)         # log — para FNN e OLS
ln_y_full = ln_y.values.reshape(-1, 1).astype(np.float32)

# Cobb-Douglas: entrada em nível, alvo em LOG (a loss log-espaço cuida do exp)
X_train, X_valid, X_test = X_full[idx_train], X_full[idx_valid], X_full[idx_test]
y_train, y_valid, y_test = ln_y_full[idx_train], ln_y_full[idx_valid], ln_y_full[idx_test]

# FNN: tudo em log
ln_X_train, ln_X_valid, ln_X_test = ln_X_full[idx_train], ln_X_full[idx_valid], ln_X_full[idx_test]
ln_y_train, ln_y_valid, ln_y_test = ln_y_full[idx_train], ln_y_full[idx_valid], ln_y_full[idx_test]

ln_X_test_tensor = torch.from_numpy(ln_X_test).float().to(device)

# =============================================================================
# 3.6 - Loss em log-espaço para a Cobb-Douglas NN
# =============================================================================
# FIX: a camada Cobb-Douglas retorna exp(sum alpha*ln(x)+bias), previsão em
# NÍVEL. Comparando via MSE de nível, a rede otimiza um objetivo diferente
# da OLS (que minimiza erro em log) e super-pondera países ricos. Esta loss
# tira o log da previsão antes de comparar com ln(y), igualando os objetivos.
class LogSpaceMSELoss(nn.Module):
    def forward(self, pred, target_log):
        pred = torch.clamp(pred, min=1e-6)
        return nn.functional.mse_loss(torch.log(pred), target_log)

# =============================================================================
# 4 - Linear Regression (treinada e avaliada no MESMO split)
# =============================================================================
# FIX: treina só no conjunto de treino, avalia no de teste — antes via 100%
# dos dados, então não havia "generalização" real sendo medida.
model = LinearRegression()
model.fit(ln_X_train, ln_y_train)
y_pred_test_total = model.predict(ln_X)
y_pred_test = model.predict(ln_X_test)
mse_regression = mean_squared_error(ln_y_test, y_pred_test)
mae_regression = mean_absolute_error(ln_y_test, y_pred_test)
r2_regression = r2_score(ln_y, y_pred_test_total)

print(f"OLS  ->  MSE: {mse_regression:.4f}  MAE: {mae_regression:.4f}  R2: {r2_regression:.4f}")
print(f"OLS coefs (s, n): {model.coef_.flatten()}")

# =============================================================================
# 5 - Random Search for Base NN (FNN, em log-espaço)
# =============================================================================
def generate_random_config(number_of_features):
    number_of_layers = random.randint(1, 3)          # FIX: teto 3 (era 4) — N pequeno
    max_neurons = min(16, 4 * number_of_features)     # FIX: teto 16 (era 32)
    hidden_layers = sorted(
        [random.randint(number_of_features, max_neurons) for _ in range(number_of_layers)],
        reverse=True
    )
    dropout = random.choice([0.0, 0.1, 0.2, 0.3])
    activation_function = random.choice([nn.ReLU, nn.Tanh, nn.GELU])
    return {
        "neural_network_hidden_layers": hidden_layers,
        "dropout": dropout,
        "activation": activation_function
    }

random_configs = [generate_random_config(number_of_features) for _ in range(n_samples)]

result_random_search = []
for random_param in random_configs:
    fnn = FlexibleNeuralNetwork(ln_X_train, ln_X_valid, ln_X_test, ln_y_train, ln_y_valid, ln_y_test, device=device)
    fnn_model, *_ , fnn_test_loss, _, _ = fnn.train_model(
        hidden_layers=random_param['neural_network_hidden_layers'],
        activation_fn=random_param['activation'],
        dropout_rate=random_param['dropout'],
        use_batchnorm=False,       # FIX: batchnorm com N pequeno em batch único é instável
        epochs=epochs_nas,
        early_stopping_patience=tries_before_stopping,
        optimizer_kwargs={"lr": learning_rate}
    )
    fnn_model.to(device)
    predicted_nn = fnn_model(ln_X_test_tensor).detach().cpu().numpy()
    mse_nn = mean_squared_error(ln_y_test, predicted_nn)
    result_random_search.append(mse_nn)

best_architecture = result_random_search.index(min(result_random_search))
neural_network_hidden_layers = random_configs[best_architecture]['neural_network_hidden_layers']
dropout_rt = random_configs[best_architecture]['dropout']
activation_fn = random_configs[best_architecture]['activation']
print(f"Best FNN arch: {neural_network_hidden_layers}, dropout={dropout_rt}")

# =============================================================================
# 6 - Basic Neural Network (final)
# =============================================================================
fnn = FlexibleNeuralNetwork(ln_X_train, ln_X_valid, ln_X_test, ln_y_train, ln_y_valid, ln_y_test, device=device)
fnn_model, fnn_train_losses, fnn_val_losses, fnn_best_val, fnn_test_loss, fnn_architecture, fnn_weights = fnn.train_model(
    hidden_layers=neural_network_hidden_layers,
    activation_fn=activation_fn,
    dropout_rate=dropout_rt,
    use_batchnorm=False,
    epochs=epochs,
    early_stopping_patience=tries_before_stopping,
    optimizer_kwargs={"lr": learning_rate}
)
fnn_model.to(device)

# =============================================================================
# 7 - Random Search for Cobb-Douglas NN
# =============================================================================
def generate_random_cobb_config(number_of_features):
    number_of_layers = random.randint(1, 2)           # FIX: teto 2 (era 4) — composição
                                                        # de muitas camadas Cobb-Douglas
                                                        # complica a extração do alpha efetivo
    max_neurons = min(8, 4 * number_of_features)       # FIX: teto 8 (era 32)
    hidden_layers = sorted(
        [random.randint(number_of_features, max_neurons) for _ in range(number_of_layers)],
        reverse=True
    )
    dropout = random.choice([0.0, 0.1, 0.2])
    return {"neural_network_hidden_layers": hidden_layers, "dropout": dropout}

random_configs_cobb = [generate_random_cobb_config(number_of_features) for _ in range(n_samples)]

result_random_search_cobb = []
for random_param in random_configs_cobb:
    cobb_fnn = LogCobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device=device)
    cobb_model, *_ , cobb_test_loss, _, cobb_weights_tmp = cobb_fnn.train_model(
        hidden_layers=random_param['neural_network_hidden_layers'],
        loss_fn=LogSpaceMSELoss,                       # FIX: loss em log-espaço
        dropout_rate=random_param['dropout'],
        use_batchnorm=False,
        early_stopping_patience=tries_before_stopping,
        epochs=epochs_nas,
        normalized=sum_equal_one,                       # FIX: 0, sem softmax
        optimizer_kwargs={"lr": learning_rate_cobb}
    )
    result_random_search_cobb.append(cobb_test_loss)

best_cobb_architecture = result_random_search_cobb.index(min(result_random_search_cobb))
cobb_hidden_layers = random_configs_cobb[best_cobb_architecture]['neural_network_hidden_layers']
cobb_dropout_rt = random_configs_cobb[best_cobb_architecture]['dropout']
print(f"Best Cobb-Douglas arch: {cobb_hidden_layers}, dropout={cobb_dropout_rt}")

# =============================================================================
# 8 - Cobb-Douglas Neural Network (final)
# =============================================================================
cobb_fnn = LogCobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device=device)
cobb_model, cobb_train_losses, cobb_val_losses, cobb_best_val_loss, cobb_test_loss, cobb_architecture, cobb_weights = cobb_fnn.train_model(
    hidden_layers=cobb_hidden_layers,
    loss_fn=LogSpaceMSELoss,
    dropout_rate=cobb_dropout_rt,
    use_batchnorm=False,
    early_stopping_patience=tries_before_stopping,
    epochs=epochs,
    normalized=sum_equal_one,
    optimizer_kwargs={"lr": learning_rate_cobb}
)
cobb_model.to(device)

# =============================================================================
# 9 - Accuracy (Calculado para a amostra INTEIRA - In-Sample)
# =============================================================================
# 1. Tensores da amostra total
X_full_tensor = torch.from_numpy(X_full).float().to(device)
ln_X_full_tensor = torch.from_numpy(ln_X_full).float().to(device)

# 2. Previsões da amostra total (FNN e Cobb-Douglas)
predicted_nn_total = fnn_model(ln_X_full_tensor).detach().cpu().numpy()

predicted_cobb_nn_level_total = cobb_model(X_full_tensor).detach().cpu().numpy()
predicted_cobb_nn_log_total = np.log(np.clip(predicted_cobb_nn_level_total, 1e-6, None))

# 3. Cálculo das métricas comparando contra ln_y_full (Amostra inteira em log)
mae_nn = mean_absolute_error(ln_y_full, predicted_nn_total)
mae_cobb_nn = mean_absolute_error(ln_y_full, predicted_cobb_nn_log_total)

mse_nn = mean_squared_error(ln_y_full, predicted_nn_total)
mse_cobb_nn = mean_squared_error(ln_y_full, predicted_cobb_nn_log_total)

r2_nn = r2_score(ln_y_full, predicted_nn_total)
r2_cobb = r2_score(ln_y_full, predicted_cobb_nn_log_total)

# 4. Cálculo para a regressão (já que a OLS usou a amostra toda no ln_X_train)
# (Se a OLS tiver sido treinada só no train, isso garante o R2 global)
y_pred_ols_total = model.predict(ln_X_full)
mse_regression = mean_squared_error(ln_y_full, y_pred_ols_total)
mae_regression = mean_absolute_error(ln_y_full, y_pred_ols_total)
r2_regression = r2_score(ln_y_full, y_pred_ols_total)

# 5. Prints
print(f"FNN   -> MSE: {mse_nn:.4f}  MAE: {mae_nn:.4f}  R2: {r2_nn:.4f}")
print(f"Cobb  -> MSE: {mse_cobb_nn:.4f}  MAE: {mae_cobb_nn:.4f}  R2: {r2_cobb:.4f}")
print(f"OLS   -> MSE: {mse_regression:.4f}  MAE: {mae_regression:.4f}  R2: {r2_regression:.4f}")

# =============================================================================
# 10 - Sensitivity Analysis
# =============================================================================
real_feature_impact = model.coef_.flatten()   # elasticidades da OLS (referência)

min_val, max_val = ln_X_train.min(), ln_X_train.max()
random_feature_impact = np.random.uniform(low=min_val, high=max_val, size=real_feature_impact.shape)

# FIX: normalized=False no model_alphas/impact, coerente com normalized=0 no treino
cobb_layers = model_alphas(cobb_weights, normalized=False)
cobb_douglas_feature_impact = np.array([
    cobb_douglas_log_impact(cobb_layers, i, normalize=False)
    for i in range(cobb_layers[0].shape[1])
], dtype=float)

neural_network_feature_impact = []
for i in range(number_of_features):
    x_sens = ln_X_test_tensor.clone()
    x_sens[:, i] *= 1.01
    base_pred = fnn_model(ln_X_test_tensor)
    pred_sens = fnn_model(x_sens)
    diff = (pred_sens - base_pred) / base_pred
    neural_network_feature_impact.append((diff.mean() / 0.01).item())
neural_network_feature_impact = np.array(neural_network_feature_impact, dtype=float)

shap_impact_percent = np.array(
    compute_shap_impact_percent(fnn_model, ln_X_train, ln_X_test, ln_y_train), dtype=float
).ravel()

ln_X_train_tensor = torch.from_numpy(ln_X_train).float().to(device)

ig = IntegratedGradients(fnn_model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(fnn_model)
gs = GradientShap(fnn_model)
fa = FeatureAblation(fnn_model)

importances_ig = ig.attribute(ln_X_test_tensor, n_steps=50).abs().mean(dim=0).detach().cpu().numpy()
importances_nt = ig_nt.attribute(ln_X_test_tensor).abs().mean(dim=0).detach().cpu().numpy()
importances_dl = dl.attribute(ln_X_test_tensor).abs().mean(dim=0).detach().cpu().numpy()
importances_gs = gs.attribute(ln_X_test_tensor, ln_X_train_tensor).abs().mean(dim=0).detach().cpu().numpy()
importances_fa = fa.attribute(ln_X_test_tensor).abs().mean(dim=0).detach().cpu().numpy()

# =============================================================================
# 11 - Results Aggregation
# =============================================================================
def _cmp(impact):
    return (
        compare_importance_spearman(real_feature_impact, impact),
        compare_importance_cosine(real_feature_impact, impact),
        topk_pairwise_rank_agreement(real_feature_impact, impact, number_of_features),
        topk_rank_correlation(real_feature_impact, impact, number_of_features),
    )

cobb_spearman, cobb_cossine, cobb_topk_pair, cobb_topk_rank = _cmp(cobb_douglas_feature_impact)
random_spearman, random_cossine, random_topk_pair, random_topk_rank = _cmp(random_feature_impact)
nn_spearman, nn_cossine, nn_topk_pair, nn_topk_rank = _cmp(neural_network_feature_impact)
nn_shap_spearman, nn_shap_cossine, nn_shap_topk_pair, nn_shap_topk_rank = _cmp(shap_impact_percent)
nn_shap_spearman_ig, nn_shap_cossine_ig, nn_shap_topk_pair_ig, nn_shap_topk_rank_ig = _cmp(importances_ig)
nn_shap_spearman_dl, nn_shap_cossine_dl, nn_shap_topk_pair_dl, nn_shap_topk_rank_dl = _cmp(importances_dl)
nn_shap_spearman_gs, nn_shap_cossine_gs, nn_shap_topk_pair_gs, nn_shap_topk_rank_gs = _cmp(importances_gs)
nn_shap_spearman_fa, nn_shap_cossine_fa, nn_shap_topk_pair_fa, nn_shap_topk_rank_fa = _cmp(importances_fa)
nn_shap_spearman_nt, nn_shap_cossine_nt, nn_shap_topk_pair_nt, nn_shap_topk_rank_nt = _cmp(importances_nt)

# NOTE: com apenas 2 features, Spearman/topk têm baixíssima resolução
# (essencialmente "os sinais concordam?"). Trate com cautela.

lst_result = [
    ('mae_nn', mae_nn), ('mae_cobb_nn', mae_cobb_nn),
    ('mse_nn', mse_nn), ('mse_cobb_nn', mse_cobb_nn),
    ('cobb_spearman', cobb_spearman), ('cobb_cossine', cobb_cossine),
    ('cobb_topk_pair', cobb_topk_pair), ('cobb_topk_rank', cobb_topk_rank),
    ('random_spearman', random_spearman), ('random_cossine', random_cossine),
    ('random_topk_pair', random_topk_pair), ('random_topk_rank', random_topk_rank),
    ('nn_spearman', nn_spearman), ('nn_cossine', nn_cossine),
    ('nn_topk_pair', nn_topk_pair), ('nn_topk_rank', nn_topk_rank),
    ('nn_shap_spearman', nn_shap_spearman), ('nn_shap_cossine', nn_shap_cossine),
    ('nn_shap_topk_pair', nn_shap_topk_pair), ('nn_shap_topk_rank', nn_shap_topk_rank),
    ('nn_shap_spearman_ig', nn_shap_spearman_ig), ('nn_shap_cossine_ig', nn_shap_cossine_ig),
    ('nn_shap_topk_pair_ig', nn_shap_topk_pair_ig), ('nn_shap_topk_rank_ig', nn_shap_topk_rank_ig),
    ('nn_shap_spearman_dl', nn_shap_spearman_dl), ('nn_shap_cossine_dl', nn_shap_cossine_dl),
    ('nn_shap_topk_pair_dl', nn_shap_topk_pair_dl), ('nn_shap_topk_rank_dl', nn_shap_topk_rank_dl),
    ('nn_shap_spearman_gs', nn_shap_spearman_gs), ('nn_shap_cossine_gs', nn_shap_cossine_gs),
    ('nn_shap_topk_pair_gs', nn_shap_topk_pair_gs), ('nn_shap_topk_rank_gs', nn_shap_topk_rank_gs),
    ('nn_shap_spearman_fa', nn_shap_spearman_fa), ('nn_shap_cossine_fa', nn_shap_cossine_fa),
    ('nn_shap_topk_pair_fa', nn_shap_topk_pair_fa), ('nn_shap_topk_rank_fa', nn_shap_topk_rank_fa),
    ('nn_shap_spearman_nt', nn_shap_spearman_nt), ('nn_shap_cossine_nt', nn_shap_cossine_nt),
    ('nn_shap_topk_pair_nt', nn_shap_topk_pair_nt), ('nn_shap_topk_rank_nt', nn_shap_topk_rank_nt),
]
result_list.append(lst_result)

# =============================================================================
# 11.5 - NOVO: comparação DIRETA de magnitude (elasticidade vs elasticidade)
# =============================================================================
# FIX: adicionado — com só 2 features, a comparação por ranking (acima) é
# pouco informativa. Como agora ambos operam em log-espaço na mesma escala,
# faz sentido comparar diretamente os coeficientes.
df_alpha_compare = pd.DataFrame({
    "feature": ["s", "n"],
    "ols_coef": real_feature_impact,
    "cobb_nn_coef": cobb_douglas_feature_impact,
})
df_alpha_compare["abs_diff"] = (df_alpha_compare["ols_coef"] - df_alpha_compare["cobb_nn_coef"]).abs()
print("\nComparação direta de elasticidades (OLS vs Cobb-Douglas NN):")
print(df_alpha_compare)

# =============================================================================
# 11.6 - Final Results WITH CONFIDENCE INTERVALS
# =============================================================================
df, df_cosine_ci, df_spearman_ci, df_mae_ci, df_mse_ci, df_topk_pair_ci, df_topk_rank_ci = generate_average_metrics_with_ci(result_list)

wkdir = "/home/jeronymo/Downloads/neural_cge_network/src/empirical_application/Solow/Results/Experiment 2_1"
os.makedirs(wkdir, exist_ok=True)
os.chdir(wkdir)

df.to_csv('simulation_result.csv', index=False)
df_cosine_ci.to_csv('df_cosine_mean_result.csv', index=False)
df_spearman_ci.to_csv('df_spearman_mean_result.csv', index=False)
df_mae_ci.to_csv('df_mae_mean_result.csv', index=False)
df_mse_ci.to_csv('df_mse_mean_result.csv', index=False)
df_topk_pair_ci.to_csv('df_topk_pair_mean_result.csv', index=False)
df_topk_rank_ci.to_csv('df_topk_rank_mean_result.csv', index=False)
df_alpha_compare.to_csv('alpha_direct_comparison.csv', index=False)

df_coef = pd.DataFrame({
    "real_feature_impact": real_feature_impact,
    "cobb_douglas_feature_impact": cobb_douglas_feature_impact,
    "neural_network_feature_impact": neural_network_feature_impact,
    "shap_impact_percent": shap_impact_percent,
    "importances_ig": importances_ig,
    "importances_dl": importances_dl,
    "importances_gs": importances_gs,
    "importances_fa": importances_fa,
    "importances_nt": importances_nt
})
df_coef.to_csv("coefficients.csv", index=False)

# =============================================================================
# 12 - Comparative Statics (Shock Experiment)
# =============================================================================
base_x = ln_X_test.mean(axis=0).astype(np.float32)
base_x_tensor = torch.tensor(base_x).unsqueeze(0).to(device)

shock_values = np.linspace(-0.2, 0.2, 9)

responses_fnn = []
responses_cobb = []

base_x_cobb = X_test.mean(axis=0).astype(np.float32)
base_x_cobb_tensor = torch.tensor(base_x_cobb).unsqueeze(0).to(device)

for shock in shock_values:
    shocked_x = base_x.copy()
    shocked_x[0] = shocked_x[0] + shock
    shocked_x_tensor = torch.tensor(shocked_x).unsqueeze(0).to(device)

    y_base_fnn = fnn_model(base_x_tensor).item()
    y_shock_fnn = fnn_model(shocked_x_tensor).item()
    responses_fnn.append(y_shock_fnn - y_base_fnn)

    shocked_x_cobb = base_x_cobb.copy()
    shocked_x_cobb[0] = shocked_x_cobb[0] * np.exp(shock)
    shocked_x_cobb_tensor = torch.tensor(shocked_x_cobb).unsqueeze(0).to(device)

    # FIX: previsão da Cobb-Douglas agora é comparada em LOG (consistente
    # com o resto do script — a rede prevê nível, então log() na resposta)
    y_base_cobb = np.log(max(cobb_model(base_x_cobb_tensor).item(), 1e-6))
    y_shock_cobb = np.log(max(cobb_model(shocked_x_cobb_tensor).item(), 1e-6))
    responses_cobb.append(y_shock_cobb - y_base_cobb)

plt.figure()
plt.plot(shock_values, responses_fnn, marker='o', label='FNN')
plt.plot(shock_values, responses_cobb, marker='o', label='Neural IOM (Cobb)')
plt.axhline(0)
plt.xlabel("Shock in ln(s)")
plt.ylabel("Change in predicted ln(output)")
plt.legend()
plt.title("Comparative Statics: Response to Investment Shock")
plt.savefig("comparative_statics_application_1_bench.png", dpi=300, bbox_inches='tight')
plt.show()
