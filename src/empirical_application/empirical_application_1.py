#--------
# Working Directory
import os
wkdir = "/home/jeronymo/Documentos/neural_cge_network"
os.chdir(wkdir)

# =============================================================================
# 1 - Packages
# =============================================================================
# External Packages
import time
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

# Built-in
from src.functions.linear_data_generator import LinearDataGenerator
from src.functions.feature_impact import compute_linear_regression_feature_impact_percent
from src.functions.feature_impact import cobb_douglas_impact,cobb_douglas_log_impact,model_alphas,compute_shap_impact_percent
from src.functions.feature_impact import compare_importance_cosine,normalize_importance,compare_importance_spearman
from src.functions.neural_network_function import FlexibleNeuralNetwork
from src.functions.cobb_douglas_act_function import CobbDouglasActivationFunction
from src.functions.log_cobb_douglas_act_function import CobbDouglasLogActivationFunction
from src.functions.cobb_doulgas_neural_network_function import CobbDouglasNeuralNetwork
from src.functions.log_cobb_doulgas_neural_network_function import LogCobbDouglasNeuralNetwork
from src.functions.result_analysis import generate_average_metrics_separated

# =============================================================================
# 2 - Parameters
# =============================================================================

#--------
# 2.1 - Experiment Parameters
# Starting and ending year
start_year = 1971
end_year = 2019
n_years = end_year - start_year

# The variable are:
    # rgdpo	- Output-side real GDP at chained PPPs (in mil. 2017US$)
    # pop - Population (in millions)
    # ccon - Real consumption of households and government, at current PPPs (in mil. 2017US$)
    # cda - Real domestic absorption, (real consumption plus investment), at current PPPs (in mil. 2017US$)
# Existing Variables
variables = ['countrycode','year','rgdpo', 'pop','ccon','cda','cgdpo']
# New Variables
gdp_growth_avg_variable_name = 'gdp_growth_avg'
pop_growth_avg_variable_name = 'pop_growth_avg'
inv_rate_avg_variable_name = 'inv_rate_avg'
gdp_percapita_start_variable_name = 'gdp_percapita_start'
# Set of variables from regression
y_experiment_1 = [gdp_growth_avg_variable_name]
x_experiment_1 = [pop_growth_avg_variable_name,inv_rate_avg_variable_name,gdp_percapita_start_variable_name]

#--------
# Neural Network Parameters
epochs = 1000
neural_network_hidden_layers = [4,3]
learning_rate = 0.00001
tries_before_stopping = 5

#--------
# Cobb-Douglas Neural Network Parameters
sum_equal_one = 1

#--------
# Device - GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--------
# Pytorch Seed
random.seed(random_state)                          # Python
np.random.seed(random_state)                       # NumPy
torch.manual_seed(random_state)                    # PyTorch CPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)          # PyTorch GPU 
    torch.cuda.manual_seed_all(random_state)      # Multiple GPUs
torch.backends.cudnn.deterministic = True  # Deterministic
torch.backends.cudnn.benchmark = False     # Disable Non Deterministic

# =============================================================================
# 3 - Import PWT Data
# =============================================================================
wkdir = "/home/jeronymo/Documentos/neural_cge_network/src/empirical_application"
os.chdir(wkdir)

# 3.1 Import data
pwt_data = pd.read_excel('pwt_data.xlsx')
base_df = pwt_data[variables]

# 3.2 Set starting year as 1971
base_df = base_df[base_df['year']>=start_year]

# 3.3 Create Variables
# GDP Per capita
base_df['gdp_percapita'] = base_df['rgdpo'] / base_df['pop']
# Investment Rate
base_df['inv_rate'] = (base_df['cda']  - base_df['ccon']) / base_df['cgdpo']

# 3.4 Growth Rate
base_df = base_df.sort_values(by=['countrycode', 'year'])
# 3.4.1 Gdp_percapita
df_1971 = base_df[base_df['year'] == start_year][['countrycode', 'gdp_percapita']].rename(columns={'gdp_percapita': 'gdp_start'})
df_2003 = base_df[base_df['year'] == end_year][['countrycode', 'gdp_percapita']].rename(columns={'gdp_percapita': 'gdp_end'})
gdp_df_growth = pd.merge(df_1971, df_2003, on='countrycode')
gdp_df_growth[gdp_growth_avg_variable_name] = (np.log(gdp_df_growth['gdp_end']) - np.log(gdp_df_growth['gdp_start'])) / n_years * 100
# 3.4.2 Population
df_1971 = base_df[base_df['year'] == start_year][['countrycode', 'pop']].rename(columns={'pop': 'pop_start'})
df_2003 = base_df[base_df['year'] == end_year][['countrycode', 'pop']].rename(columns={'pop': 'pop_end'})
pop_df_growth = pd.merge(df_1971, df_2003, on='countrycode')
pop_df_growth[pop_growth_avg_variable_name] = (np.log(pop_df_growth['pop_end']) - np.log(pop_df_growth['pop_start'])) / n_years * 100
# 3.4.3 Investment Rate
inv_rate_avg = base_df.groupby('countrycode')['inv_rate'].mean().reset_index()
inv_rate_avg = inv_rate_avg.rename(columns={'inv_rate': inv_rate_avg_variable_name})
# 3.4.4 Initial Income
starting_income = base_df[base_df['year'] == start_year][['countrycode', 'gdp_percapita']]
starting_income = starting_income.rename(columns={'gdp_percapita': gdp_percapita_start_variable_name})

# 3.5 Create Dataframe
experiment_1_df = pd.DataFrame()
experiment_1_df['countrycode'] = df_1971['countrycode']
experiment_1_df = experiment_1_df.merge(gdp_df_growth[['countrycode', 'gdp_growth_avg']], on='countrycode', how='left')
experiment_1_df = experiment_1_df.merge(pop_df_growth[['countrycode', 'pop_growth_avg']], on='countrycode', how='left')
experiment_1_df = experiment_1_df.merge(inv_rate_avg[['countrycode', 'inv_rate_avg']], on='countrycode', how='left')
experiment_1_df = experiment_1_df.merge(starting_income[['countrycode', 'gdp_percapita_start']], on='countrycode', how='left')
experiment_1_df = experiment_1_df.dropna()

# =============================================================================
# 4 - Linear Regression
# =============================================================================
y = experiment_1_df[y_experiment_1]
x = experiment_1_df[x_experiment_1]
model = LinearRegression()
model.fit(x, y)
real_coef = model.coef_

# =============================================================================
# 5 - Neural Networks
# =============================================================================
