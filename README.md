# Estimativa Neural via Modelo de Equilíbrio Geral Computável

Este projeto explora uma abordagem inovadora para estimar os parâmetros de uma rede neural com base em um modelo de Equilíbrio Geral Computável (CGE). A ideia central é simular a produção em cadeias setoriais (z1 → z2 → y), onde os insumos são processados por "setores" análogos a neurônios, e o objetivo final é prever um produto com base em parâmetros lineares encadeados. Essa estrutura permite estimar os pesos da rede diretamente via regressões lineares compostas, sem o uso tradicional de backpropagation. Em seguida, a estimativa é comparada com uma rede neural treinada normalmente com PyTorch (ou scikit-learn, em ambientes mais leves).

## Estrutura do Projeto

- **Geração de dados sintéticos**: simula relações entre insumo e produto por meio de equações lineares aninhadas.
- **Modelo CGE-like**: estima os parâmetros da rede com base em um modelo econométrico inspirado na estrutura CGE.
- **Rede Neural Tradicional**: utiliza PyTorch (ou `MLPRegressor`) para ajustar os pesos via backpropagation.
- **Avaliação**: as previsões de ambos os modelos são comparadas com métricas de acurácia, incluindo RMSE e R².

## Requisitos

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- (Opcional) PyTorch ou scikit-learn

## Resultados Esperados

- Curvas de previsão comparando a abordagem baseada em CGE com a rede neural tradicional.
- Métricas de desempenho (R² e RMSE) para avaliação da acurácia de ambas as abordagens.

## Contribuição

Esta metodologia busca aproximar conceitos de economia computacional e aprendizado de máquina. Ela pode ser útil para pesquisadores que desejam interpretar redes neurais como sistemas de produção encadeados, onde cada camada tem um significado econômico claro.
