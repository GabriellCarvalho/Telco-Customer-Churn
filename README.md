# Telco Customer Churn

Previsão de cancelamento (churn) de clientes de telecom com Machine Learning.

## 📊 O que tem no notebook?

### 1. **Carregamento do Dataset**
- Dataset do Kaggle: 7.043 clientes × 21 colunas
- Alvo: `Churn` (Yes/No)
- Sem valores ausentes (exceto 11 em TotalCharges)

### 2. **Análise Exploratória (EDA)**
- Distribuição de variáveis numéricas (tenure, MonthlyCharges, TotalCharges)
- Taxa de churn por categoria (Contract, InternetService, TechSupport, etc.)
- Gráficos de correlação entre features e churn
- Heatmaps: Churn × Contract × PaymentMethod

**Insights principais:**
- Month-to-month: 42% churn
- Electronic check: 45% churn
- 2-year contract: <3% churn

### 3. **Preparação dos Dados**
- Conversão: `Churn` (Yes/No → 1/0)
- `TotalCharges` (string → float)
- Divisão: 80% treino / 20% teste
- Pipeline de pré-processamento:
  - Numéricas: Imputation (median) + StandardScaler
  - Categóricas: Imputation (moda) + OneHotEncoder
- Output: 45 features após transformação

### 4. **Feature Importance**
- Gradient Boosting para calcular importância
- Top drivers: Contract (38.9%), tenure (13.9%), InternetService (8.7%)
- Agregação por variável original

### 5. **Machine Learning**
Treinou 4 modelos com GridSearchCV (3-fold CV, métrica: AUC):

| Modelo | AUC CV | Best Params |
|--------|--------|------------|
| **Gradient Boosting** | 0.8487 | lr=0.05, n_est=200, depth=2 |
| Random Forest | 0.8462 | depth=10, samples_leaf=5, n_est=500 |
| Logistic Regression | 0.8459 | C=10 |
| SVM | 0.8350 | kernel=linear, C=2 |

### 6. **Avaliação no Test Set**
- Métricas por modelo: AUC, Precision, Recall, Gini
- Gradient Boosting melhor: AUC 0.8457
- Decile plots: performance por faixa de score


## 📈 Resumo dos Resultados

- **Melhor modelo**: Gradient Boosting (AUC: 0.8457)
- **Top 3 features**: Contract, tenure, InternetService
- **Recomendações**: contratos longos, pagamento automático, add-ons de suporte

---

**Dataset**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
