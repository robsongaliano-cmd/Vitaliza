# Vitaliza · Churn Intelligence
**Artefato 2 — Módulo 2 | Inteli MBA IA & Dados**

Sistema de predição e explicabilidade de churn com integração LLM para análise prescritiva.

---

## Estrutura do projeto

```
artefato2_vitaliza/
├── train.py              # Pipeline de treinamento
├── app.py                # Interface web (Streamlit)
├── llm_explainer.py      # Módulo de explicabilidade via LLM
├── churn_model.joblib    # Modelo serializado (gerado pelo train.py)
├── metrics.json          # Métricas de validação (gerado pelo train.py)
├── requirements.txt      # Dependências
└── README.md
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Treinar o modelo

Coloque o arquivo `gym_churn_us_1.csv` na pasta do projeto e execute:

```bash
python train.py --data gym_churn_us_1.csv
```

Isso gera `churn_model.joblib` e `metrics.json`.

### 2. Configurar a chave da API

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Ou crie o arquivo `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "sk-or-..."
```

### 3. Rodar a interface web

```bash
streamlit run app.py
```

Acesse em: http://localhost:8501

---

## Funcionalidades

| Modo | Descrição |
|------|-----------|
| Análise individual | Preenche dados de um cliente e obtém score + SHAP + explicação LLM |
| Análise em lote | Upload do CSV → scores para toda a base + top 3 explicações |
| Métricas do modelo | Matriz de confusão, ROC-AUC, SHAP comparativo |

---

## Deploy no Streamlit Cloud (link público)

1. Faça upload do projeto para um repositório GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte o repositório e selecione `app.py` como arquivo principal
4. Em **Secrets**, adicione: `OPENROUTER_API_KEY = "sua-chave"`
5. Clique em **Deploy** — o link público é gerado em minutos

---

## Modelo

- Algoritmo: Random Forest (200 árvores, class_weight=balanced)
- Features: 13 variáveis comportamentais e demográficas
- Validação: ROC-AUC = 0.968 | F1 = 0.837 | Sem overfit (gap CV/teste = 0.032)
- Explicabilidade: SHAP TreeExplainer + análise prescritiva via Claude (Anthropic)

---

## Checklist Artefato 2

- [x] Modelo de churn validado (sem overfit, sem vazamento, métricas adequadas)
- [x] Explicabilidade: SHAP values + feature importances + explicação em linguagem natural
- [x] Serviço web servindo inferência
- [x] Código dividido em pipeline de treinamento e pipeline de inferência
- [x] Inferência usando arquivo serializado (joblib)
- [ ] Demo: compartilhar link (Streamlit Cloud) ou vídeo
