"""
app.py — Interface web Vitaliza Churn Intelligence
Artefato 2 — Módulo 2 Inteli MBA IA & Dados

Execução:
    streamlit run app.py

Requer:
    churn_model.joblib e metrics.json gerados pelo train.py
    ANTHROPIC_API_KEY configurado (env ou .streamlit/secrets.toml)
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from llm_explainer import FEATURE_NAMES_PT, explain

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vitaliza · Churn Intelligence",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-badge-high   { background:#FEE2E2; color:#991B1B; padding:6px 16px; border-radius:20px; font-weight:600; }
    .risk-badge-medium { background:#FEF3C7; color:#92400E; padding:6px 16px; border-radius:20px; font-weight:600; }
    .risk-badge-low    { background:#D1FAE5; color:#065F46; padding:6px 16px; border-radius:20px; font-weight:600; }
    .metric-box { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; text-align:center; }
    .section-header { font-size:1.1rem; font-weight:600; color:#1E293B; margin:1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Carregamento de recursos ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("churn_model.joblib")
    if not model_path.exists():
        st.error("Modelo não encontrado. Execute `python train.py` primeiro.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_metrics():
    metrics_path = Path("metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)

pipeline = load_model()
metrics  = load_metrics()
model    = pipeline.named_steps["model"]
scaler   = pipeline.named_steps["scaler"]

FEATURES = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "Month_to_end_contract",
    "Lifetime", "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]


def compute_shap_for_client(client_df: pd.DataFrame) -> np.ndarray:
    """Retorna SHAP values (vetor 1D) para a classe churn=1."""
    scaled = pd.DataFrame(scaler.transform(client_df), columns=FEATURES)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
    return sv[0]  # primeiro (e único) cliente


def risk_label(prob: float) -> tuple[str, str]:
    if prob >= 0.65:
        return "ALTO", "risk-badge-high"
    if prob >= 0.35:
        return "MODERADO", "risk-badge-medium"
    return "BAIXO", "risk-badge-low"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dumbbell.png", width=60)
    st.title("Vitaliza")
    st.caption("Churn Intelligence · Artefato 2")

    st.divider()
    tab_mode = st.radio("Modo", ["Análise individual", "Análise em lote (CSV)", "Métricas do modelo"])

    st.divider()
    api_key = st.text_input(
        "Chave API Anthropic",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Necessária para gerar explicações via LLM",
    )

# ─────────────────────────────────────────────────────────────────────────────
# MODO 1 — ANÁLISE INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
if tab_mode == "Análise individual":
    st.title("🔍 Análise de cliente")
    st.caption("Preencha os dados e obtenha o score de churn + explicação personalizada.")

    with st.form("client_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Perfil**")
            gender       = st.selectbox("Gênero", [0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
            age          = st.slider("Idade", 18, 65, 30)
            near_loc     = st.selectbox("Mora perto da academia?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            partner      = st.selectbox("Tem parceiro(a)?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            promo        = st.selectbox("Indicado por amigo?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            phone        = st.selectbox("Telefone cadastrado?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

        with col2:
            st.markdown("**Contrato**")
            contract     = st.selectbox("Período do contrato", [1, 6, 12], format_func=lambda x: f"{x} mes(es)")
            months_left  = st.slider("Meses até fim do contrato", 0.0, 12.0, 3.0, 0.5)
            lifetime     = st.slider("Tempo de plataforma (meses)", 0, 36, 4)
            extra_spend  = st.number_input("Gasto adicional médio (R$)", 0.0, 500.0, 80.0, 10.0)

        with col3:
            st.markdown("**Engajamento**")
            group_visits = st.selectbox("Participa de aulas em grupo?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            freq_total   = st.slider("Freq. média total (aulas/semana)", 0.0, 7.0, 2.0, 0.5)
            freq_current = st.slider("Freq. no mês atual (aulas/semana)", 0.0, 7.0, 1.0, 0.5)

        submitted = st.form_submit_button("Analisar cliente ➜", use_container_width=True, type="primary")

    if submitted:
        client_data = {
            "gender": gender, "Near_Location": near_loc, "Partner": partner,
            "Promo_friends": promo, "Phone": phone, "Contract_period": contract,
            "Group_visits": group_visits, "Age": age,
            "Avg_additional_charges_total": extra_spend,
            "Month_to_end_contract": months_left, "Lifetime": lifetime,
            "Avg_class_frequency_total": freq_total,
            "Avg_class_frequency_current_month": freq_current,
        }

        client_df   = pd.DataFrame([client_data])[FEATURES]
        churn_prob  = pipeline.predict_proba(client_df)[0][1]
        sv          = compute_shap_for_client(client_df)
        shap_dict   = dict(zip(FEATURES, sv.tolist()))

        label, badge_class = risk_label(churn_prob)

        st.divider()
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Probabilidade de churn", f"{churn_prob:.1%}")
        with c2:
            st.markdown(f"Nível de risco<br><span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)
        with c3:
            st.progress(float(churn_prob), text=f"Score: {churn_prob:.3f}")

        st.markdown('<p class="section-header">🔬 Fatores mais relevantes (SHAP)</p>', unsafe_allow_html=True)

        shap_sorted = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        shap_df = pd.DataFrame([
            {
                "Variável": FEATURE_NAMES_PT.get(f, f),
                "Impacto SHAP": round(v, 4),
                "Direção": "↑ Aumenta risco" if v > 0 else "↓ Reduz risco",
            }
            for f, v in shap_sorted
        ])
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header">🤖 Explicação e prescrição via LLM</p>', unsafe_allow_html=True)

        if not api_key:
            st.warning("Configure sua chave API Anthropic na barra lateral para gerar a explicação.")
        else:
            with st.spinner("Gerando análise prescritiva..."):
                try:
                    explanation = explain(
                        churn_prob=churn_prob,
                        shap_values=shap_dict,
                        client_data=client_data,
                        api_key=api_key,
                    )
                    st.markdown(explanation)
                except Exception as e:
                    st.error(f"Erro ao chamar LLM: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 2 — ANÁLISE EM LOTE
# ─────────────────────────────────────────────────────────────────────────────
elif tab_mode == "Análise em lote (CSV)":
    st.title("📂 Análise em lote")
    st.caption("Faça upload do CSV de clientes e veja os scores de churn de toda a base.")

    uploaded = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "Churn" in df.columns:
            df_feat = df.drop("Churn", axis=1)
        else:
            df_feat = df.copy()

        missing = [c for c in FEATURES if c not in df_feat.columns]
        if missing:
            st.error(f"Colunas ausentes no CSV: {missing}")
            st.stop()

        probs = pipeline.predict_proba(df_feat[FEATURES])[:, 1]
        df["churn_prob"] = probs
        df["risco"]      = pd.cut(probs, bins=[0, 0.35, 0.65, 1.0], labels=["Baixo", "Moderado", "Alto"])

        st.subheader("Distribuição de risco")
        risk_counts = df["risco"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Alto",     int(risk_counts.get("Alto", 0)))
        col2.metric("🟡 Moderado", int(risk_counts.get("Moderado", 0)))
        col3.metric("🟢 Baixo",    int(risk_counts.get("Baixo", 0)))

        st.subheader("Top 20 clientes com maior risco")
        top20 = df.nlargest(20, "churn_prob")[["churn_prob", "risco"] + FEATURES]
        top20.insert(0, "Rank", range(1, 21))
        st.dataframe(
            top20.style.background_gradient(subset=["churn_prob"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

        st.download_button(
            "⬇ Baixar CSV com scores",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="vitaliza_churn_scores.csv",
            mime="text/csv",
        )

        if api_key:
            st.subheader("Explicações LLM — Top 3 clientes em risco")
            top3_idx = df["churn_prob"].nlargest(3).index
            for i, idx in enumerate(top3_idx, 1):
                row = df_feat.loc[idx]
                row_df   = pd.DataFrame([row])[FEATURES]
                prob_val = float(df.loc[idx, "churn_prob"])
                sv       = compute_shap_for_client(row_df)
                sd       = dict(zip(FEATURES, sv.tolist()))

                with st.expander(f"Cliente #{i} — Risco {prob_val:.1%}", expanded=(i == 1)):
                    with st.spinner("Gerando análise..."):
                        try:
                            exp = explain(
                                churn_prob=prob_val,
                                shap_values=sd,
                                client_data=row.to_dict(),
                                api_key=api_key,
                            )
                            st.markdown(exp)
                        except Exception as e:
                            st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MODO 3 — MÉTRICAS DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
elif tab_mode == "Métricas do modelo":
    st.title("📊 Validação do modelo")

    if not metrics:
        st.error("Arquivo metrics.json não encontrado. Execute train.py primeiro.")
        st.stop()

    cm = metrics["confusion_matrix"]

    st.subheader("Matriz de confusão")
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Real: Não churn", "Real: Churn"],
        columns=["Previsto: Não churn", "Previsto: Churn"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Métricas de performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",    f"{metrics['roc_auc']:.3f}")
    col2.metric("F1-Score",   f"{metrics['f1']:.3f}")
    col3.metric("Precisão",   f"{metrics['precision']:.3f}")
    col4.metric("Recall",     f"{metrics['recall']:.3f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Acurácia",        f"{metrics['accuracy']:.3f}")
    col6.metric("Especificidade",  f"{metrics['specificity']:.3f}")
    col7.metric("FPR",             f"{metrics['fpr']:.3f}")
    col8.metric("FNR",             f"{metrics['fnr']:.3f}")

    st.caption(f"Cross-validation ROC-AUC: {metrics.get('cv_roc_auc_mean', 'N/A'):.3f} ± {metrics.get('cv_roc_auc_std', 'N/A'):.3f}")

    st.subheader("SHAP — importância média das variáveis")
    shap_data = metrics.get("shap_mean_abs", {})
    fi_data   = metrics.get("feature_importances", {})

    if shap_data:
        shap_df = pd.DataFrame([
            {
                "Variável": FEATURE_NAMES_PT.get(f, f),
                "SHAP (média abs.)": round(v, 4),
                "Feature Importance": round(fi_data.get(f, 0), 4),
            }
            for f, v in shap_data.items()
        ]).sort_values("SHAP (média abs.)", ascending=False)

        st.dataframe(
            shap_df.style.background_gradient(subset=["SHAP (média abs.)"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )
