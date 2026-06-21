# -*- coding: utf-8 -*-
"""
app.py — Vitaliza Churn Intelligence
Artefato 2 — Módulo 2 Inteli MBA IA & Dados
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

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vitaliza · Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system — wealth-app dark ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fundo geral */
.stApp { background: #0F1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161F !important;
    border-right: 1px solid #1E2235;
}
[data-testid="stSidebar"] * { color: #8B92A5 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F0F2F5 !important; }
[data-testid="stSidebar"] .stRadio label { color: #8B92A5 !important; }
[data-testid="stSidebar"] .stRadio [data-checked="true"] + div { color: #00D4AA !important; }

/* Inputs */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: #1C1F2E !important;
    border: 1px solid #2A2E42 !important;
    color: #F0F2F5 !important;
    border-radius: 6px !important;
}
.stSlider [data-testid="stSlider"] { color: #00D4AA; }
.stSlider .rc-slider-track { background: #00D4AA !important; }
.stSlider .rc-slider-handle { border-color: #00D4AA !important; }

/* Botões */
.stButton button {
    background: #00D4AA !important;
    color: #0F1117 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    letter-spacing: 0.03em;
}
.stButton button:hover { background: #00BFAA !important; }

/* Métricas */
[data-testid="metric-container"] {
    background: #1C1F2E;
    border: 1px solid #2A2E42;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #8B92A5 !important; font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F0F2F5 !important; font-size: 28px !important; font-weight: 500 !important; }

/* Dataframes */
.stDataFrame { background: #1C1F2E !important; border: 1px solid #2A2E42 !important; border-radius: 10px !important; }
[data-testid="stDataFrameResizable"] { background: #1C1F2E !important; }

/* Texto geral */
h1, h2, h3 { color: #F0F2F5 !important; font-weight: 500 !important; }
p, li, span { color: #8B92A5; }
.stMarkdown p { color: #8B92A5; }

/* Expander */
[data-testid="stExpander"] {
    background: #1C1F2E !important;
    border: 1px solid #2A2E42 !important;
    border-radius: 10px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1C1F2E !important;
    border: 1px dashed #2A2E42 !important;
    border-radius: 10px !important;
}

/* Caption */
.stCaption { color: #555B70 !important; font-size: 11px !important; }

/* Divider */
hr { border-color: #1E2235 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #13161F; }
::-webkit-scrollbar-thumb { background: #2A2E42; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Componentes HTML customizados ─────────────────────────────────────────────
def risk_card(prob: float):
    """Card principal de risco — assinatura visual do app."""
    pct = prob * 100
    if prob >= 0.65:
        color, label, bg = "#FF4D6D", "RISCO ALTO", "#FF4D6D18"
    elif prob >= 0.35:
        color, label, bg = "#FFB800", "RISCO MODERADO", "#FFB80018"
    else:
        color, label, bg = "#00D4AA", "RISCO BAIXO", "#00D4AA18"

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {color}30; border-radius:14px;
                padding:28px 32px; margin-bottom:20px; position:relative; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; width:{pct}%; height:3px;
                  background:linear-gradient(90deg, {color}80, {color}); border-radius:3px 0 0 0;"></div>
      <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:8px;">
        <span style="font-size:56px; font-weight:600; color:{color}; line-height:1; letter-spacing:-2px;">{pct:.1f}<span style="font-size:28px; font-weight:400;">%</span></span>
        <span style="font-size:11px; font-weight:600; color:{color}; letter-spacing:0.12em; background:{color}22;
                     padding:4px 10px; border-radius:99px; border:1px solid {color}44;">{label}</span>
      </div>
      <p style="font-size:12px; color:#8B92A5; margin:0; letter-spacing:0.02em;">
        Probabilidade de cancelamento estimada pelo modelo
      </p>
    </div>
    """, unsafe_allow_html=True)


def shap_bar_chart(shap_dict: dict, top_n: int = 8):
    """Barras SHAP customizadas — estilo wealth dashboard."""
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    max_val = max(abs(v) for _, v in sorted_items) or 1

    rows = ""
    for feat, val in sorted_items:
        nome = FEATURE_NAMES_PT.get(feat, feat)
        pct = abs(val) / max_val * 100
        color = "#FF4D6D" if val > 0 else "#00D4AA"
        direction = "right" if val > 0 else "left"
        signal = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        rows += f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
          <span style="width:180px; font-size:12px; color:#8B92A5; text-align:right;
                       white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{nome}</span>
          <div style="flex:1; height:6px; background:#1E2235; border-radius:3px; position:relative;">
            <div style="position:absolute; {'right:50%' if val < 0 else 'left:50%'};
                        width:{pct/2:.1f}%; height:6px; background:{color};
                        border-radius:3px; opacity:0.85;"></div>
            <div style="position:absolute; left:50%; width:1px; height:10px;
                        top:-2px; background:#2A2E42;"></div>
          </div>
          <span style="width:52px; font-size:12px; color:{color}; font-weight:500;
                       font-family:monospace; text-align:right;">{signal}</span>
        </div>"""

    st.markdown(f"""
    <div style="background:#1C1F2E; border:1px solid #2A2E42; border-radius:12px; padding:20px 24px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:16px; padding-bottom:10px; border-bottom:1px solid #1E2235;">
        <span style="font-size:11px; color:#555B70; letter-spacing:0.08em; text-transform:uppercase;">Variável</span>
        <span style="font-size:11px; color:#555B70; letter-spacing:0.08em; text-transform:uppercase;">Impacto SHAP</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:12px; font-size:10px; color:#555B70;">
        <span style="margin-left:192px;">← reduz risco</span>
        <span>aumenta risco →</span>
      </div>
      {rows}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin:28px 0 16px;">
      <p style="font-size:11px; color:#00D4AA; letter-spacing:0.12em; text-transform:uppercase;
                font-weight:600; margin:0 0 4px;">{subtitle}</p>
      <h3 style="font-size:18px; color:#F0F2F5; font-weight:500; margin:0;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def kv_row(label: str, value: str, color: str = "#F0F2F5"):
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding:10px 0; border-bottom:1px solid #1E2235;">
      <span style="font-size:12px; color:#8B92A5;">{label}</span>
      <span style="font-size:13px; color:{color}; font-weight:500; font-family:monospace;">{value}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Recursos ──────────────────────────────────────────────────────────────────
FEATURES = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "Month_to_end_contract",
    "Lifetime", "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]


@st.cache_resource
def load_model():
    p = Path("churn_model.joblib")
    if not p.exists():
        st.error("Modelo não encontrado. Execute `python train.py` primeiro.")
        st.stop()
    return joblib.load(p)


@st.cache_data
def load_metrics():
    p = Path("metrics.json")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


pipeline = load_model()
metrics  = load_metrics()
model    = pipeline.named_steps["model"]
scaler   = pipeline.named_steps["scaler"]


def compute_shap(client_df: pd.DataFrame) -> np.ndarray:
    scaled = pd.DataFrame(scaler.transform(client_df), columns=FEATURES)
    explainer   = shap.TreeExplainer(model)
    sv = explainer.shap_values(scaled)
    sv = sv[1] if isinstance(sv, list) else sv[:, :, 1]
    return sv[0]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 24px;">
      <p style="font-size:20px; font-weight:600; color:#F0F2F5; margin:0; letter-spacing:-0.5px;">◈ Vitaliza</p>
      <p style="font-size:11px; color:#555B70; margin:2px 0 0; letter-spacing:0.08em;">CHURN INTELLIGENCE</p>
    </div>
    """, unsafe_allow_html=True)

    tab_mode = st.radio("", ["Análise individual", "Análise em lote", "Métricas do modelo"],
                        label_visibility="collapsed")

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    api_key = st.text_input(
        "Chave OpenRouter",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
        help="Necessária para gerar análises via LLM",
        placeholder="sk-or-v1-...",
    )

    st.markdown("""
    <div style="margin-top:auto; padding-top:40px;">
      <p style="font-size:10px; color:#2A2E42; letter-spacing:0.06em;">
        INTELI MBA · ARTEFATO 2<br>Módulo 2 — IA & Dados
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODO 1 — ANÁLISE INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
if tab_mode == "Análise individual":

    section_header("Análise de cliente", "PREDIÇÃO DE CHURN")

    with st.form("client_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p style="font-size:11px; color:#555B70; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;">Perfil</p>', unsafe_allow_html=True)
            gender      = st.selectbox("Gênero", [0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
            age         = st.slider("Idade", 18, 65, 30)
            near_loc    = st.selectbox("Mora perto da academia?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            partner     = st.selectbox("Tem parceiro(a)?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            promo       = st.selectbox("Indicado por amigo?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            phone       = st.selectbox("Telefone cadastrado?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

        with col2:
            st.markdown('<p style="font-size:11px; color:#555B70; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;">Contrato</p>', unsafe_allow_html=True)
            contract    = st.selectbox("Período do contrato", [1, 6, 12], format_func=lambda x: f"{x} mês" if x == 1 else f"{x} meses")
            months_left = st.slider("Meses até fim do contrato", 0.0, 12.0, 3.0, 0.5)
            lifetime    = st.slider("Tempo de plataforma (meses)", 0, 36, 4)
            extra_spend = st.number_input("Gasto adicional médio (R$)", 0.0, 500.0, 80.0, 10.0)

        with col3:
            st.markdown('<p style="font-size:11px; color:#555B70; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;">Engajamento</p>', unsafe_allow_html=True)
            group_visits = st.selectbox("Aulas em grupo?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            freq_total   = st.slider("Freq. média total (aulas/sem)", 0.0, 7.0, 2.0, 0.5)
            freq_current = st.slider("Freq. mês atual (aulas/sem)", 0.0, 7.0, 1.0, 0.5)

        submitted = st.form_submit_button("Analisar cliente", use_container_width=True, type="primary")

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

        client_df  = pd.DataFrame([client_data])[FEATURES]
        churn_prob = pipeline.predict_proba(client_df)[0][1]
        sv         = compute_shap(client_df)
        shap_dict  = dict(zip(FEATURES, sv.tolist()))

        st.markdown("<hr style='margin:24px 0 20px;'>", unsafe_allow_html=True)

        # Card de risco principal
        risk_card(churn_prob)

        # Métricas secundárias
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{churn_prob:.3f}")
        c2.metric("Lifetime", f"{lifetime} meses")
        c3.metric("Freq. atual", f"{freq_current:.1f}/sem")
        c4.metric("Contrato", f"{contract} mês" if contract == 1 else f"{contract} meses")

        # SHAP
        section_header("Fatores de risco", "EXPLICABILIDADE · SHAP VALUES")
        shap_bar_chart(shap_dict)

        # LLM
        section_header("Análise prescritiva", "IA · RECOMENDAÇÃO")
        if not api_key:
            st.markdown("""
            <div style="background:#1C1F2E; border:1px solid #2A2E42; border-radius:10px;
                        padding:20px 24px; color:#555B70; font-size:13px;">
              Configure sua chave OpenRouter na barra lateral para gerar a análise.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Gerando análise..."):
                try:
                    explanation = explain(
                        churn_prob=churn_prob,
                        shap_values=shap_dict,
                        client_data=client_data,
                        api_key=api_key,
                    )
                    st.markdown(f"""
                    <div style="background:#1C1F2E; border:1px solid #2A2E42; border-radius:12px;
                                padding:24px 28px; line-height:1.8; color:#C8CDD8; font-size:14px;">
                        {explanation.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erro ao chamar LLM: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 2 — ANÁLISE EM LOTE
# ─────────────────────────────────────────────────────────────────────────────
elif tab_mode == "Análise em lote":

    section_header("Análise em lote", "PREDIÇÃO · BASE COMPLETA")

    uploaded = st.file_uploader("Arraste o CSV de clientes aqui", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df_feat = df.drop("Churn", axis=1) if "Churn" in df.columns else df.copy()

        missing = [c for c in FEATURES if c not in df_feat.columns]
        if missing:
            st.error(f"Colunas ausentes: {missing}")
            st.stop()

        probs = pipeline.predict_proba(df_feat[FEATURES])[:, 1]
        df["Probabilidade de churn"] = probs
        df["Risco"] = pd.cut(probs, bins=[0, 0.35, 0.65, 1.0],
                             labels=["Baixo", "Moderado", "Alto"])

        # KPIs
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de clientes", len(df))
        c2.metric("Em risco alto", int((probs >= 0.65).sum()),
                  delta=f"{(probs >= 0.65).mean():.1%} da base",
                  delta_color="inverse")
        c3.metric("Risco moderado", int(((probs >= 0.35) & (probs < 0.65)).sum()))
        c4.metric("Score médio", f"{probs.mean():.3f}")

        section_header("Top 20 — maior risco de churn", "RANKING")

        top20 = (df.nlargest(20, "Probabilidade de churn")
                   [["Probabilidade de churn", "Risco"] + FEATURES]
                   .reset_index(drop=True))
        top20.insert(0, "#", range(1, 21))
        top20["Probabilidade de churn"] = top20["Probabilidade de churn"].apply(lambda x: f"{x:.1%}")
        st.dataframe(top20, use_container_width=True, hide_index=True)

        st.download_button(
            "Exportar CSV com scores",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="vitaliza_churn_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if api_key:
            section_header("Análises LLM — top 3 em risco", "IA · PRESCRIÇÕES")
            top3_idx = df["Probabilidade de churn"].nlargest(3).index if "Probabilidade de churn" in df.columns else []
            for i, idx in enumerate(df.nlargest(3, "Probabilidade de churn").index, 1):
                row      = df_feat.loc[idx]
                prob_val = float(pipeline.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
                sv       = compute_shap(pd.DataFrame([row])[FEATURES])
                sd       = dict(zip(FEATURES, sv.tolist()))

                with st.expander(f"Cliente #{i} — {prob_val:.1%} de risco", expanded=(i == 1)):
                    risk_card(prob_val)
                    shap_bar_chart(sd, top_n=6)
                    if api_key:
                        with st.spinner("Gerando análise..."):
                            try:
                                exp = explain(prob_val, sd, row.to_dict(), api_key=api_key)
                                st.markdown(f"""
                                <div style="background:#13161F; border:1px solid #1E2235;
                                            border-radius:10px; padding:20px 24px;
                                            color:#C8CDD8; font-size:13px; line-height:1.8; margin-top:16px;">
                                    {exp.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MODO 3 — MÉTRICAS DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
elif tab_mode == "Métricas do modelo":

    section_header("Validação do modelo", "PERFORMANCE · RANDOM FOREST")

    if not metrics:
        st.error("metrics.json não encontrado. Execute train.py primeiro.")
        st.stop()

    cm = metrics["confusion_matrix"]

    # KPIs principais
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")
    c2.metric("F1-Score",  f"{metrics['f1']:.3f}")
    c3.metric("Precisão",  f"{metrics['precision']:.3f}")
    c4.metric("Recall",    f"{metrics['recall']:.3f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Acurácia",       f"{metrics['accuracy']:.3f}")
    c6.metric("Especificidade", f"{metrics['specificity']:.3f}")
    c7.metric("FPR",            f"{metrics['fpr']:.3f}")
    c8.metric("FNR",            f"{metrics['fnr']:.3f}")

    st.caption(f"Cross-validation ROC-AUC: {metrics.get('cv_roc_auc_mean', 0):.3f} ± {metrics.get('cv_roc_auc_std', 0):.3f} · sem overfit confirmado")

    # Matriz de confusão
    section_header("Matriz de confusão", "CONJUNTO DE TESTE · 800 CLIENTES")

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; max-width:500px;">
      <div style="background:#00D4AA18; border:1px solid #00D4AA33; border-radius:10px; padding:20px; text-align:center;">
        <p style="font-size:36px; font-weight:600; color:#00D4AA; margin:0; line-height:1;">{cm['tn']}</p>
        <p style="font-size:11px; color:#00D4AA; margin:6px 0 0; letter-spacing:0.08em;">VERDADEIROS NEGATIVOS</p>
        <p style="font-size:11px; color:#555B70; margin:4px 0 0;">Não churn · correto</p>
      </div>
      <div style="background:#FF4D6D18; border:1px solid #FF4D6D33; border-radius:10px; padding:20px; text-align:center;">
        <p style="font-size:36px; font-weight:600; color:#FF4D6D; margin:0; line-height:1;">{cm['fp']}</p>
        <p style="font-size:11px; color:#FF4D6D; margin:6px 0 0; letter-spacing:0.08em;">FALSOS POSITIVOS</p>
        <p style="font-size:11px; color:#555B70; margin:4px 0 0;">Falso alarme</p>
      </div>
      <div style="background:#FF4D6D18; border:1px solid #FF4D6D33; border-radius:10px; padding:20px; text-align:center;">
        <p style="font-size:36px; font-weight:600; color:#FF4D6D; margin:0; line-height:1;">{cm['fn']}</p>
        <p style="font-size:11px; color:#FF4D6D; margin:6px 0 0; letter-spacing:0.08em;">FALSOS NEGATIVOS</p>
        <p style="font-size:11px; color:#555B70; margin:4px 0 0;">Churn não detectado ⚠</p>
      </div>
      <div style="background:#00D4AA18; border:1px solid #00D4AA33; border-radius:10px; padding:20px; text-align:center;">
        <p style="font-size:36px; font-weight:600; color:#00D4AA; margin:0; line-height:1;">{cm['tp']}</p>
        <p style="font-size:11px; color:#00D4AA; margin:6px 0 0; letter-spacing:0.08em;">VERDADEIROS POSITIVOS</p>
        <p style="font-size:11px; color:#555B70; margin:4px 0 0;">Churn detectado ✓</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP
    section_header("Importância das variáveis", "SHAP · FEATURE IMPORTANCE")

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
        ]).sort_values("SHAP (média abs.)", ascending=False).reset_index(drop=True)

        st.dataframe(shap_df, use_container_width=True, hide_index=True)
