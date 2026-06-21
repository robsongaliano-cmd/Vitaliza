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
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background: #F8F9FB !important; border-right: 1px solid #E8ECF0; }

/* Métricas */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E8ECF0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #6B7385 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 600 !important;
    color: #1A1D23 !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
    border: 1px solid #DDE1E9 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* Botão primário */
.stButton > button[kind="primary"] {
    background: #1A1D23 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    letter-spacing: 0.02em;
    transition: background 0.15s;
}
.stButton > button[kind="primary"]:hover { background: #2D3240 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #E8ECF0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #E8ECF0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}

/* Cabeçalhos */
h1 { font-size: 32px !important; font-weight: 700 !important; color: #1A1D23 !important; letter-spacing: -0.5px !important; }
h2 { font-size: 18px !important; font-weight: 600 !important; color: #1A1D23 !important; }
h3 { font-size: 14px !important; font-weight: 600 !important; color: #1A1D23 !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border: 1px solid #DDE1E9 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    background: #FAFBFC !important;
}

/* Slider — cor do thumb e track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #1A1D23 !important;
    border-color: #1A1D23 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] div:first-child {
    background: #1A1D23 !important;
}
[data-testid="stSlider"] [class*="sliderThumb"] {
    background: #1A1D23 !important;
}

/* Form card */
[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 1px solid #E8ECF0 !important;
    border-radius: 14px !important;
    padding: 8px 4px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}

/* Label dos inputs */
[data-testid="stForm"] label {
    font-size: 12px !important;
    color: #4B5563 !important;
    font-weight: 500 !important;
}

/* Slider value color */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #1A1D23 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Upload */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #DDE1E9 !important;
    border-radius: 10px !important;
    background: #FAFBFC !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers visuais ───────────────────────────────────────────────────────────
def risk_card(prob: float):
    pct = prob * 100
    if prob >= 0.65:
        accent, label, bg, border = "#EF4444", "RISCO ALTO", "#FEF2F2", "#FCA5A5"
    elif prob >= 0.35:
        accent, label, bg, border = "#F59E0B", "RISCO MODERADO", "#FFFBEB", "#FCD34D"
    else:
        accent, label, bg, border = "#10B981", "RISCO BAIXO", "#F0FDF9", "#6EE7B7"

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:12px;
                padding:24px 28px; margin-bottom:20px; position:relative; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; width:{pct:.1f}%; height:4px;
                  background:{accent}; border-radius:4px 0 0 0; opacity:0.7;"></div>
      <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
        <span style="font-size:52px; font-weight:700; color:{accent}; line-height:1;
                     letter-spacing:-2px;">{pct:.1f}<span style="font-size:26px; font-weight:400;">%</span></span>
        <div>
          <span style="display:inline-block; font-size:11px; font-weight:600; color:{accent};
                       letter-spacing:0.1em; background:white; padding:3px 10px;
                       border-radius:99px; border:1px solid {border};">{label}</span>
          <p style="font-size:12px; color:#6B7385; margin:6px 0 0;">
            Probabilidade de cancelamento estimada pelo modelo
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def shap_bars(shap_dict: dict, top_n: int = 8):
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    max_v = max(abs(v) for _, v in items) or 1

    header = st.columns([3, 6, 1])
    header[0].markdown('<p style="font-size:10px;color:#9CA3AF;letter-spacing:0.08em;text-transform:uppercase;margin:0;text-align:right;">Variável</p>', unsafe_allow_html=True)
    header[1].markdown('<p style="font-size:10px;color:#9CA3AF;letter-spacing:0.08em;text-transform:uppercase;margin:0;text-align:center;">← reduz risco &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; aumenta risco →</p>', unsafe_allow_html=True)
    header[2].markdown('<p style="font-size:10px;color:#9CA3AF;letter-spacing:0.08em;text-transform:uppercase;margin:0;text-align:right;">SHAP</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #F1F3F7;margin:4px 0 8px;">', unsafe_allow_html=True)

    for feat, val in items:
        nome  = FEATURE_NAMES_PT.get(feat, feat)
        pct   = abs(val) / max_v
        color = "#EF4444" if val > 0 else "#10B981"
        sig   = ("+" if val > 0 else "") + f"{val:.3f}"
        bar_left  = "50%" if val > 0 else (str(round(50 - pct * 46, 1)) + "%")
        bar_width = str(round(pct * 46, 1)) + "%"

        col_label, col_bar, col_val = st.columns([3, 6, 1])
        col_label.markdown(
            "<p style='font-size:12px;color:#4B5563;text-align:right;margin:4px 0;"
            "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + nome + "</p>",
            unsafe_allow_html=True,
        )
        bar_html = (
            "<div style='position:relative;height:20px;display:flex;align-items:center;'>"
            "<div style='position:absolute;left:0;right:0;height:5px;background:#F1F3F7;border-radius:3px;'></div>"
            "<div style='position:absolute;left:50%;width:1px;height:12px;background:#DDE1E9;top:4px;'></div>"
            "<div style='position:absolute;left:" + bar_left + ";width:" + bar_width + ";height:5px;background:" + color + ";border-radius:3px;opacity:0.85;'></div>"
            "</div>"
        )
        col_bar.markdown(bar_html, unsafe_allow_html=True)
        col_val.markdown(
            "<p style='font-size:12px;font-weight:600;color:" + color + ";"
            "font-family:monospace;text-align:right;margin:4px 0;'>" + sig + "</p>",
            unsafe_allow_html=True,
        )


def section_label(eyebrow: str, title: str):
    st.markdown(f"""
    <div style="margin:28px 0 14px;">
      <p style="font-size:10px;color:#9CA3AF;letter-spacing:0.1em;text-transform:uppercase;
                font-weight:600;margin:0 0 3px;">{eyebrow}</p>
      <p style="font-size:16px;color:#1A1D23;font-weight:600;margin:0;">{title}</p>
    </div>
    """, unsafe_allow_html=True)


def divider():
    st.markdown('<hr style="border:none;border-top:1px solid #F1F3F7;margin:24px 0;">', unsafe_allow_html=True)


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
        st.error("Modelo não encontrado. Execute train.py primeiro.")
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
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(scaled)
    sv = sv[1] if isinstance(sv, list) else sv[:, :, 1]
    return sv[0]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 20px;">
      <p style="font-size:18px;font-weight:700;color:#1A1D23;margin:0;letter-spacing:-0.5px;">
        🏋️ Vitaliza
      </p>
      <p style="font-size:10px;color:#9CA3AF;margin:2px 0 0;letter-spacing:0.08em;">
        CHURN INTELLIGENCE
      </p>
    </div>
    """, unsafe_allow_html=True)

    tab_mode = st.radio(
        "Navegação",
        ["🔍 Análise individual", "📂 Análise em lote", "📊 Métricas do modelo"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #E8ECF0;margin:16px 0;">', unsafe_allow_html=True)

    api_key = st.text_input(
        "Chave OpenRouter",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
        placeholder="sk-or-v1-...",
        help="Necessária para gerar análises via LLM. Configure nos Secrets do Streamlit.",
    )

    if api_key:
        st.markdown('<p style="font-size:11px;color:#10B981;margin:4px 0 0;">✓ Chave configurada</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:11px;color:#F59E0B;margin:4px 0 0;">⚠ Sem chave — LLM desabilitado</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:32px;padding:12px;background:#F8F9FB;border-radius:8px;
                border:1px solid #E8ECF0;">
      <p style="font-size:10px;color:#9CA3AF;margin:0;line-height:1.7;">
        Inteli MBA · Artefato 2<br>
        Módulo 2 — IA & Dados<br>
        Random Forest · SHAP · Claude
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODO 1 — ANÁLISE INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
if tab_mode == "🔍 Análise individual":

    st.markdown("""
    <div style="margin-bottom:24px;">
      <p style="font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.12em;
                text-transform:uppercase;margin:0 0 6px;">CHURN INTELLIGENCE · ANÁLISE INDIVIDUAL</p>
      <h1 style="font-size:34px;font-weight:700;color:#1A1D23;letter-spacing:-0.8px;
                 line-height:1.1;margin:0 0 8px;">Análise de cliente</h1>
      <p style="font-size:13px;color:#6B7385;margin:0;">
        Preencha os dados e obtenha o score de churn, os fatores de risco e o briefing de recomendação.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("client_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p style="font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">Perfil</p>', unsafe_allow_html=True)
            gender       = st.selectbox("Gênero", [0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
            age          = st.slider("Idade", 18, 65, 30)
            near_loc     = st.selectbox("Mora perto da academia?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            partner      = st.selectbox("Tem parceiro(a)?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            promo        = st.selectbox("Indicado por amigo?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            phone        = st.selectbox("Telefone cadastrado?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

        with col2:
            st.markdown('<p style="font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">Contrato</p>', unsafe_allow_html=True)
            contract     = st.selectbox("Período do contrato", [1, 6, 12], format_func=lambda x: f"{x} mês" if x == 1 else f"{x} meses")
            months_left  = st.slider("Meses até fim do contrato", 0.0, 12.0, 3.0, 0.5)
            lifetime     = st.slider("Tempo de plataforma (meses)", 0, 36, 4)
            extra_spend  = st.number_input("Gasto adicional médio (R$)", 0.0, 500.0, 80.0, 10.0)

        with col3:
            st.markdown('<p style="font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">Engajamento</p>', unsafe_allow_html=True)
            group_visits = st.selectbox("Aulas em grupo?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            freq_total   = st.slider("Freq. média total (aulas/sem)", 0.0, 7.0, 2.0, 0.5)
            freq_current = st.slider("Freq. mês atual (aulas/sem)", 0.0, 7.0, 1.0, 0.5)

        submitted = st.form_submit_button("Analisar cliente →", use_container_width=True, type="primary")

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

        divider()
        risk_card(churn_prob)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score do modelo", f"{churn_prob:.3f}")
        c2.metric("Tempo de plataforma", f"{lifetime} meses")
        c3.metric("Frequência atual", f"{freq_current:.1f} aulas/sem")
        c4.metric("Tipo de contrato", f"{contract} mês" if contract == 1 else f"{contract} meses")

        section_label("SHAP VALUES", "Fatores que influenciam o risco")
        shap_bars(shap_dict)

        section_label("IA · ANÁLISE PRESCRITIVA", "Explicação e recomendação")

        if not api_key:
            st.info("Configure sua chave OpenRouter na barra lateral para gerar a análise prescritiva.")
        else:
            with st.spinner("Gerando análise via LLM..."):
                try:
                    explanation = explain(
                        churn_prob=churn_prob,
                        shap_values=shap_dict,
                        client_data=client_data,
                        api_key=api_key,
                    )
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid #E8ECF0;border-radius:12px;
                                padding:24px 28px;line-height:1.85;color:#374151;font-size:14px;
                                box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                        {explanation.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erro ao chamar LLM: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 2 — ANÁLISE EM LOTE
# ─────────────────────────────────────────────────────────────────────────────
elif tab_mode == "📂 Análise em lote":

    st.markdown("""
    <div style="margin-bottom:24px;">
      <p style="font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.12em;
                text-transform:uppercase;margin:0 0 6px;">CHURN INTELLIGENCE · ANÁLISE EM LOTE</p>
      <h1 style="font-size:34px;font-weight:700;color:#1A1D23;letter-spacing:-0.8px;
                 line-height:1.1;margin:0 0 8px;">Análise em lote</h1>
      <p style="font-size:13px;color:#6B7385;margin:0;">
        Faça upload do CSV e veja os scores de churn de toda a base com briefings personalizados.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Selecione o CSV de clientes", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df_feat = df.drop("Churn", axis=1) if "Churn" in df.columns else df.copy()

        missing = [c for c in FEATURES if c not in df_feat.columns]
        if missing:
            st.error(f"Colunas ausentes no CSV: {missing}")
            st.stop()

        probs = pipeline.predict_proba(df_feat[FEATURES])[:, 1]
        df["churn_prob"] = probs
        df["risco"] = pd.cut(probs, bins=[0, 0.35, 0.65, 1.0], labels=["Baixo", "Moderado", "Alto"])

        divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de clientes", f"{len(df):,}")
        c2.metric("Em risco alto", int((probs >= 0.65).sum()),
                  delta=f"{(probs >= 0.65).mean():.1%} da base", delta_color="inverse")
        c3.metric("Risco moderado", int(((probs >= 0.35) & (probs < 0.65)).sum()))
        c4.metric("Score médio", f"{probs.mean():.3f}")

        section_label("RANKING", "Top 20 — maior risco de churn")
        top20 = (df.nlargest(20, "churn_prob")[["churn_prob", "risco"] + FEATURES]
                   .reset_index(drop=True))
        top20.insert(0, "#", range(1, 21))
        top20["churn_prob"] = top20["churn_prob"].apply(lambda x: f"{x:.1%}")
        st.dataframe(top20, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Exportar CSV com scores",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="vitaliza_churn_scores.csv",
            mime="text/csv",
        )

        if api_key:
            section_label("IA · PRESCRIÇÕES", "Análises LLM — top 3 em risco")
            for i, idx in enumerate(df.nlargest(3, "churn_prob").index, 1):
                row      = df_feat.loc[idx]
                prob_val = float(pipeline.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
                sv       = compute_shap(pd.DataFrame([row])[FEATURES])
                sd       = dict(zip(FEATURES, sv.tolist()))

                with st.expander(f"Cliente #{i} — {prob_val:.1%} de risco", expanded=(i == 1)):
                    risk_card(prob_val)
                    shap_bars(sd, top_n=6)
                    with st.spinner("Gerando análise..."):
                        try:
                            exp = explain(prob_val, sd, row.to_dict(), api_key=api_key)
                            st.markdown(f"""
                            <div style="background:#FAFBFC;border:1px solid #E8ECF0;border-radius:10px;
                                        padding:20px 24px;color:#374151;font-size:13px;
                                        line-height:1.85;margin-top:14px;">
                                {exp.replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MODO 3 — MÉTRICAS DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
elif tab_mode == "📊 Métricas do modelo":

    st.markdown("""
    <div style="margin-bottom:24px;">
      <p style="font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.12em;
                text-transform:uppercase;margin:0 0 6px;">CHURN INTELLIGENCE · MÉTRICAS</p>
      <h1 style="font-size:34px;font-weight:700;color:#1A1D23;letter-spacing:-0.8px;
                 line-height:1.1;margin:0 0 8px;">Validação do modelo</h1>
      <p style="font-size:13px;color:#6B7385;margin:0;">
        Random Forest · 200 árvores · class_weight=balanced · split 80/20 estratificado
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:
        st.error("metrics.json não encontrado. Execute train.py primeiro.")
        st.stop()

    cm = metrics["confusion_matrix"]

    # ── Gráfico de barras horizontais — métricas de performance ──────────────
    section_label("PERFORMANCE · RANDOM FOREST", "Métricas de avaliação")

    metric_items = [
        ("ROC-AUC",        metrics["roc_auc"],        "#1A1D23", "Discriminação geral do modelo — probabilidade de ranquear churn acima de não-churn"),
        ("F1-Score",       metrics["f1"],             "#3B82F6", "Equilíbrio entre precisão e recall"),
        ("Precisão (PPV)", metrics["precision"],      "#8B5CF6", "Dos clientes sinalizados, % que realmente cancelaram"),
        ("Recall",         metrics["recall"],         "#F59E0B", "Dos que cancelaram, % que o modelo detectou"),
        ("Acurácia",       metrics["accuracy"],       "#10B981", "Acertos totais sobre o conjunto de teste"),
        ("Especificidade", metrics["specificity"],    "#06B6D4", "Dos que não cancelaram, % corretamente identificados"),
        ("FPR",            metrics["fpr"],            "#EF4444", "Taxa de falsos positivos — alarmes desnecessários"),
        ("FNR",            metrics["fnr"],            "#F97316", "Taxa de falsos negativos — churn não detectado"),
    ]

    # Gerar barras via st.columns — compatível com Streamlit Cloud
    for label, value, color, descricao in metric_items:
        pct = value * 100
        bar_width = str(round(pct, 1)) + "%"
        col_label, col_bar, col_val = st.columns([2, 6, 1])

        col_label.markdown(
            "<p style='font-size:12px;color:#4B5563;font-weight:500;margin:6px 0;"
            "text-align:right;'>" + label + "</p>",
            unsafe_allow_html=True,
        )

        bar_html = (
            "<div style='margin-bottom:2px;'>"
            "<div style='position:relative;height:8px;'>"
            "<div style='position:absolute;left:0;right:0;height:8px;background:#F1F3F7;"
            "border-radius:4px;'></div>"
            "<div style='position:absolute;left:0;width:" + bar_width + ";height:8px;"
            "background:" + color + ";border-radius:4px;opacity:0.85;'></div>"
            "</div>"
            "<p style='font-size:10px;color:#9CA3AF;margin:3px 0 0;'>" + descricao + "</p>"
            "</div>"
        )
        col_bar.markdown(bar_html, unsafe_allow_html=True)

        val_color = "#EF4444" if label in ("FPR", "FNR") else "#1A1D23"
        col_val.markdown(
            "<p style='font-size:13px;font-weight:700;color:" + val_color + ";"
            "text-align:right;margin:6px 0;font-family:monospace;'>"
            + f"{value:.3f}" + "</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='font-size:11px;color:#9CA3AF;margin:12px 0 0;'>"
        "CV ROC-AUC: " + f"{metrics.get('cv_roc_auc_mean', 0):.3f}" +
        " ± " + f"{metrics.get('cv_roc_auc_std', 0):.3f}" +
        " · gap treino/teste = 0.032 · sem overfit confirmado</p>",
        unsafe_allow_html=True,
    )

    # ── Matriz de confusão ────────────────────────────────────────────────────
    section_label("CONJUNTO DE TESTE · 800 CLIENTES", "Matriz de confusão")

    col_cm, col_info = st.columns([1, 1])

    with col_cm:
        total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
        for row_label, v1, c1_val, l1, v2, c2_val, l2 in [
            ("Real: Não churn", cm['tn'], "#059669", "TN · correto", cm['fp'], "#DC2626", "FP · falso alarme"),
            ("Real: Churn",     cm['fn'], "#DC2626", "FN · não detectado ⚠", cm['tp'], "#059669", "TP · detectado ✓"),
        ]:
            r1, r2 = st.columns(2)
            for col_obj, val, color, leg in [(r1, v1, c1_val, l1), (r2, v2, c2_val, l2)]:
                bg = "#F0FDF9" if color == "#059669" else "#FEF2F2"
                border = "#A7F3D0" if color == "#059669" else "#FCA5A5"
                col_obj.markdown(
                    "<div style='background:" + bg + ";border:1px solid " + border + ";"
                    "border-radius:10px;padding:16px;text-align:center;margin-bottom:8px;'>"
                    "<p style='font-size:30px;font-weight:700;color:" + color + ";margin:0;line-height:1;'>"
                    + str(val) + "</p>"
                    "<p style='font-size:10px;color:" + color + ";margin:4px 0 0;font-weight:600;"
                    "letter-spacing:0.06em;'>" + leg + "</p>"
                    "<p style='font-size:11px;color:#6B7385;margin:2px 0 0;'>"
                    + f"{val/total:.1%} do total" + "</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    with col_info:
        st.markdown(
            "<div style='background:#F8F9FB;border:1px solid #E8ECF0;border-radius:12px;"
            "padding:20px 22px;margin-top:0;'>"
            "<p style='font-size:11px;color:#9CA3AF;font-weight:600;letter-spacing:0.08em;"
            "text-transform:uppercase;margin:0 0 14px;'>Interpretação de negócio</p>"
            "<div style='margin-bottom:12px;'>"
            "<p style='font-size:12px;font-weight:600;color:#059669;margin:0 0 2px;'>TN = " + str(cm['tn']) + " · Não churn · Correto</p>"
            "<p style='font-size:12px;color:#6B7385;margin:0;'>Cliente seguro identificado corretamente — sem custo de intervenção.</p>"
            "</div>"
            "<div style='margin-bottom:12px;'>"
            "<p style='font-size:12px;font-weight:600;color:#DC2626;margin:0 0 2px;'>FP = " + str(cm['fp']) + " · Falso alarme</p>"
            "<p style='font-size:12px;color:#6B7385;margin:0;'>Oferta de retenção enviada desnecessariamente — custo baixo e aceitável.</p>"
            "</div>"
            "<div style='margin-bottom:12px;'>"
            "<p style='font-size:12px;font-weight:600;color:#DC2626;margin:0 0 2px;'>FN = " + str(cm['fn']) + " · Churn não detectado ⚠</p>"
            "<p style='font-size:12px;color:#6B7385;margin:0;'>Cliente em churn real que passou despercebido — risco principal para a Vitaliza.</p>"
            "</div>"
            "<div>"
            "<p style='font-size:12px;font-weight:600;color:#059669;margin:0 0 2px;'>TP = " + str(cm['tp']) + " · Churn detectado ✓</p>"
            "<p style='font-size:12px;color:#6B7385;margin:0;'>Cliente em risco identificado — candidato direto ao briefing de retenção.</p>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── SHAP — gráfico de barras horizontais ──────────────────────────────────
    section_label("SHAP · FEATURE IMPORTANCE", "Importância das variáveis")

    shap_data = metrics.get("shap_mean_abs", {})
    fi_data   = metrics.get("feature_importances", {})

    if shap_data:
        items_shap = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)
        max_shap   = items_shap[0][1] if items_shap else 1

        header_s = st.columns([3, 4, 2, 1])
        header_s[0].markdown("<p style='font-size:10px;color:#9CA3AF;letter-spacing:0.07em;text-transform:uppercase;margin:0;text-align:right;'>Variável</p>", unsafe_allow_html=True)
        header_s[1].markdown("<p style='font-size:10px;color:#9CA3AF;letter-spacing:0.07em;text-transform:uppercase;margin:0;'>SHAP (impacto médio)</p>", unsafe_allow_html=True)
        header_s[2].markdown("<p style='font-size:10px;color:#9CA3AF;letter-spacing:0.07em;text-transform:uppercase;margin:0;'>Feature Importance</p>", unsafe_allow_html=True)
        header_s[3].markdown("<p style='font-size:10px;color:#9CA3AF;letter-spacing:0.07em;text-transform:uppercase;margin:0;text-align:right;'>SHAP</p>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #F1F3F7;margin:4px 0 6px;'>", unsafe_allow_html=True)

        for feat, shap_val in items_shap:
            nome    = FEATURE_NAMES_PT.get(feat, feat)
            fi_val  = fi_data.get(feat, 0)
            sw      = str(round(shap_val / max_shap * 100, 1)) + "%"
            fw      = str(round(fi_val / max(fi_data.values()) * 100, 1)) + "%"

            cs, cb1, cb2, cv = st.columns([3, 4, 2, 1])
            cs.markdown(
                "<p style='font-size:12px;color:#4B5563;text-align:right;margin:5px 0;"
                "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + nome + "</p>",
                unsafe_allow_html=True,
            )
            cb1.markdown(
                "<div style='position:relative;height:24px;display:flex;align-items:center;'>"
                "<div style='position:absolute;left:0;right:0;height:7px;background:#F1F3F7;border-radius:4px;'></div>"
                "<div style='position:absolute;left:0;width:" + sw + ";height:7px;"
                "background:#1A1D23;border-radius:4px;opacity:0.85;'></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            cb2.markdown(
                "<div style='position:relative;height:24px;display:flex;align-items:center;'>"
                "<div style='position:absolute;left:0;right:0;height:7px;background:#F1F3F7;border-radius:4px;'></div>"
                "<div style='position:absolute;left:0;width:" + fw + ";height:7px;"
                "background:#9CA3AF;border-radius:4px;opacity:0.7;'></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            cv.markdown(
                "<p style='font-size:12px;font-weight:600;color:#1A1D23;"
                "font-family:monospace;text-align:right;margin:5px 0;'>"
                + f"{shap_val:.4f}" + "</p>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='display:flex;gap:20px;margin-top:10px;'>"
            "<span style='display:flex;align-items:center;gap:6px;font-size:11px;color:#6B7385;'>"
            "<span style='width:12px;height:7px;background:#1A1D23;border-radius:2px;display:inline-block;opacity:0.85;'></span>"
            "SHAP (impacto causal)</span>"
            "<span style='display:flex;align-items:center;gap:6px;font-size:11px;color:#6B7385;'>"
            "<span style='width:12px;height:7px;background:#9CA3AF;border-radius:2px;display:inline-block;opacity:0.7;'></span>"
            "Feature Importance (impureza)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
