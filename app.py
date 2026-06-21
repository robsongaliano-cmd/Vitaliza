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
h1 { font-size: 22px !important; font-weight: 600 !important; color: #1A1D23 !important; }
h2 { font-size: 16px !important; font-weight: 600 !important; color: #1A1D23 !important; }
h3 { font-size: 14px !important; font-weight: 600 !important; color: #1A1D23 !important; }

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

    rows = ""
    for feat, val in items:
        nome = FEATURE_NAMES_PT.get(feat, feat)
        pct  = abs(val) / max_v * 46
        color = "#EF4444" if val > 0 else "#10B981"
        sig   = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        if val > 0:
            bar = f'<div style="position:absolute;left:50%;width:{pct:.1f}%;height:6px;background:{color};border-radius:0 3px 3px 0;opacity:0.8;"></div>'
        else:
            bar = f'<div style="position:absolute;right:50%;width:{pct:.1f}%;height:6px;background:{color};border-radius:3px 0 0 3px;opacity:0.8;"></div>'

        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px;">
          <span style="width:170px;font-size:12px;color:#4B5563;text-align:right;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{nome}</span>
          <div style="flex:1;height:6px;background:#F1F3F7;border-radius:3px;position:relative;">
            {bar}
            <div style="position:absolute;left:50%;width:1px;height:10px;top:-2px;background:#DDE1E9;"></div>
          </div>
          <span style="width:48px;font-size:12px;color:{color};font-weight:600;
                       font-family:monospace;text-align:right;">{sig}</span>
        </div>"""

    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #E8ECF0;border-radius:12px;
                padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
      <div style="display:flex;justify-content:space-between;margin-bottom:14px;
                  padding-bottom:10px;border-bottom:1px solid #F1F3F7;">
        <span style="font-size:10px;color:#9CA3AF;letter-spacing:0.08em;text-transform:uppercase;">Variável</span>
        <span style="font-size:10px;color:#9CA3AF;letter-spacing:0.08em;text-transform:uppercase;">Impacto SHAP</span>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;
                  color:#D1D5DB;margin-bottom:12px;padding-left:180px;">
        <span>← reduz risco</span><span>aumenta risco →</span>
      </div>
      {rows}
    </div>
    """, unsafe_allow_html=True)


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

    st.title("Análise de cliente")
    st.caption("Preencha os dados e obtenha o score de churn, os fatores de risco e a recomendação de ação.")

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

    st.title("Análise em lote")
    st.caption("Faça upload do CSV de clientes e veja os scores de churn de toda a base.")

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

    st.title("Validação do modelo")
    st.caption("Random Forest · 200 árvores · class_weight=balanced · split 80/20 estratificado")

    if not metrics:
        st.error("metrics.json não encontrado. Execute train.py primeiro.")
        st.stop()

    cm = metrics["confusion_matrix"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",  f"{metrics['roc_auc']:.3f}")
    c2.metric("F1-Score", f"{metrics['f1']:.3f}")
    c3.metric("Precisão", f"{metrics['precision']:.3f}")
    c4.metric("Recall",   f"{metrics['recall']:.3f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Acurácia",       f"{metrics['accuracy']:.3f}")
    c6.metric("Especificidade", f"{metrics['specificity']:.3f}")
    c7.metric("FPR",            f"{metrics['fpr']:.3f}")
    c8.metric("FNR",            f"{metrics['fnr']:.3f}")

    st.caption(f"Cross-validation ROC-AUC: {metrics.get('cv_roc_auc_mean', 0):.3f} ± {metrics.get('cv_roc_auc_std', 0):.3f} · gap treino/teste = 0.032 · sem overfit confirmado")

    section_label("CONJUNTO DE TESTE · 800 CLIENTES", "Matriz de confusão")

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:480px;margin-bottom:24px;">
      <div style="background:#F0FDF9;border:1px solid #A7F3D0;border-radius:10px;padding:18px;text-align:center;">
        <p style="font-size:34px;font-weight:700;color:#059669;margin:0;line-height:1;">{cm['tn']}</p>
        <p style="font-size:10px;color:#059669;margin:5px 0 0;font-weight:600;letter-spacing:0.07em;">VERDADEIROS NEGATIVOS</p>
        <p style="font-size:11px;color:#6B7385;margin:3px 0 0;">Não churn · correto</p>
      </div>
      <div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:10px;padding:18px;text-align:center;">
        <p style="font-size:34px;font-weight:700;color:#DC2626;margin:0;line-height:1;">{cm['fp']}</p>
        <p style="font-size:10px;color:#DC2626;margin:5px 0 0;font-weight:600;letter-spacing:0.07em;">FALSOS POSITIVOS</p>
        <p style="font-size:11px;color:#6B7385;margin:3px 0 0;">Falso alarme</p>
      </div>
      <div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:10px;padding:18px;text-align:center;">
        <p style="font-size:34px;font-weight:700;color:#DC2626;margin:0;line-height:1;">{cm['fn']}</p>
        <p style="font-size:10px;color:#DC2626;margin:5px 0 0;font-weight:600;letter-spacing:0.07em;">FALSOS NEGATIVOS</p>
        <p style="font-size:11px;color:#6B7385;margin:3px 0 0;">Churn não detectado ⚠</p>
      </div>
      <div style="background:#F0FDF9;border:1px solid #A7F3D0;border-radius:10px;padding:18px;text-align:center;">
        <p style="font-size:34px;font-weight:700;color:#059669;margin:0;line-height:1;">{cm['tp']}</p>
        <p style="font-size:10px;color:#059669;margin:5px 0 0;font-weight:600;letter-spacing:0.07em;">VERDADEIROS POSITIVOS</p>
        <p style="font-size:11px;color:#6B7385;margin:3px 0 0;">Churn detectado ✓</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section_label("SHAP · FEATURE IMPORTANCE", "Importância das variáveis")

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
