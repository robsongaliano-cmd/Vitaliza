# -*- coding: utf-8 -*-
"""
llm_explainer.py — Explicabilidade e análise prescritiva via LLM
Vitaliza | Artefato 2 — Módulo 2 Inteli MBA IA & Dados

Usa o OpenRouter (compatível com API OpenAI) para chamar Claude via chave OpenRouter.

Configuração:
    export OPENROUTER_API_KEY="sk-or-..."
    pip install openai
"""

import os
import json
from typing import Optional
from openai import OpenAI

FEATURE_NAMES_PT = {
    "gender": "Gênero",
    "Near_Location": "Próx. da academia",
    "Partner": "Tem parceiro(a)",
    "Promo_friends": "Indicado por amigo",
    "Phone": "Telefone cadastrado",
    "Contract_period": "Período do contrato",
    "Group_visits": "Visitas em grupo",
    "Age": "Idade",
    "Avg_additional_charges_total": "Gasto adicional médio",
    "Month_to_end_contract": "Meses até fim do contrato",
    "Lifetime": "Tempo de plataforma (meses)",
    "Avg_class_frequency_total": "Freq. média total de aulas",
    "Avg_class_frequency_current_month": "Freq. de aulas no mês atual",
}

# Modelo a usar via OpenRouter — Claude Sonnet é o melhor custo-benefício
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-5"

SYSTEM_PROMPT = """Você é um consultor especializado em retenção de clientes para a Vitaliza, 
uma plataforma de fitness digital. Você analisa dados de clientes e fornece:

1. EXPLICAÇÃO: Por que este cliente está (ou não está) em risco de cancelamento, 
   em linguagem clara para gestores não técnicos.

2. PRESCRIÇÃO: Uma recomendação de ação concreta e personalizada para o time de 
   Customer Success — qual intervenção fazer, com qual argumento, e por quê.

Seja direto, objetivo e use linguagem de negócios. Máximo 4 parágrafos no total.
Responda sempre em português brasileiro."""


def build_prompt(
    churn_prob: float,
    shap_values: dict[str, float],
    client_data: dict,
    feature_names_pt: dict = FEATURE_NAMES_PT,
) -> str:
    """Monta o prompt com os dados do cliente e os SHAP values."""

    risk_label = (
        "MUITO ALTO" if churn_prob >= 0.75 else
        "ALTO"       if churn_prob >= 0.50 else
        "MODERADO"   if churn_prob >= 0.30 else
        "BAIXO"
    )

    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    risk_factors    = [(f, v) for f, v in sorted_shap if v > 0.01][:5]
    protect_factors = [(f, v) for f, v in sorted_shap if v < -0.01][:3]

    def fmt_feature(feat, val):
        nome = feature_names_pt.get(feat, feat)
        dado = client_data.get(feat, "N/A")
        sinal = "↑ aumenta risco" if val > 0 else "↓ reduz risco"
        return f"  - {nome}: valor={dado} | impacto SHAP={val:+.3f} ({sinal})"

    risk_lines    = "\n".join(fmt_feature(f, v) for f, v in risk_factors)    or "  Nenhum fator de risco predominante."
    protect_lines = "\n".join(fmt_feature(f, v) for f, v in protect_factors) or "  Nenhum fator protetor relevante."

    return f"""Analise este cliente da Vitaliza e forneça explicação + prescrição.

=== SCORE DO MODELO ===
Probabilidade de churn: {churn_prob:.1%} (risco {risk_label})

=== FATORES QUE AUMENTAM O RISCO (SHAP positivos) ===
{risk_lines}

=== FATORES QUE REDUZEM O RISCO (SHAP negativos) ===
{protect_lines}

=== DADOS BRUTOS DO CLIENTE ===
{json.dumps({feature_names_pt.get(k, k): v for k, v in client_data.items()}, ensure_ascii=False, indent=2)}

Por favor, forneça:
**EXPLICAÇÃO:** (por que este cliente está neste nível de risco)
**PRESCRIÇÃO:** (o que o time de Customer Success deve fazer — seja específico na ação e no argumento)"""


def explain(
    churn_prob: float,
    shap_values: dict[str, float],
    client_data: dict,
    api_key: Optional[str] = None,
) -> str:
    """
    Chama o LLM via OpenRouter e retorna explicação + prescrição em texto.

    Args:
        churn_prob:   probabilidade de churn (0.0 a 1.0)
        shap_values:  dict {feature_name: shap_value} para este cliente
        client_data:  dict {feature_name: valor_original}
        api_key:      chave OpenRouter (ou usa OPENROUTER_API_KEY do env)

    Returns:
        String com a resposta do LLM
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "Chave da API não encontrada. "
            "Configure a variável de ambiente OPENROUTER_API_KEY ou passe api_key."
        )

    # OpenRouter usa o mesmo formato da API OpenAI
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = build_prompt(churn_prob, shap_values, client_data)

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    return response.choices[0].message.content


def explain_batch(
    results: list[dict],
    api_key: Optional[str] = None,
    top_n: int = 5,
) -> list[dict]:
    """Gera explicações para os N clientes com maior risco de churn."""
    sorted_results = sorted(results, key=lambda x: x["churn_prob"], reverse=True)
    top = sorted_results[:top_n]

    for item in top:
        item["explanation"] = explain(
            churn_prob=item["churn_prob"],
            shap_values=item["shap_values"],
            client_data=item["client_data"],
            api_key=api_key,
        )

    return top


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_shap = {
        "Lifetime": 0.139,
        "Avg_class_frequency_current_month": 0.029,
        "Month_to_end_contract": 0.028,
        "Contract_period": 0.028,
        "Age": 0.023,
        "Avg_additional_charges_total": -0.014,
        "Group_visits": -0.005,
    }
    sample_data = {
        "Lifetime": 2,
        "Avg_class_frequency_current_month": 0.5,
        "Month_to_end_contract": 1.0,
        "Contract_period": 1,
        "Age": 26,
        "Avg_additional_charges_total": 50.0,
        "Group_visits": 0,
    }
    print("Testando llm_explainer.py com OpenRouter...")
    result = explain(churn_prob=0.92, shap_values=sample_shap, client_data=sample_data)
    print(result)
