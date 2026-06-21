# -*- coding: utf-8 -*-
"""
llm_explainer.py — Briefing de personalização via LLM
Vitaliza | Artefato 2 — Módulo 2 Inteli MBA IA & Dados

Framework: As 5 Promessas (Empower · Know · Reach · Show · Delight)
Instrumento: Briefing de personalização — 7 campos
Fonte: Abraham & Edelman, "Personalization Done Right", HBR (nov-dez 2024)

Configuração:
    export OPENROUTER_API_KEY="sk-or-..."
    pip install openai
"""

import os
import json
from typing import Optional
from openai import OpenAI

# ── Nomes em português ────────────────────────────────────────────────────────
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

OPENROUTER_MODEL = "anthropic/claude-3-haiku"

# ── System prompt — persona do consultor ─────────────────────────────────────
SYSTEM_PROMPT = """Você é um consultor especializado em personalização e retenção de clientes para a Vitaliza, uma plataforma de fitness digital.

Seu trabalho segue o framework das 5 Promessas de Personalização (Abraham & Edelman, HBR 2024):
- EMPOWER ME: a mensagem deve ajudar o cliente a conseguir o que quer — não empurrar produto.
- KNOW ME: usar só dados legítimos e comportamentais, com transparência. Nunca revelar ao cliente que ele foi identificado como "em risco".
- REACH ME: recomendar o canal e o momento certos para aquele perfil específico.
- SHOW ME: produzir conteúdo único e relevante para o indivíduo — não mensagem genérica.
- DELIGHT ME: sugerir como medir e iterar.

Você entrega um BRIEFING DE PERSONALIZAÇÃO estruturado em 7 campos + uma mensagem pronta.
Responda sempre em português brasileiro. Seja direto e objetivo."""


# ── Lógica de persona ─────────────────────────────────────────────────────────
def _infer_persona(client_data: dict, shap_values: dict, churn_prob: float) -> dict:
    """Infere a persona e o contexto de risco a partir dos dados do cliente."""
    lifetime      = client_data.get("Lifetime", 0)
    freq_current  = client_data.get("Avg_class_frequency_current_month", 0)
    freq_total    = client_data.get("Avg_class_frequency_total", 0)
    contract      = client_data.get("Contract_period", 1)
    months_left   = client_data.get("Month_to_end_contract", 0)
    group_visits  = client_data.get("Group_visits", 0)
    promo_friends = client_data.get("Promo_friends", 0)
    age           = client_data.get("Age", 30)
    extra_spend   = client_data.get("Avg_additional_charges_total", 0)

    # Classificar persona
    if lifetime <= 1:
        persona_nome = "Novato em Risco"
        persona_desc = "cliente no primeiro mês, ainda não formou hábito"
    elif freq_current < freq_total * 0.5 and lifetime > 2:
        persona_nome = "Engajado em Queda"
        persona_desc = "cliente com histórico de uso mas frequência caindo no mês atual"
    elif contract >= 6 and freq_current == 0:
        persona_nome = "Silencioso Anual"
        persona_desc = "cliente em contrato longo que parou de usar — churn silencioso"
    elif group_visits == 0 and freq_current < 1.5:
        persona_nome = "Solitário Desengajado"
        persona_desc = "cliente sem vínculos sociais na plataforma e baixa frequência"
    else:
        persona_nome = "Em Risco Moderado"
        persona_desc = "cliente com sinais mistos, necessita atenção preventiva"

    # Gatilho principal (maior SHAP positivo)
    sorted_shap = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
    top_gatilho_feat, top_gatilho_val = sorted_shap[0] if sorted_shap else ("Lifetime", 0)
    top_gatilho_nome = FEATURE_NAMES_PT.get(top_gatilho_feat, top_gatilho_feat)

    # Canal recomendado por perfil
    if age < 28:
        canal = "push notification no app + story no Instagram"
    elif group_visits == 1 or promo_friends == 1:
        canal = "WhatsApp personalizado (via amigo/grupo como ponto de contato)"
    elif contract >= 6:
        canal = "e-mail com assunto personalizado + ligação do CS se sem resposta em 48h"
    else:
        canal = "e-mail personalizado + push de reativação"

    # Tom por perfil
    if lifetime <= 1:
        tom = "acolhedor e encorajador — é o primeiro contato real de valor"
    elif persona_nome == "Engajado em Queda":
        tom = "amigável e atento, sem culpa e sem alarme — reconhece o esforço passado"
    elif persona_nome == "Silencioso Anual":
        tom = "discreto e útil — oferece algo concreto sem expor que percebeu a ausência"
    else:
        tom = "direto e propositivo — uma sugestão clara de próximo passo"

    return {
        "persona_nome": persona_nome,
        "persona_desc": persona_desc,
        "top_gatilho_nome": top_gatilho_nome,
        "top_gatilho_val": top_gatilho_val,
        "canal": canal,
        "tom": tom,
        "freq_delta": round(freq_current - freq_total, 2),
        "contrato_tipo": f"{int(contract)} mês" if contract == 1 else f"{int(contract)} meses",
        "meses_restantes": months_left,
        "group_visits": group_visits,
        "extra_spend": extra_spend,
    }


# ── Construtor do prompt ──────────────────────────────────────────────────────
def build_prompt(
    churn_prob: float,
    shap_values: dict,
    client_data: dict,
) -> str:

    ctx = _infer_persona(client_data, shap_values, churn_prob)

    risk_label = (
        "MUITO ALTO" if churn_prob >= 0.75 else
        "ALTO"       if churn_prob >= 0.50 else
        "MODERADO"   if churn_prob >= 0.30 else
        "BAIXO"
    )

    # Top 4 fatores SHAP
    top_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    shap_lines = "\n".join(
        f"  - {FEATURE_NAMES_PT.get(f, f)}: {'+' if v > 0 else ''}{v:.3f} "
        f"({'aumenta' if v > 0 else 'reduz'} risco)"
        for f, v in top_shap
    )

    dados_cliente = {
        FEATURE_NAMES_PT.get(k, k): v
        for k, v in client_data.items()
    }

    return f"""Você recebeu os dados de um cliente da Vitaliza. Produza um BRIEFING DE PERSONALIZAÇÃO completo seguindo exatamente a estrutura abaixo.

=== CONTEXTO DO MODELO ===
Score de churn: {churn_prob:.1%} (risco {risk_label})
Persona inferida: {ctx['persona_nome']} — {ctx['persona_desc']}
Principal gatilho de risco: {ctx['top_gatilho_nome']} (SHAP {ctx['top_gatilho_val']:+.3f})
Delta de frequência (atual vs. média histórica): {ctx['freq_delta']:+.2f} aulas/semana

Top fatores SHAP:
{shap_lines}

Dados comportamentais:
{json.dumps(dados_cliente, ensure_ascii=False, indent=2)}

=== TAREFA ===
Produza o briefing nos 7 campos abaixo. Seja específico — use os dados reais acima em cada campo.
Ao final, escreva a MENSAGEM PRONTA para envio ao cliente.

Formate a resposta EXATAMENTE assim:

**BRIEFING DE PERSONALIZAÇÃO**

**1 · Persona-alvo**
[quem é este cliente, com base nos dados — sem usar a palavra "churn" ou "risco"]

**2 · Dados de entrada**
[comportamento observado + score + gatilho principal — o que justifica a ação agora]

**3 · Objetivo**
[ação observável e mensurável que o cliente deve tomar, com prazo — não "engajar", mas algo concreto]

**4 · Tom / estilo**
[como falar com este perfil específico — e o que jamais deve soar nesta mensagem]

**5 · Formato / canal**
[canal recomendado, formato (e-mail, push, WhatsApp), com justificativa baseada no perfil]

**6 · Comprimento**
[número de palavras ou segundos — e estrutura (ex: 1 parágrafo + 1 CTA)]

**7 · Restrições**
[mínimo 3 itens: o que esta mensagem NÃO pode fazer, prometer ou revelar — o campo mais importante]

---

**MENSAGEM PRONTA**
[Escreva aqui a mensagem real, no canal e tom especificados, pronta para envio. Use o nome genérico "você" — não invente nome. Inclua assunto se for e-mail.]

---

**PROMESSA EMPOWER ME**
[Uma frase: esta mensagem ajuda o cliente ou só a empresa? O que ela entrega de valor real para o cliente?]"""


# ── Função principal ──────────────────────────────────────────────────────────
def explain(
    churn_prob: float,
    shap_values: dict,
    client_data: dict,
    api_key: Optional[str] = None,
) -> str:
    """
    Gera o briefing de personalização + mensagem pronta via LLM.

    Args:
        churn_prob:   probabilidade de churn (0.0 a 1.0)
        shap_values:  dict {feature_name: shap_value} para este cliente
        client_data:  dict {feature_name: valor_original}
        api_key:      chave OpenRouter (ou usa OPENROUTER_API_KEY do env)

    Returns:
        String com o briefing estruturado + mensagem pronta
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "Chave da API não encontrada. "
            "Configure a variável OPENROUTER_API_KEY ou passe api_key."
        )

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = build_prompt(churn_prob, shap_values, client_data)

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        max_tokens=1200,
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    return response.choices[0].message.content


def explain_batch(
    results: list,
    api_key: Optional[str] = None,
    top_n: int = 5,
) -> list:
    """Gera briefings para os N clientes com maior risco."""
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
        "Lifetime": 0.252,
        "Month_to_end_contract": 0.062,
        "Avg_class_frequency_current_month": 0.022,
        "Age": -0.182,
        "Promo_friends": -0.034,
    }
    sample_data = {
        "Lifetime": 0,
        "Avg_class_frequency_current_month": 1.3,
        "Month_to_end_contract": 1.0,
        "Contract_period": 1,
        "Age": 32,
        "Avg_additional_charges_total": 161.0,
        "Group_visits": 0,
        "Promo_friends": 1,
        "gender": 1,
        "Near_Location": 1,
        "Partner": 1,
        "Phone": 0,
        "Avg_class_frequency_total": 1.7,
    }
    print("Testando briefing de personalização...\n")
    result = explain(churn_prob=0.675, shap_values=sample_shap, client_data=sample_data)
    print(result)
