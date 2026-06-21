# -*- coding: utf-8 -*-
"""
llm_explainer.py — Briefing de personalização via LLM
Vitaliza | Artefato 2 — Módulo 2 Inteli MBA IA & Dados

Framework: As 5 Promessas (Empower · Know · Reach · Show · Delight)
Instrumento: Briefing de personalização — 7 campos
Fonte: Abraham & Edelman, "Personalization Done Right", HBR (nov-dez 2024)

Integração com o Board Recommendation Deck:
  - Segmentos alinhados: Early Dropper · Disengaged · Monthly at Risk ·
    Loyal Emerging · Loyal Engaged · Socially Protected · Sleeping Dog
  - Prompt dinâmico incorpora segmento + score + variáveis relevantes do modelo
  - Trade-off precisão/recall explicitado por segmento para o operador de CS

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

# ── Segmentos do Board Deck — alinhados com a segmentação comportamental ──────
# Fonte: plataforma Vitaliza Churn Intelligence + Board Recommendation Deck
SEGMENTS = {
    "Early Dropper": {
        "descricao": "cliente no 1º mês, ainda não criou hábito (61,4% churn · LTV/CAC 1,40)",
        "driver": "Onboarding insuficiente — não percebe valor antes da 1ª cobrança",
        "hipotese": "CONFIRMADA no dataset",
        "acao_estrategica": "Onboarding ativo dias D3, D7, D14 — antes do dropout",
        "canal": "push notification no app (D3) + e-mail (D7) + ligação CS (D14)",
        "tom": "acolhedor e encorajador — primeiro contato real de valor, sem pressão",
        "restricoes": [
            "Não mencionar cancelamento ou risco de perda",
            "Não oferecer desconto na primeira mensagem — isso desvaloriza o produto",
            "Não prometer resultados físicos específicos",
            "Não revelar que identificou a ausência de uso",
        ],
        "nao_intervir": False,
        "ltv_cac": "1,40 — abaixo do break-even; prioridade é criar hábito, não desconto",
        "modelo_nota": "Recall elevado para este segmento é prioritário — perder um Early Dropper (FN) tem custo baixo mas volume alto; falso positivo (FP) gera custo de campanha desnecessário em cliente que não cancelaria",
    },
    "Disengaged": {
        "descricao": "queda de frequência >0,5 sess/semana em relação à média — 76,2% churn · LTV/CAC 0,22",
        "driver": "Deterioração detectável antes do cancelamento (hipótese H3 Marcelo — CONFIRMADA)",
        "hipotese": "CONFIRMADA — delta de frequência é o sinal preditivo mais forte (SHAP top feature)",
        "acao_estrategica": "Reengajamento humano em 48h ao detectar a queda — antes do cancelamento efetivo",
        "canal": "WhatsApp ou ligação do CS — contato humano, não automatizado",
        "tom": "amigável e atento, sem culpa e sem alarme — reconhece o esforço passado",
        "restricoes": [
            "Não mencionar que detectou a queda de frequência",
            "Não oferecer desconto como primeiro contato — sinaliza desespero",
            "Não prometer resultado físico imediato",
            "Não enviar push genérico — precisa de contato humano",
        ],
        "nao_intervir": False,
        "ltv_cac": "0,22 — custo de reter pode superar o valor; avaliar custo de intervir vs. deixar ir",
        "modelo_nota": "Threshold operacional 70 prioriza precisão aqui para evitar FP que acione Sleeping Dogs erroneamente; FN (Disengaged não detectado) tem custo alto pois churn é 76,2%",
    },
    "Monthly at Risk": {
        "descricao": "contrato mensal sem fidelidade — cada renovação é uma nova decisão (8,8% churn · LTV/CAC 13,59)",
        "driver": "Baixo comprometimento contratual — alternativas facilmente disponíveis",
        "hipotese": "CONFIRMADA — H1: contratos mensais têm 42,3% churn vs 2,4% anuais",
        "acao_estrategica": "Oferta de upgrade para semestral/anual com incentivo no mês 2",
        "canal": "e-mail com assunto personalizado + push de reforço 48h depois",
        "tom": "direto e propositivo — mostra o valor concreto do contrato longo",
        "restricoes": [
            "Não revelar que monitora o padrão de renovação",
            "Não prometer resultado físico como argumento de venda",
            "Não usar urgência falsa ('oferta expira em 24h')",
        ],
        "nao_intervir": False,
        "ltv_cac": "13,59 — maior retorno potencial de toda a base; cada cliente convertido para anual multiplica o LTV",
        "modelo_nota": "Segmento com maior ROI de retenção — precisão elevada importa para não desperdiçar oferta em quem já renovaria; recall secundário pois churn é 8,8%",
    },
    "Loyal Emerging": {
        "descricao": "em transição para engajamento pleno — 1,7% churn · LTV/CAC crescente",
        "driver": "Hábito em formação — empurrão social pode consolidar ou perder",
        "hipotese": "HIPÓTESE — sem evidência confirmatória suficiente",
        "acao_estrategica": "Convite para desafio em grupo ou atividade social — não retenção reativa",
        "canal": "push notification no app + convite in-app para grupo",
        "tom": "celebrativo e encorajador — reconhece o progresso, não corrige falha",
        "restricoes": [
            "Não tratar como cliente em risco — ele está em evolução positiva",
            "Não oferecer desconto — devalua o que ele está construindo",
            "Não revelar monitoramento",
        ],
        "nao_intervir": False,
        "ltv_cac": "crescente — objetivo é migrar para Loyal Engaged, não apenas reter",
        "modelo_nota": "Risco de FP alto se tratado como cliente em churn — modelo deve ser calibrado para não classificar Loyal Emerging como alto risco",
    },
    "Socially Protected": {
        "descricao": "protegido por vínculos sociais — grupos e indicações — 0,8% churn",
        "driver": "Custo social de sair é alto — rede de contatos na plataforma",
        "hipotese": "CONFIRMADA — H2: vínculos sociais reduzem churn de 33% para 10,7%",
        "acao_estrategica": "Fortalecer a rede social — não intervir individualmente",
        "canal": "notificação de grupo ou atividade coletiva — nunca mensagem individual de retenção",
        "tom": "comunitário — endereça o grupo, não o indivíduo",
        "restricoes": [
            "Não contatar individualmente com oferta — pode romper a sensação de comunidade",
            "Não revelar monitoramento de uso",
            "Não oferecer desconto — ele não está saindo por preço",
        ],
        "nao_intervir": False,
        "ltv_cac": "alto — custo de intervenção deve ser coletivo, não individual",
        "modelo_nota": "Falso positivo (FP) é o risco principal — acionar este cliente individualmente pode perturbar o vínculo social que o retém",
    },
    "Sleeping Dog": {
        "descricao": "ativo por inércia — lifetime >6m + frequência <0,5/sem — 1,9% churn · R$153k ARR passivo",
        "driver": "Pagamento por inércia — qualquer contato pode virar lembrete de cancelar",
        "hipotese": "CASE EXPLÍCITO — 'don't wake the sleeping dogs'",
        "acao_estrategica": "NÃO INTERVIR — excluir de TODAS as campanhas",
        "canal": "NENHUM — cliente excluído da lista de disparo",
        "tom": "N/A — não gerar mensagem",
        "restricoes": [
            "NÃO ENVIAR NENHUMA MENSAGEM",
            "Excluir explicitamente de toda lista de campanha",
            "Monitorar apenas para detectar saída natural na renovação",
        ],
        "nao_intervir": True,
        "ltv_cac": "passivo — R$153k ARR em risco de ser destruído por qualquer notificação",
        "modelo_nota": "Threshold 70 é insuficiente para este segmento — Sleeping Dogs têm score alto por padrão comportamental, mas NÃO devem ser acionados. Flag sleeping_dog_flag deve excluí-los ANTES do disparo, independente do score",
    },
    "Em Risco Moderado": {
        "descricao": "sinais mistos — necessita atenção preventiva",
        "driver": "Combinação de fatores sem padrão dominante claro",
        "hipotese": "EM AVALIAÇÃO — aguardar mais dados do pipeline",
        "acao_estrategica": "Monitorar por 2 semanas antes de intervir",
        "canal": "e-mail de valor (conteúdo, não oferta) — sem urgência",
        "tom": "útil e discreto — entrega valor sem expor monitoramento",
        "restricoes": [
            "Não oferecer desconto sem evidência de sensibilidade a preço",
            "Não revelar monitoramento",
            "Não criar urgência artificial",
        ],
        "nao_intervir": False,
        "ltv_cac": "variável — depende do contrato",
        "modelo_nota": "Score moderado (30-65%) — avaliar custo de intervenção vs. probabilidade de churn antes de acionar CS",
    },
}

# ── System prompt — persona do consultor ─────────────────────────────────────
SYSTEM_PROMPT = """Você é um consultor especializado em personalização e retenção de clientes para a Vitaliza, uma plataforma de fitness digital.

Seu trabalho segue o framework das 5 Promessas de Personalização (Abraham & Edelman, HBR 2024):
- EMPOWER ME: a mensagem deve ajudar o cliente a conseguir o que quer — não empurrar produto.
- KNOW ME: usar só dados legítimos e comportamentais, com transparência. Nunca revelar ao cliente que ele foi identificado como "em risco".
- REACH ME: recomendar o canal e o momento certos para aquele perfil específico.
- SHOW ME: produzir conteúdo único e relevante para o indivíduo — não mensagem genérica.
- DELIGHT ME: sugerir como medir e iterar.

Você recebe o segmento comportamental do cliente (alinhado com o Board Recommendation Deck da Vitaliza),
o score do modelo preditivo (Random Forest · AUC 0,977 · threshold 70), os SHAP values das features
mais relevantes e os dados comportamentais reais do CSV.

O prompt que você recebe já incorpora dinamicamente todas essas variáveis — seu trabalho é usá-las
para produzir um briefing único para este cliente, não um template genérico.

Você entrega um BRIEFING DE PERSONALIZAÇÃO estruturado em 7 campos + uma mensagem pronta.
Responda sempre em português brasileiro. Seja direto e objetivo."""


# ── Classificação de segmento — alinhada com o Board Deck ────────────────────
def _classify_segment(client_data: dict, shap_values: dict, churn_prob: float) -> tuple[str, dict]:
    """
    Classifica o cliente em um dos 7 segmentos do Board Recommendation Deck.
    Usa as mesmas regras determinísticas da plataforma Vitaliza Churn Intelligence.

    Retorna: (nome_segmento, metadados_do_segmento)
    """
    lifetime     = client_data.get("Lifetime", 0)
    freq_atual   = client_data.get("Avg_class_frequency_current_month", 0)
    freq_total   = client_data.get("Avg_class_frequency_total", 0)
    contract     = client_data.get("Contract_period", 1)
    group_visits = client_data.get("Group_visits", 0)
    promo        = client_data.get("Promo_friends", 0)
    freq_delta   = freq_total - freq_atual  # queda de frequência

    # Regras sequenciais — mesma lógica da plataforma
    if lifetime > 6 and freq_atual < 0.5:
        seg = "Sleeping Dog"
    elif freq_delta > 0.5 and lifetime > 1:
        seg = "Disengaged"
    elif lifetime <= 1:
        seg = "Early Dropper"
    elif contract == 1 and lifetime >= 2:
        seg = "Monthly at Risk"
    elif (group_visits == 1 or promo == 1) and freq_atual >= 1.0:
        seg = "Socially Protected"
    elif freq_atual >= 1.5 and lifetime >= 3 and churn_prob < 0.20:
        seg = "Loyal Engaged" if freq_atual >= 2.5 else "Loyal Emerging"
    else:
        seg = "Em Risco Moderado"

    return seg, SEGMENTS.get(seg, SEGMENTS["Em Risco Moderado"])


# ── Construtor do prompt dinâmico ─────────────────────────────────────────────
def build_prompt(
    churn_prob: float,
    shap_values: dict,
    client_data: dict,
) -> str:
    """
    Constrói o prompt com engenharia dinâmica:
    - segmento classificado (Board Deck)
    - score do modelo + threshold operacional
    - top SHAP values com direção e magnitude
    - dados comportamentais reais
    - contexto de trade-off precisão/recall por segmento
    """
    seg_nome, seg_meta = _classify_segment(client_data, shap_values, churn_prob)

    # Não gerar briefing para Sleeping Dog
    if seg_meta["nao_intervir"]:
        return _build_sleeping_dog_prompt(seg_nome, seg_meta, client_data, churn_prob)

    risk_label = (
        "MUITO ALTO" if churn_prob >= 0.75 else
        "ALTO"       if churn_prob >= 0.50 else
        "MODERADO"   if churn_prob >= 0.30 else
        "BAIXO"
    )

    # Score vs threshold
    score_int = int(churn_prob * 100)
    acima_threshold = score_int >= 70

    # Top 5 SHAP — separados por direção
    shap_sorted = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    fatores_risco    = [(f, v) for f, v in shap_sorted if v > 0.005]
    fatores_protecao = [(f, v) for f, v in shap_sorted if v < -0.005]

    def fmt_shap(feat, val):
        nome = FEATURE_NAMES_PT.get(feat, feat)
        dado = client_data.get(feat, "N/A")
        direcao = "↑ aumenta risco" if val > 0 else "↓ reduz risco"
        return f"  • {nome}: valor={dado} | SHAP={val:+.3f} ({direcao})"

    risco_lines    = "\n".join(fmt_shap(f, v) for f, v in fatores_risco)    or "  (nenhum)"
    protecao_lines = "\n".join(fmt_shap(f, v) for f, v in fatores_protecao) or "  (nenhum)"

    # Dados brutos em PT
    dados_pt = {FEATURE_NAMES_PT.get(k, k): v for k, v in client_data.items()}

    # Delta de frequência
    freq_delta = client_data.get("Avg_class_frequency_total", 0) - client_data.get("Avg_class_frequency_current_month", 0)

    return f"""Você recebeu os dados de um cliente da Vitaliza. Produza um BRIEFING DE PERSONALIZAÇÃO completo.

=== MODELO PREDITIVO — OUTPUT ===
Score de churn: {score_int}/100 (probabilidade {churn_prob:.1%} · risco {risk_label})
Threshold operacional: 70 — score {'ACIMA' if acima_threshold else 'ABAIXO'} do threshold → {'intervenção recomendada' if acima_threshold else 'monitorar'}
Modelo: Random Forest · AUC 0,977 · Recall 86% · Precisão 89% · threshold 70

=== SEGMENTO COMPORTAMENTAL (Board Recommendation Deck) ===
Segmento: {seg_nome}
Descrição: {seg_meta['descricao']}
Driver hipotético: {seg_meta['driver']}
Status hipótese: {seg_meta['hipotese']}
LTV/CAC do segmento: {seg_meta['ltv_cac']}
Ação estratégica definida para este segmento: {seg_meta['acao_estrategica']}

=== ANÁLISE DE TRADE-OFF PRECISÃO/RECALL PARA ESTE SEGMENTO ===
{seg_meta['modelo_nota']}

=== FATORES QUE AUMENTAM O RISCO (SHAP positivos) ===
{risco_lines}

=== FATORES QUE REDUZEM O RISCO (SHAP negativos) ===
{protecao_lines}

=== DELTA DE FREQUÊNCIA ===
Queda de frequência (hist. - atual): {freq_delta:+.2f} sess/semana {'⚠ SINAL DE ALERTA' if freq_delta > 0.5 else ''}

=== DADOS COMPORTAMENTAIS DO CLIENTE ===
{json.dumps(dados_pt, ensure_ascii=False, indent=2)}

=== CONTEXTO PRÉ-DEFINIDO PARA ESTE SEGMENTO ===
Canal recomendado: {seg_meta['canal']}
Tom / estilo: {seg_meta['tom']}
Restrições conhecidas:
{chr(10).join('  • ' + r for r in seg_meta['restricoes'])}

=== TAREFA ===
Produza o briefing NOS 7 CAMPOS abaixo. Use os dados reais acima — não use placeholders.
O briefing deve ser específico para ESTE cliente, não um template do segmento.

Formate a resposta EXATAMENTE assim:

**BRIEFING DE PERSONALIZAÇÃO — {seg_nome.upper()}**

**1 · Persona-alvo**
[quem é este cliente específico, com base nos dados reais — sem usar as palavras "churn" ou "risco"]

**2 · Dados de entrada**
[comportamento observado + score {score_int}/100 + principal feature SHAP + o que justifica agir AGORA]

**3 · Objetivo**
[ação observável e mensurável, com prazo — não "engajar", mas algo concreto e verificável]

**4 · Tom / estilo**
[como falar com ESTE cliente — o que jamais deve soar nesta mensagem específica]

**5 · Formato / canal**
[canal específico com justificativa baseada nos dados — não no segmento genérico]

**6 · Comprimento**
[número de palavras e estrutura exata: ex. "Assunto (6 palavras) + 2 parágrafos + 1 CTA"]

**7 · Restrições**
[mínimo 4 itens: o que esta mensagem NÃO pode fazer, prometer ou revelar]

---

**MENSAGEM PRONTA**
[A mensagem real, no canal e tom especificados, pronta para envio. Use "você". Inclua assunto se for e-mail ou WhatsApp.]

---

**NOTA DO MODELO — TRADE-OFF PARA O OPERADOR DE CS**
[Uma frase sobre o que o score {score_int}/100 significa para quem vai executar a intervenção: o risco de falso positivo (FP) e falso negativo (FN) neste caso específico, e o que isso implica para a decisão de acionar ou não.]"""


def _build_sleeping_dog_prompt(seg_nome, seg_meta, client_data, churn_prob):
    """Prompt especial para Sleeping Dog — instrui o LLM a NÃO gerar mensagem."""
    score_int = int(churn_prob * 100)
    return f"""Este cliente foi classificado como SLEEPING DOG pelo modelo.

=== CLASSIFICAÇÃO ===
Segmento: {seg_nome}
Score: {score_int}/100
LTV passivo: {seg_meta['ltv_cac']}
Regra do Board Deck: EXCLUIR DE TODAS AS CAMPANHAS

=== ALERTA DO MODELO ===
{seg_meta['modelo_nota']}

=== TAREFA ===
Produza uma resposta estruturada com:

**ALERTA — SLEEPING DOG IDENTIFICADO**

**Por que NÃO intervir:**
[Explique em 2 frases por que qualquer mensagem é contraproducente para este cliente]

**Risco financeiro de uma intervenção:**
[Quantifique o risco: ARR passivo em jogo + probabilidade de aceleração do cancelamento]

**O que fazer:**
[Ação correta: excluir da lista, monitorar renovação passivamente]

**NOTA DO MODELO — TRADE-OFF:**
[Score {score_int}/100 parece alto, mas este é um falso positivo sistemático do modelo para Sleeping Dogs — explique por que o threshold 70 não se aplica a este segmento]"""


# ── Função principal ──────────────────────────────────────────────────────────
def explain(
    churn_prob: float,
    shap_values: dict,
    client_data: dict,
    api_key: Optional[str] = None,
) -> str:
    """
    Gera o briefing de personalização via LLM com prompt dinâmico.

    O prompt incorpora:
    - Segmento comportamental (alinhado com Board Recommendation Deck)
    - Score do modelo + comparação com threshold operacional (70)
    - SHAP values das features mais relevantes com direção
    - Dados comportamentais reais do cliente
    - Trade-off precisão/recall por segmento para o operador de CS

    Args:
        churn_prob:   probabilidade de churn (0.0 a 1.0)
        shap_values:  dict {feature_name: shap_value} para este cliente
        client_data:  dict {feature_name: valor_original}
        api_key:      chave OpenRouter (ou usa OPENROUTER_API_KEY do env)

    Returns:
        String com o briefing estruturado + mensagem pronta + nota para o operador
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
        max_tokens=1400,
        temperature=0.35,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    return response.choices[0].message.content


def get_segment(client_data: dict, shap_values: dict, churn_prob: float) -> str:
    """Retorna o nome do segmento para exibição no app."""
    seg_nome, _ = _classify_segment(client_data, shap_values, churn_prob)
    return seg_nome


def explain_batch(
    results: list,
    api_key: Optional[str] = None,
    top_n: int = 5,
) -> list:
    """Gera briefings para os N clientes com maior risco (excluindo Sleeping Dogs)."""
    sorted_results = sorted(results, key=lambda x: x["churn_prob"], reverse=True)

    # Filtrar Sleeping Dogs da lista de intervenção
    actionable = []
    sleeping_dogs = []
    for item in sorted_results:
        seg, meta = _classify_segment(item["client_data"], item["shap_values"], item["churn_prob"])
        item["segment"] = seg
        if meta["nao_intervir"]:
            sleeping_dogs.append(item)
        else:
            actionable.append(item)

    top = actionable[:top_n]
    for item in top:
        item["explanation"] = explain(
            churn_prob=item["churn_prob"],
            shap_values=item["shap_values"],
            client_data=item["client_data"],
            api_key=api_key,
        )

    return top, sleeping_dogs


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Caso 1: Early Dropper (esperado)
    print("=== TESTE 1: Early Dropper ===")
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
        "Avg_class_frequency_total": 1.3,
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
    }
    seg, meta = _classify_segment(sample_data, sample_shap, 0.675)
    print(f"Segmento classificado: {seg}")
    print(f"Não intervir: {meta['nao_intervir']}")

    # Caso 2: Sleeping Dog
    print("\n=== TESTE 2: Sleeping Dog ===")
    sd_data = dict(sample_data)
    sd_data["Lifetime"] = 8
    sd_data["Avg_class_frequency_current_month"] = 0.2
    sd_data["Avg_class_frequency_total"] = 2.5
    seg2, meta2 = _classify_segment(sd_data, sample_shap, 0.85)
    print(f"Segmento classificado: {seg2}")
    print(f"Não intervir: {meta2['nao_intervir']}")

    print("\nTestando briefing Early Dropper via LLM...")
    result = explain(churn_prob=0.675, shap_values=sample_shap, client_data=sample_data)
    print(result)
