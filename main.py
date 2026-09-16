import os
import sys
import time
import requests
import smtplib
from email.message import EmailMessage
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from fredapi import Fred
from google import genai
from google.genai import types
from google.genai.errors import ServerError


# =====================================================================
# 1. LEITURA DE VARIÁVEIS DE AMBIENTE (SECRETS)
# =====================================================================

FRED_API_KEY = os.environ.get("FRED_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Configuração do e-mail
EMAIL_RECIPIENT = "heldermendes1979@gmail.com"
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", EMAIL_RECIPIENT)
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_SMTP_HOST = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "465"))


# =====================================================================
# 2. CONFIGURAÇÕES
# =====================================================================

LOOKBACK_YEARS = 2

# Número máximo de caracteres por mensagem do Telegram.
# Mantido abaixo do limite oficial para permitir alguma margem.
TELEGRAM_MAX_CHAR = 3800
# Envie o payload técnico ao Telegram apenas quando necessário.
SEND_RAW_PAYLOAD_TO_TELEGRAM = False


# =====================================================================
# 3. MAPA DE SÉRIES DO FRED
# =====================================================================
#
# A análise foi dividida em grupos:
#
# LIQUIDEZ
# JUROS
# INFLAÇÃO
# ATIVIDADE
# MERCADO DE TRABALHO
# CRÉDITO
# CONDIÇÕES FINANCEIRAS
# MERCADOS
# DÓLAR / COMMODITIES
#
# Observação:
# Algumas séries possuem frequências diferentes (diária, semanal,
# mensal ou trimestral). O processamento posterior normaliza as datas.
# =====================================================================

# =====================================================================
# DICIONÁRIO DE METADADOS DAS SÉRIES DO FRED
# =====================================================================
#
# Campos:
#
# fred_id:
#     ID oficial da série no FRED.
#
# nome:
#     Nome amigável usado no relatório.
#
# grupo:
#     Bloco macroeconômico ao qual pertence.
#
# unidade:
#     Unidade econômica da série.
#
# natureza:
#     Tipo econômico do indicador:
#     estoque / fluxo / taxa / índice / spread / preço.
#
# frequencia:
#     Frequência original aproximada da série.
#
# maior_e_melhor:
#     Indica se, isoladamente, um aumento costuma ser positivo para
#     a variável econômica/mercado analisado.
#
# interpretacao_alta:
#     Como interpretar um aumento.
#
# interpretacao_baixa:
#     Como interpretar uma queda.
#
# horizonte_principal:
#     Horizonte mais adequado para avaliar a tendência.
#
# comparacoes:
#     Janelas temporais mais úteis.
#
# tipo_momentum:
#     Como avaliar momentum.
#
# impacto_risco:
#     Relação geral com risk-on/risk-off.
#
# observacao:
#     Cuidados específicos para interpretação.
# =====================================================================

SERIES_METADATA = {

    # ================================================================
    # LIQUIDEZ
    # ================================================================

    "WALCL": {
        "nome": "Fed Total Assets",
        "coluna": "Fed_Total_Assets_M",
        "grupo": "Liquidez",
        "unidade": "USD milhões",
        "natureza": "estoque",
        "frequencia": "semanal",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Expansão do balanço do Fed. Pode representar maior liquidez, "
            "mas não deve ser interpretada isoladamente como expansão de "
            "liquidez disponível para ativos de risco."
        ),
        "interpretacao_baixa": (
            "Contração do balanço do Fed, geralmente associada a QT quando "
            "não houver outros fatores compensatórios."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual e absoluta",
        "impacto_risco": "expansão tende a favorecer condições financeiras",
        "observacao": (
            "Não confundir tamanho do balanço com liquidez líquida."
        ),
    },

    "WTREGEN": {
        "nome": "Treasury General Account (TGA)",
        "coluna": "TGA_Balance_M",
        "grupo": "Liquidez",
        "unidade": "USD milhões",
        "natureza": "estoque",
        "frequencia": "semanal",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Em igualdade de condições, aumento do TGA retira liquidez "
            "do sistema bancário."
        ),
        "interpretacao_baixa": (
            "Em igualdade de condições, redução do TGA libera liquidez "
            "para o sistema."
        ),
        "horizonte_principal": "30-90 dias",
        "comparacoes": ["30D", "90D"],
        "tipo_momentum": "variação absoluta",
        "impacto_risco": "TGA subindo tende a ser restritivo para liquidez",
        "observacao": (
            "O efeito depende da forma de financiamento e dos fluxos "
            "entre Tesouro, bancos e mercado."
        ),
    },

    "RRPONTSYD": {
        "nome": "ON RRP",
        "coluna": "ON_RRP_B",
        "grupo": "Liquidez",
        "unidade": "USD bilhões",
        "natureza": "estoque",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Maior utilização do ON RRP representa recursos estacionados "
            "no Fed e pode reduzir liquidez disponível nos mercados."
        ),
        "interpretacao_baixa": (
            "Queda do ON RRP pode liberar recursos para o sistema financeiro."
        ),
        "horizonte_principal": "30-90 dias",
        "comparacoes": ["30D", "90D"],
        "tipo_momentum": "variação absoluta",
        "impacto_risco": "queda tende a ser favorável à liquidez enquanto houver saldo relevante",
        "observacao": (
            "O impacto marginal diminui à medida que o saldo se aproxima "
            "de níveis muito baixos."
        ),
    },

    "WRESBAL": {
        "nome": "Bank Reserves",
        "coluna": "Bank_Reserves_M",
        "grupo": "Liquidez",
        "unidade": "USD milhões",
        "natureza": "estoque",
        "frequencia": "semanal",
        "maior_e_melhor": True,
        "interpretacao_alta": (
            "Maior nível de reservas bancárias tende a indicar condições "
            "de liquidez bancária mais confortáveis."
        ),
        "interpretacao_baixa": (
            "Queda das reservas pode indicar redução da folga de liquidez."
        ),
        "horizonte_principal": "30-90 dias",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação absoluta",
        "impacto_risco": "reservas crescendo tendem a favorecer condições financeiras",
        "observacao": (
            "Reservas elevadas não significam necessariamente expansão "
            "do crédito ou dos preços dos ativos."
        ),
    },

    "M2SL": {
        "nome": "M2",
        "coluna": "M2_B",
        "grupo": "Liquidez",
        "unidade": "USD bilhões",
        "natureza": "estoque monetário",
        "frequencia": "mensal",
        "maior_e_melhor": True,
        "interpretacao_alta": (
            "Expansão da quantidade de moeda ampla na economia."
        ),
        "interpretacao_baixa": (
            "Contração ou desaceleração da moeda ampla."
        ),
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["90D", "365D"],
        "tipo_momentum": "variação percentual anualizada",
        "impacto_risco": "crescimento monetário tende a apoiar condições financeiras no médio prazo",
        "observacao": (
            "M2 é mensal e possui relação imperfeita e variável com preços "
            "dos ativos no curto prazo."
        ),
    },


    # ================================================================
    # POLÍTICA MONETÁRIA
    # ================================================================

    "FEDFUNDS": {
        "nome": "Federal Funds Rate",
        "coluna": "Fed_Funds_Rate",
        "grupo": "Política Monetária",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "mensal",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Política monetária mais restritiva, em igualdade de condições."
        ),
        "interpretacao_baixa": (
            "Política monetária mais expansionista, em igualdade de condições."
        ),
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "taxas maiores tendem a apertar condições financeiras",
        "observacao": (
            "Interpretar junto com inflação, crescimento e expectativas "
            "de política monetária."
        ),
    },

    "EFFR": {
        "nome": "Effective Federal Funds Rate",
        "coluna": "Effective_Fed_Funds_Rate",
        "grupo": "Política Monetária",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": "Condições monetárias mais restritivas.",
        "interpretacao_baixa": "Condições monetárias menos restritivas.",
        "horizonte_principal": "1-30 dias",
        "comparacoes": ["30D", "90D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "taxa maior tende a restringir liquidez",
        "observacao": (
            "Usar em conjunto com Fed Funds e demais indicadores."
        ),
    },


    # ================================================================
    # CURVA DE JUROS
    # ================================================================

    "SOFR": {
        "nome": "Secured Overnight Financing Rate",
        "coluna": "SOFR",
        "grupo": "Política Monetária",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": "Condições de financiamento overnight mais restritivas.",
        "interpretacao_baixa": "Condições de financiamento overnight menos restritivas.",
        "horizonte_principal": "1-30 dias",
        "comparacoes": ["7D", "30D", "90D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "alta persistente tende a apertar condições financeiras de curto prazo",
        "observacao": "Usar em conjunto com Fed Funds, EFFR e demais indicadores de funding.",
    },

    "DGS3MO": {
        "nome": "Treasury 3M",
        "coluna": "Yield_3M",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior custo de financiamento de curto prazo.",
        "interpretacao_baixa": "Menor custo de financiamento de curto prazo.",
        "horizonte_principal": "1-90 dias",
        "comparacoes": ["30D", "90D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "taxas curtas maiores tendem a apertar condições financeiras",
        "observacao": "Interpretar principalmente em conjunto com Fed Funds.",
    },

    "DGS2": {
        "nome": "Treasury 2Y",
        "coluna": "Yield_2Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Mercado pode estar precificando juros de curto/médio prazo "
            "mais altos ou maior prêmio."
        ),
        "interpretacao_baixa": (
            "Pode indicar expectativa de política monetária mais branda."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "aumento persistente tende a pressionar ativos de risco",
        "observacao": "Não interpretar isoladamente.",
    },

    "DGS5": {
        "nome": "Treasury 5Y",
        "coluna": "Yield_5Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Maior taxa intermediária; pode refletir inflação, crescimento "
            "ou expectativas de juros."
        ),
        "interpretacao_baixa": "Menor taxa intermediária.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "depende da causa do movimento",
        "observacao": "Separar juros reais de expectativas de inflação.",
    },

    "DGS10": {
        "nome": "Treasury 10Y",
        "coluna": "Yield_10Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Pode refletir maior crescimento, inflação, prêmio de prazo "
            "ou risco fiscal."
        ),
        "interpretacao_baixa": (
            "Pode refletir menor crescimento, inflação ou busca por segurança."
        ),
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "depende do componente responsável pelo movimento",
        "observacao": (
            "Analisar conjuntamente com real yield e breakeven."
        ),
    },

    "DGS30": {
        "nome": "Treasury 30Y",
        "coluna": "Yield_30Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Pode refletir maior prêmio de prazo, inflação esperada, "
            "crescimento ou risco fiscal."
        ),
        "interpretacao_baixa": "Menor taxa longa.",
        "horizonte_principal": "6-24 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "aumento persistente pode pressionar valuation",
        "observacao": "Muito sensível ao prêmio de prazo.",
    },

    "T10Y2Y": {
        "nome": "Curva 10Y-2Y",
        "coluna": "Yield_Curve_10Y2Y",
        "grupo": "Curva de Juros",
        "unidade": "pontos percentuais",
        "natureza": "spread de taxas",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Maior inclinação da curva. A interpretação depende da causa."
        ),
        "interpretacao_baixa": (
            "Maior achatamento ou inversão da curva."
        ),
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "inversão pode sinalizar restrição monetária; desinclinação pode ter causas diversas",
        "observacao": (
            "Não interpretar simplesmente curva positiva como bullish "
            "ou negativa como bearish."
        ),
    },

    "T10Y3M": {
        "nome": "Curva 10Y-3M",
        "coluna": "Yield_Curve_10Y3M",
        "grupo": "Curva de Juros",
        "unidade": "pontos percentuais",
        "natureza": "spread de taxas",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": "Maior inclinação.",
        "interpretacao_baixa": "Maior achatamento ou inversão.",
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "deve ser analisado em conjunto com ciclo econômico",
        "observacao": "Indicador de curva, não previsão isolada de recessão.",
    },


    # ================================================================
    # JUROS REAIS
    # ================================================================

    "DFII5": {
        "nome": "Real Yield 5Y",
        "coluna": "Real_Yield_5Y",
        "grupo": "Juros Reais",
        "unidade": "%",
        "natureza": "taxa real",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Aumento do custo real de capital."
        ),
        "interpretacao_baixa": (
            "Redução do custo real de capital."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "juros reais maiores tendem a pressionar valuation",
        "observacao": "Importante para ações de duration longa e ouro.",
    },

    "DFII10": {
        "nome": "Real Yield 10Y",
        "coluna": "Real_Yield_10Y",
        "grupo": "Juros Reais",
        "unidade": "%",
        "natureza": "taxa real",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Maior custo real de capital e maior taxa livre de risco real."
        ),
        "interpretacao_baixa": (
            "Menor custo real de capital."
        ),
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": (
            "aumento tende a pressionar ativos de duration longa e ouro"
        ),
        "observacao": (
            "Uma das séries mais importantes para valuation de ativos de risco."
        ),
    },

    "DFII30": {
        "nome": "Real Yield 30Y",
        "coluna": "Real_Yield_30Y",
        "grupo": "Juros Reais",
        "unidade": "%",
        "natureza": "taxa real",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior custo real de capital de longo prazo.",
        "interpretacao_baixa": "Menor custo real de capital.",
        "horizonte_principal": "6-24 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "aumento tende a pressionar valuation de duration longa",
        "observacao": "Interpretar em conjunto com Treasury 30Y.",
    },


    # ================================================================
    # EXPECTATIVAS DE INFLAÇÃO
    # ================================================================

    "T5YIE": {
        "nome": "Breakeven Inflation 5Y",
        "coluna": "Breakeven_Inflation_5Y",
        "grupo": "Expectativas de Inflação",
        "unidade": "%",
        "natureza": "expectativa implícita",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Maior inflação implícita esperada pelo mercado."
        ),
        "interpretacao_baixa": (
            "Menor inflação implícita esperada."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "aumento persistente pode pressionar juros e valuation",
        "observacao": (
            "Não representa expectativa pura de inflação; contém prêmio "
            "de liquidez e outros componentes."
        ),
    },

    "T10YIE": {
        "nome": "Breakeven Inflation 10Y",
        "coluna": "Breakeven_Inflation_10Y",
        "grupo": "Expectativas de Inflação",
        "unidade": "%",
        "natureza": "expectativa implícita",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": "Maior inflação implícita de longo prazo.",
        "interpretacao_baixa": "Menor inflação implícita.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "aumento pode pressionar juros nominais",
        "observacao": "Interpretar junto com juros reais.",
    },


    # ================================================================
    # INFLAÇÃO
    # ================================================================

    "CPIAUCSL": {
        "nome": "CPI",
        "coluna": "CPI",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior nível de preços.",
        "interpretacao_baixa": "Menor nível de preços.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual anual",
        "impacto_risco": "inflação persistente pode manter política monetária restritiva",
        "observacao": (
            "O nível do índice não deve ser confundido com a taxa de inflação."
        ),
    },

    "CPILFESL": {
        "nome": "Core CPI",
        "coluna": "Core_CPI",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior inflação subjacente.",
        "interpretacao_baixa": "Menor inflação subjacente.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual anual",
        "impacto_risco": "persistência pode manter juros elevados",
        "observacao": "Dar atenção a serviços e componentes persistentes.",
    },

    "PCEPI": {
        "nome": "PCE",
        "coluna": "PCE",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior nível de preços.",
        "interpretacao_baixa": "Menor nível de preços.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual anual",
        "impacto_risco": "inflação elevada tende a limitar cortes de juros",
        "observacao": "Indicador de inflação preferido pelo Fed.",
    },

    "PCEPILFE": {
        "nome": "Core PCE",
        "coluna": "Core_PCE",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior inflação subjacente.",
        "interpretacao_baixa": "Menor inflação subjacente.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual anual",
        "impacto_risco": "persistência tende a manter política monetária restritiva",
        "observacao": "Dar preferência ao momentum da inflação, não apenas ao nível.",
    },


    # ================================================================
    # ATIVIDADE
    # ================================================================

    "GDPC1": {
        "nome": "Real GDP",
        "coluna": "Real_GDP",
        "grupo": "Ciclo Econômico",
        "unidade": "USD bilhões encadeados",
        "natureza": "fluxo",
        "frequencia": "trimestral",
        "maior_e_melhor": True,
        "interpretacao_alta": "Maior atividade econômica.",
        "interpretacao_baixa": "Menor atividade econômica.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["90D", "365D"],
        "tipo_momentum": "crescimento trimestral/anual",
        "impacto_risco": "crescimento maior tende a favorecer ativos cíclicos",
        "observacao": (
            "É um indicador atrasado; não deve ser usado sozinho para "
            "identificar mudanças recentes."
        ),
    },

    "INDPRO": {
        "nome": "Industrial Production",
        "coluna": "Industrial_Production",
        "grupo": "Ciclo Econômico",
        "unidade": "índice",
        "natureza": "índice de atividade",
        "frequencia": "mensal",
        "maior_e_melhor": True,
        "interpretacao_alta": "Maior produção industrial.",
        "interpretacao_baixa": "Menor produção industrial.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual",
        "impacto_risco": "crescimento tende a favorecer ativos cíclicos",
        "observacao": "Indicador coincidente do ciclo industrial.",
    },

    "HOUST": {
        "nome": "Housing Starts",
        "coluna": "Housing_Starts",
        "grupo": "Ciclo Econômico",
        "unidade": "milhares de unidades anualizadas",
        "natureza": "fluxo",
        "frequencia": "mensal",
        "maior_e_melhor": True,
        "interpretacao_alta": "Maior atividade de construção residencial.",
        "interpretacao_baixa": "Menor atividade de construção.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual",
        "impacto_risco": "aumento tende a indicar atividade econômica mais forte",
        "observacao": "Volátil; utilizar tendência e não observação isolada.",
    },

    "RSAFS": {
        "nome": "Retail Sales",
        "coluna": "Retail_Sales",
        "grupo": "Ciclo Econômico",
        "unidade": "USD milhões",
        "natureza": "fluxo",
        "frequencia": "mensal",
        "maior_e_melhor": True,
        "interpretacao_alta": "Maior consumo nominal.",
        "interpretacao_baixa": "Menor consumo nominal.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual",
        "impacto_risco": "crescimento favorece ativos cíclicos",
        "observacao": (
            "É nominal; deve ser interpretado em conjunto com inflação."
        ),
    },

    "UNRATE": {
        "nome": "Unemployment Rate",
        "coluna": "Unemployment_Rate",
        "grupo": "Mercado de Trabalho",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "mensal",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior desemprego e possível deterioração do mercado de trabalho.",
        "interpretacao_baixa": "Menor desemprego.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança em pontos percentuais",
        "impacto_risco": "aumento persistente pode sinalizar desaceleração",
        "observacao": (
            "É um indicador atrasado; combine com payrolls e jobless claims."
        ),
    },

    "PAYEMS": {
        "nome": "Nonfarm Payrolls",
        "coluna": "Nonfarm_Payrolls",
        "grupo": "Mercado de Trabalho",
        "unidade": "milhares",
        "natureza": "estoque/emprego",
        "frequencia": "mensal",
        "maior_e_melhor": True,
        "interpretacao_alta": "Maior nível de emprego.",
        "interpretacao_baixa": "Menor nível de emprego.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "crescimento mensal e média de 3 meses",
        "impacto_risco": "crescimento do emprego tende a sustentar consumo",
        "observacao": (
            "Mais útil avaliar média móvel de 3 meses do que uma única leitura."
        ),
    },

    "ICSA": {
        "nome": "Initial Jobless Claims",
        "coluna": "Initial_Jobless_Claims",
        "grupo": "Mercado de Trabalho",
        "unidade": "número de pedidos",
        "natureza": "fluxo",
        "frequencia": "semanal",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior número de novos pedidos de seguro-desemprego.",
        "interpretacao_baixa": "Menor número de pedidos.",
        "horizonte_principal": "1-3 meses",
        "comparacoes": ["30D", "90D"],
        "tipo_momentum": "média móvel e variação percentual",
        "impacto_risco": "aumento persistente sinaliza deterioração do trabalho",
        "observacao": (
            "Indicador relativamente rápido. Avaliar média móvel para reduzir ruído."
        ),
    },


    # ================================================================
    # CRÉDITO
    # ================================================================

    "BAMLH0A0HYM2": {
        "nome": "High Yield OAS",
        "coluna": "HY_Spread_Pct",
        "grupo": "Crédito",
        "unidade": "%",
        "natureza": "spread de crédito",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Maior prêmio de risco exigido para crédito High Yield; "
            "deterioração das condições de crédito."
        ),
        "interpretacao_baixa": (
            "Menor prêmio de risco; condições de crédito mais benignas."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação em pontos percentuais e z-score",
        "impacto_risco": "spreads subindo tendem a sinalizar risk-off",
        "observacao": (
            "Uma das séries mais importantes para detectar estresse "
            "financeiro."
        ),
    },

    "BAMLC0A0CM": {
        "nome": "Investment Grade OAS",
        "coluna": "IG_Spread_Pct",
        "grupo": "Crédito",
        "unidade": "%",
        "natureza": "spread de crédito",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior prêmio de risco no crédito Investment Grade.",
        "interpretacao_baixa": "Menor prêmio de risco.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação em pontos percentuais e z-score",
        "impacto_risco": "aumento persistente sinaliza deterioração financeira",
        "observacao": "Menos sensível que HY a deteriorações extremas.",
    },


    # ================================================================
    # CONDIÇÕES FINANCEIRAS
    # ================================================================

    "NFCI": {
        "nome": "Chicago Fed National Financial Conditions Index",
        "coluna": "Chicago_Financial_Conditions",
        "grupo": "Condições Financeiras",
        "unidade": "índice",
        "natureza": "índice composto",
        "frequencia": "semanal",
        "maior_e_melhor": False,
        "interpretacao_alta": (
            "Condições financeiras mais apertadas."
        ),
        "interpretacao_baixa": (
            "Condições financeiras mais frouxas."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "mudança absoluta",
        "impacto_risco": "aumento tende a ser negativo para ativos de risco",
        "observacao": (
            "Valores positivos representam condições mais apertadas "
            "que a média histórica do indicador."
        ),
    },


    # ================================================================
    # AÇÕES
    # ================================================================

    "SP500": {
        "nome": "S&P 500",
        "coluna": "SP500",
        "grupo": "Mercado de Ações",
        "unidade": "pontos",
        "natureza": "índice de preço",
        "frequencia": "diária",
        "maior_e_melhor": True,
        "interpretacao_alta": "Valorização das ações americanas de grande capitalização.",
        "interpretacao_baixa": "Desvalorização.",
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "retorno percentual",
        "impacto_risco": "alta representa maior apetite por risco",
        "observacao": (
            "Preço não deve ser interpretado isoladamente; combinar com "
            "juros reais, crédito, liquidez e valuation quando disponível."
        ),
    },

    "NASDAQCOM": {
        "nome": "NASDAQ Composite",
        "coluna": "NASDAQ",
        "grupo": "Mercado de Ações",
        "unidade": "pontos",
        "natureza": "índice de preço",
        "frequencia": "diária",
        "maior_e_melhor": True,
        "interpretacao_alta": "Valorização do mercado acionário de tecnologia/crescimento.",
        "interpretacao_baixa": "Desvalorização.",
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "retorno percentual",
        "impacto_risco": "sensível a liquidez e juros reais",
        "observacao": (
            "Possui duration elevada; juros reais são particularmente relevantes."
        ),
    },


    # ================================================================
    # VOLATILIDADE
    # ================================================================

    "VIXCLS": {
        "nome": "VIX",
        "coluna": "VIX",
        "grupo": "Volatilidade",
        "unidade": "índice",
        "natureza": "índice de volatilidade implícita",
        "frequencia": "diária",
        "maior_e_melhor": False,
        "interpretacao_alta": "Maior volatilidade implícita e maior demanda por proteção.",
        "interpretacao_baixa": "Menor volatilidade implícita.",
        "horizonte_principal": "1-30 dias",
        "comparacoes": ["5D", "30D", "90D"],
        "tipo_momentum": "variação percentual e nível absoluto",
        "impacto_risco": "aumento rápido tende a sinalizar risk-off",
        "observacao": (
            "Nível baixo não significa ausência de risco; observar mudanças rápidas."
        ),
    },


    # ================================================================
    # DÓLAR
    # ================================================================

    "DTWEXBGS": {
        "nome": "Broad Dollar Index",
        "coluna": "Dollar_Broad_Index",
        "grupo": "Dólar",
        "unidade": "índice",
        "natureza": "índice cambial",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Dólar mais forte frente a uma cesta ampla de moedas."
        ),
        "interpretacao_baixa": (
            "Dólar mais fraco."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "retorno percentual",
        "impacto_risco": (
            "Dólar forte pode apertar condições financeiras globais; "
            "efeito depende do contexto."
        ),
        "observacao": (
            "Particularmente importante para liquidez global e mercados emergentes."
        ),
    },

    "DEXUSEU": {
        "nome": "USD/EUR",
        "coluna": "Dollar_Euro",
        "grupo": "Dólar",
        "unidade": "USD por EUR",
        "natureza": "câmbio",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": "Euro mais forte frente ao dólar.",
        "interpretacao_baixa": "Dólar mais forte frente ao euro.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual",
        "impacto_risco": "efeito indireto sobre condições financeiras",
        "observacao": "Não confundir cotação com índice amplo do dólar.",
    },

    "DEXJPUS": {
        "nome": "USD/JPY",
        "coluna": "Dollar_Yen",
        "grupo": "Dólar",
        "unidade": "JPY por USD",
        "natureza": "câmbio",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": "Dólar mais forte / iene mais fraco.",
        "interpretacao_baixa": "Dólar mais fraco / iene mais forte.",
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual",
        "impacto_risco": "pode sinalizar mudanças em carry trade e liquidez global",
        "observacao": "Interpretar em conjunto com juros japoneses e americanos.",
    },


    # ================================================================
    # COMMODITIES
    # ================================================================

    "DCOILWTICO": {
        "nome": "WTI Crude Oil",
        "coluna": "WTI_Oil",
        "grupo": "Commodities",
        "unidade": "USD/barril",
        "natureza": "preço",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Maior preço do petróleo; pode aumentar inflação e favorecer "
            "produtores de energia."
        ),
        "interpretacao_baixa": (
            "Menor preço; pode aliviar inflação, mas também sinalizar "
            "menor demanda global."
        ),
        "horizonte_principal": "1-6 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "retorno percentual",
        "impacto_risco": "depende de oferta versus demanda",
        "observacao": (
            "Preço alto não é automaticamente bullish ou bearish; "
            "identificar o mecanismo."
        ),
    },

    "GOLDAMGBD228NLBM": {
        "nome": "Gold",
        "coluna": "Gold_USD",
        "grupo": "Commodities",
        "unidade": "USD/onça",
        "natureza": "preço",
        "frequencia": "diária",
        "maior_e_melhor": None,
        "interpretacao_alta": (
            "Pode refletir queda dos juros reais, busca por proteção, "
            "compras institucionais ou outros fatores."
        ),
        "interpretacao_baixa": (
            "Pode refletir aumento dos juros reais ou redução da demanda "
            "por proteção, entre outros fatores."
        ),
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "retorno percentual",
        "impacto_risco": (
            "Frequentemente sensível a juros reais, dólar e risco sistêmico."
        ),
        "observacao": (
            "Não assumir que ouro subindo significa necessariamente risk-off."
        ),
    },


    # ================================================================
    # CRIPTO
    # ================================================================

    "CBBTCUSD": {
        "nome": "Bitcoin",
        "coluna": "Bitcoin_USD",
        "grupo": "Cripto",
        "unidade": "USD",
        "natureza": "preço",
        "frequencia": "diária",
        "maior_e_melhor": True,
        "interpretacao_alta": "Valorização do Bitcoin.",
        "interpretacao_baixa": "Desvalorização.",
        "horizonte_principal": "1-12 meses",
        "comparacoes": ["7D", "30D", "90D", "365D"],
        "tipo_momentum": "retorno percentual e volatilidade",
        "impacto_risco": (
            "Sensível a liquidez, condições financeiras e apetite por risco."
        ),
        "observacao": (
            "Alta volatilidade; não interpretar movimentos de curto prazo "
            "como mudança macroestrutural sem confirmação."
        ),
    },
}

# =====================================================================
# 4. MAPA DERIVADO E REGRAS DE CÁLCULO
# =====================================================================

SERIES_MAP = {
    fred_id: metadata["coluna"]
    for fred_id, metadata in SERIES_METADATA.items()
}

# Regras quantitativas por natureza do indicador.
# Isso impede que índices, preços, taxas, spreads, fluxos e estoques sejam
# tratados com a mesma métrica.
CALC_RULES = {
    "Fed_Total_Assets_M": "level_change",
    "TGA_Balance_M": "level_change",
    "ON_RRP_B": "level_change",
    "Bank_Reserves_M": "level_change",
    "M2_B": "yoy_pct",
    "Net_Liquidity_B": "level_change",

    "Fed_Funds_Rate": "pp_change",
    "Effective_Fed_Funds_Rate": "pp_change",
    "SOFR": "pp_change",
    "Yield_3M": "pp_change",
    "Yield_2Y": "pp_change",
    "Yield_5Y": "pp_change",
    "Yield_10Y": "pp_change",
    "Yield_30Y": "pp_change",
    "Yield_Curve_10Y2Y": "pp_change",
    "Yield_Curve_10Y3M": "pp_change",
    "Real_Yield_5Y": "pp_change",
    "Real_Yield_10Y": "pp_change",
    "Real_Yield_30Y": "pp_change",
    "Breakeven_Inflation_5Y": "pp_change",
    "Breakeven_Inflation_10Y": "pp_change",

    "CPI": "inflation_index",
    "Core_CPI": "inflation_index",
    "PCE": "inflation_index",
    "Core_PCE": "inflation_index",
    "Real_GDP": "gdp",
    "Industrial_Production": "yoy_pct",
    "Housing_Starts": "yoy_pct",
    "Retail_Sales": "yoy_pct",
    "Unemployment_Rate": "pp_change",
    "Nonfarm_Payrolls": "payrolls",
    "Initial_Jobless_Claims": "claims",

    "HY_Spread_Pct": "spread",
    "IG_Spread_Pct": "spread",
    "Chicago_Financial_Conditions": "index_change",

    "SP500": "return_pct",
    "NASDAQ": "return_pct",
    "VIX": "vix",
    "Dollar_Broad_Index": "return_pct",
    "Dollar_Euro": "return_pct",
    "Dollar_Yen": "return_pct",
    "WTI_Oil": "return_pct",
    "Gold_USD": "return_pct",
    "Bitcoin_USD": "return_pct",
}


# =====================================================================
# 5. COLETA DOS DADOS DO FRED
# =====================================================================

def validate_environment():
    required = {
        "FRED_API_KEY": FRED_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("Variáveis de ambiente ausentes: " + ", ".join(missing))


def fetch_macro_data(api_key: str, lookback_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    if not api_key:
        raise ValueError("FRED_API_KEY não encontrada.")

    fred = Fred(api_key=api_key.strip())
    start_date = (
        datetime.today() - timedelta(days=365 * lookback_years)
    ).strftime("%Y-%m-%d")

    data = {}
    print("\nColetando séries temporais do FRED...\n")

    for series_id, col_name in SERIES_MAP.items():
        try:
            series = fred.get_series(
                series_id,
                observation_start=start_date,
            )

            if series is None or len(series) == 0:
                print(f"  [!] {series_id:<22} -> sem dados")
                continue

            series = pd.Series(series)
            series.index = pd.to_datetime(series.index)
            series = pd.to_numeric(series, errors="coerce").dropna()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            data[col_name] = series
            print(f"  [✓] {series_id:<22} -> {col_name}")

        except Exception as exc:
            print(f"  [✗] {series_id:<22} -> erro: {exc}")

    if not data:
        raise RuntimeError("Nenhuma série foi coletada do FRED.")

    # Cada série mantém sua frequência e suas datas reais.
    # Não usar resample/ffill aqui.
    return pd.DataFrame(data).sort_index()


# =====================================================================
# 6. FUNÇÕES QUANTITATIVAS
# =====================================================================

def latest_observation(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return pd.NaT, np.nan
    s = df[column].dropna()
    if s.empty:
        return pd.NaT, np.nan
    return s.index[-1], float(s.iloc[-1])


def series_asof(df: pd.DataFrame, column: str, target_date: pd.Timestamp) -> float:
    """Última observação real da série até a data-alvo, sem criar dados."""
    if column not in df.columns:
        return np.nan
    s = df[column].dropna()
    if s.empty:
        return np.nan
    values = s.loc[:target_date]
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def value_days_ago(df: pd.DataFrame, column: str, days: int):
    latest_date, _ = latest_observation(df, column)
    if pd.isna(latest_date):
        return pd.NaT, np.nan
    target = latest_date - pd.Timedelta(days=days)
    s = df[column].dropna()
    values = s.loc[:target]
    if values.empty:
        return pd.NaT, np.nan
    return values.index[-1], float(values.iloc[-1])


def absolute_change(current, previous):
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    return float(current - previous)


def percentage_change(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return float((current / previous - 1.0) * 100.0)


def calculate_rolling_zscore(series: pd.Series, window: int = 504) -> pd.Series:
    """Z-score usando observações reais, sem ffill artificial."""
    s = series.dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float)
    min_periods = min(60, len(s))
    mean = s.rolling(window=window, min_periods=min_periods).mean()
    std = s.rolling(window=window, min_periods=min_periods).std()
    return (s - mean) / std.replace(0, np.nan)


def calculate_percentile(series: pd.Series, value: float) -> float:
    s = series.dropna().astype(float)
    if s.empty or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100.0)


def moving_average(series: pd.Series, observations: int) -> float:
    s = series.dropna().astype(float)
    if s.empty:
        return np.nan
    return float(s.tail(observations).mean())


def calculate_liquidity_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy = Fed Total Assets - TGA - ON RRP.
    A referência temporal é a data de observação do WALCL.
    TGA e ON RRP usam a última observação real disponível até a data do WALCL.
    """
    required = ["Fed_Total_Assets_M", "TGA_Balance_M", "ON_RRP_B"]
    if not all(col in df.columns for col in required):
        return df

    walcl = df["Fed_Total_Assets_M"].dropna()
    records = []

    for date, fed_assets_m in walcl.items():
        tga_m = series_asof(df, "TGA_Balance_M", date)
        rrp_b = series_asof(df, "ON_RRP_B", date)
        if pd.isna(tga_m) or pd.isna(rrp_b):
            continue

        net_liquidity_b = (
            float(fed_assets_m) / 1000.0
            - float(tga_m) / 1000.0
            - float(rrp_b)
        )
        records.append((date, net_liquidity_b))

    if records:
        df = df.copy()
        df["Net_Liquidity_B"] = pd.Series(dict(records), dtype=float)

    return df


def process_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_liquidity_proxy(df.copy())

    # Z-score de crédito sobre observações originais.
    for source, target in [
        ("HY_Spread_Pct", "HY_Spread_ZScore"),
        ("IG_Spread_Pct", "IG_Spread_ZScore"),
    ]:
        if source in df.columns:
            df[target] = calculate_rolling_zscore(df[source])

    # Curvas calculadas independentemente das séries de spread do FRED.
    if "Yield_10Y" in df.columns and "Yield_2Y" in df.columns:
        y10 = df["Yield_10Y"].dropna()
        values = {}
        for date in y10.index:
            a = series_asof(df, "Yield_10Y", date)
            b = series_asof(df, "Yield_2Y", date)
            if not pd.isna(a) and not pd.isna(b):
                values[date] = a - b
        if values:
            df["Yield_Curve_10Y2Y_Calc"] = pd.Series(values, dtype=float)

    if "Yield_10Y" in df.columns and "Yield_3M" in df.columns:
        y10 = df["Yield_10Y"].dropna()
        values = {}
        for date in y10.index:
            a = series_asof(df, "Yield_10Y", date)
            b = series_asof(df, "Yield_3M", date)
            if not pd.isna(a) and not pd.isna(b):
                values[date] = a - b
        if values:
            df["Yield_Curve_10Y3M_Calc"] = pd.Series(values, dtype=float)

    # Payrolls: estoque de emprego -> fluxo mensal de vagas.
    if "Nonfarm_Payrolls" in df.columns:
        payroll = df["Nonfarm_Payrolls"].dropna()
        df["Payroll_MoM_Change"] = payroll.diff()

    # Claims: média móvel de 4 observações semanais.
    if "Initial_Jobless_Claims" in df.columns:
        claims = df["Initial_Jobless_Claims"].dropna()
        df["Claims_4W_MA"] = claims.rolling(4, min_periods=4).mean()

    return df


# =====================================================================
# 7. CÁLCULOS POR TIPO DE INDICADOR
# =====================================================================

def indicator_metrics(df: pd.DataFrame, column: str, rule: str) -> dict:
    latest_date, current = latest_observation(df, column)

    result = {
        "date": latest_date,
        "current": current,
        "rule": rule,
        "change_7d": np.nan,
        "change_30d": np.nan,
        "change_90d": np.nan,
        "change_365d": np.nan,
        "yoy": np.nan,
        "mom": np.nan,
        "ann_3m": np.nan,
        "ann_6m": np.nan,
        "qoq_annualized": np.nan,
        "zscore": np.nan,
        "percentile": np.nan,
        "ma20": np.nan,
        "max30": np.nan,
    }

    if pd.isna(latest_date) or pd.isna(current):
        return result

    if rule == "level_change":
        for days in [7, 30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            result[f"change_{days}d"] = absolute_change(current, previous)

    elif rule == "pp_change":
        for days in [7, 30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            result[f"change_{days}d"] = absolute_change(current, previous)

    elif rule == "return_pct":
        for days in [7, 30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            result[f"change_{days}d"] = percentage_change(current, previous)

    elif rule == "inflation_index":
        # Para séries mensais, use a quantidade de observações, não dias
        # corridos. Isso evita falhas em meses com 28/29/30/31 dias.
        s = df[column].dropna().astype(float)

        if len(s) >= 2:
            prev_1m = float(s.iloc[-2])
            result["mom"] = percentage_change(current, prev_1m)

        if len(s) >= 4:
            prev_3m = float(s.iloc[-4])
            if prev_3m > 0:
                result["ann_3m"] = ((current / prev_3m) ** 4 - 1.0) * 100.0

        if len(s) >= 7:
            prev_6m = float(s.iloc[-7])
            if prev_6m > 0:
                result["ann_6m"] = ((current / prev_6m) ** 2 - 1.0) * 100.0

        if len(s) >= 13:
            prev_12m = float(s.iloc[-13])
            result["yoy"] = percentage_change(current, prev_12m)

    elif rule == "gdp":
        # PIB trimestral: usar observações anteriores reais.
        s = df[column].dropna().astype(float)
        if len(s) >= 2:
            prev_q = float(s.iloc[-2])
            if prev_q > 0:
                result["qoq_annualized"] = ((current / prev_q) ** 4 - 1.0) * 100.0
        if len(s) >= 5:
            prev_y = float(s.iloc[-5])
            result["yoy"] = percentage_change(current, prev_y)

    elif rule == "yoy_pct":
        prev_y = series_asof(df, column, latest_date - pd.Timedelta(days=370))
        result["yoy"] = percentage_change(current, prev_y)

        for days in [30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            result[f"change_{days}d"] = percentage_change(current, previous)

    elif rule == "payrolls":
        # PAYEMS é estoque de emprego; sua diferença mensal representa o fluxo de vagas.
        stock = df[column].dropna().astype(float)
        flow = df.get("Payroll_MoM_Change", pd.Series(dtype=float)).dropna()
        result["current"] = current
        if not flow.empty:
            result["mom"] = float(flow.iloc[-1])
            result["ann_3m"] = moving_average(flow, 3)
            result["ann_6m"] = moving_average(flow, 6)
            result["date"] = stock.index[-1]

    elif rule == "claims":
        claims = df[column].dropna().astype(float)
        ma4 = df.get("Claims_4W_MA", pd.Series(dtype=float)).dropna()
        if not ma4.empty:
            result["current"] = float(ma4.iloc[-1])
            result["date"] = ma4.index[-1]
        # Aproximação YoY usando a observação semanal mais próxima de 52 semanas.
        if len(claims) >= 53:
            prev_y = float(claims.iloc[-53])
            result["yoy"] = percentage_change(float(claims.iloc[-1]), prev_y)

    elif rule == "spread":
        for days in [7, 30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            change = absolute_change(current, previous)
            result[f"change_{days}d"] = change * 100.0 if not pd.isna(change) else np.nan

        z_column = (
            "HY_Spread_ZScore"
            if column == "HY_Spread_Pct"
            else "IG_Spread_ZScore"
        )
        if z_column in df.columns:
            z = df[z_column].dropna()
            if not z.empty:
                result["zscore"] = float(z.iloc[-1])

        result["percentile"] = calculate_percentile(df[column], current)

    elif rule == "index_change":
        for days in [7, 30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            result[f"change_{days}d"] = absolute_change(current, previous)

    elif rule == "vix":
        for days in [7, 30, 90, 365]:
            _, previous = value_days_ago(df, column, days)
            result[f"change_{days}d"] = absolute_change(current, previous)

        s = df[column].dropna()
        result["ma20"] = moving_average(s, 20)
        recent = s.tail(30)
        if not recent.empty:
            result["max30"] = float(recent.max())
        result["percentile"] = calculate_percentile(s, current)

    return result


# =====================================================================
# 8. PAYLOAD QUANTITATIVO PARA O GEMINI
# =====================================================================

def fmt(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:,.{decimals}f}"


def fmt_date(value):
    if value is None or pd.isna(value):
        return "N/D"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def interpretation_direction(metadata: dict, metric: dict) -> str:
    if pd.isna(metric.get("current", np.nan)):
        return "EVIDÊNCIA INSUFICIENTE"

    rule = metric.get("rule")
    if rule == "inflation_index":
        change = metric.get("mom", np.nan)
    elif rule == "gdp":
        change = metric.get("qoq_annualized", np.nan)
    elif rule == "payrolls":
        change = metric.get("mom", np.nan)
    elif rule == "claims":
        change = metric.get("yoy", np.nan)
    else:
        change = metric.get("change_30d", np.nan)

    if pd.isna(change):
        return "SEM COMPARAÇÃO DISPONÍVEL"
    if change > 0:
        return metadata.get("interpretacao_alta", "Subindo")
    if change < 0:
        return metadata.get("interpretacao_baixa", "Caindo")
    return "Sem mudança relevante"


def build_indicator_line(df: pd.DataFrame, fred_id: str) -> str:
    metadata = SERIES_METADATA[fred_id]
    column = metadata["coluna"]
    rule = CALC_RULES.get(column, "level_change")
    metric = indicator_metrics(df, column, rule)

    unit = metadata["unidade"]
    decimals = 2
    if unit in ["USD milhões", "USD bilhões", "milhares", "número de pedidos"]:
        decimals = 0

    parts = [
        f"- {metadata['nome']}",
        f"obs={fmt_date(metric['date'])}",
        f"valor={fmt(metric['current'], decimals)} {unit}",
    ]

    if rule == "inflation_index":
        parts += [
            f"MoM={fmt(metric['mom'])}%",
            f"3M_an={fmt(metric['ann_3m'])}%",
            f"6M_an={fmt(metric['ann_6m'])}%",
            f"YoY={fmt(metric['yoy'])}%",
        ]
    elif rule == "gdp":
        parts += [
            f"QoQ_an={fmt(metric['qoq_annualized'])}%",
            f"YoY={fmt(metric['yoy'])}%",
        ]
    elif rule == "spread":
        parts += [
            f"Δ7D={fmt(metric['change_7d'])}bps",
            f"Δ30D={fmt(metric['change_30d'])}bps",
            f"Δ90D={fmt(metric['change_90d'])}bps",
            f"Δ1Y={fmt(metric['change_365d'])}bps",
            f"Z={fmt(metric['zscore'])}",
            f"percentil={fmt(metric['percentile'])}",
        ]
    elif rule == "payrolls":
        parts += [
            f"nível_emprego={fmt(metric['current'], 0)} mil",
            f"vagas_mês={fmt(metric['mom'], 0)} mil",
            f"média3M={fmt(metric['ann_3m'], 0)} mil",
            f"média6M={fmt(metric['ann_6m'], 0)} mil",
        ]
    elif rule == "claims":
        parts += [
            f"média4S={fmt(metric['current'], 0)}",
            f"YoY={fmt(metric['yoy'])}%",
        ]
    elif rule == "vix":
        parts += [
            f"Δ7D={fmt(metric['change_7d'])}",
            f"Δ30D={fmt(metric['change_30d'])}",
            f"média20D={fmt(metric['ma20'])}",
            f"máx30D={fmt(metric['max30'])}",
            f"percentil={fmt(metric['percentile'])}",
        ]
    elif rule == "return_pct":
        parts += [
            f"ret7D={fmt(metric['change_7d'])}%",
            f"ret30D={fmt(metric['change_30d'])}%",
            f"ret90D={fmt(metric['change_90d'])}%",
            f"ret1Y={fmt(metric['change_365d'])}%",
        ]
    elif rule == "pp_change":
        parts += [
            f"Δ7D={fmt(metric['change_7d'])}pp",
            f"Δ30D={fmt(metric['change_30d'])}pp",
            f"Δ90D={fmt(metric['change_90d'])}pp",
            f"Δ1Y={fmt(metric['change_365d'])}pp",
        ]
    else:
        parts += [
            f"Δ30D={fmt(metric['change_30d'])}",
            f"Δ90D={fmt(metric['change_90d'])}",
            f"Δ1Y={fmt(metric['change_365d'])}",
        ]

    return " | ".join(parts)


def build_liquidity_line(df: pd.DataFrame) -> str:
    if "Net_Liquidity_B" not in df.columns:
        return "- Liquidez líquida: N/D"

    s = df["Net_Liquidity_B"].dropna()
    if s.empty:
        return "- Liquidez líquida: N/D"

    current = float(s.iloc[-1])
    date = s.index[-1]
    p30 = series_asof(df, "Net_Liquidity_B", date - pd.Timedelta(days=30))
    p90 = series_asof(df, "Net_Liquidity_B", date - pd.Timedelta(days=90))
    p365 = series_asof(df, "Net_Liquidity_B", date - pd.Timedelta(days=365))

    return (
        f"- Liquidez Líquida: obs={fmt_date(date)} | valor={fmt(current)} B | "
        f"Δ30D={fmt(absolute_change(current, p30))} B | "
        f"Δ90D={fmt(absolute_change(current, p90))} B | "
        f"Δ1Y={fmt(absolute_change(current, p365))} B"
    )


def generate_agent_prompt_payload(df: pd.DataFrame) -> str:
    sections = []
    execution_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections += [
        "=" * 78,
        "DADOS MACROECONÔMICOS QUANTITATIVOS — FRED",
        f"Execução: {execution_date}",
        "As datas de observação são individuais para cada série.",
        "=" * 78,
    ]

    grouped = {}
    for fred_id, metadata in SERIES_METADATA.items():
        grouped.setdefault(metadata["grupo"], []).append(fred_id)

    group_order = [
        "Liquidez",
        "Política Monetária",
        "Curva de Juros",
        "Juros Reais",
        "Expectativas de Inflação",
        "Inflação",
        "Ciclo Econômico",
        "Mercado de Trabalho",
        "Crédito",
        "Condições Financeiras",
        "Mercado de Ações",
        "Volatilidade",
        "Dólar",
        "Commodities",
        "Cripto",
    ]

    for group in group_order:
        ids = grouped.get(group, [])
        if not ids:
            continue
        sections += [f"\n{group.upper()}", "-" * 78]
        for fred_id in ids:
            column = SERIES_METADATA[fred_id]["coluna"]
            if column in df.columns:
                sections.append(build_indicator_line(df, fred_id))

    if "Net_Liquidity_B" in df.columns:
        sections += [
            "\nLIQUIDEZ LÍQUIDA — PROXY CONSTRUÍDA",
            "-" * 78,
            "Proxy = Fed Total Assets - TGA - ON RRP.",
            "A proxy é calculada em datas do WALCL e não representa uma medida oficial única de liquidez de mercado.",
            build_liquidity_line(df),
        ]

    sections += [
        "\nREGRAS DE LEITURA",
        "-" * 78,
        "1. Não confundir estoque com fluxo.",
        "2. Não confundir nível com taxa de crescimento.",
        "3. Inflação: priorizar MoM, 3M anualizado, 6M anualizado e YoY.",
        "4. PIB: priorizar QoQ anualizado e YoY.",
        "5. Ativos: utilizar retorno percentual.",
        "6. Spreads: utilizar bps, z-score e percentil.",
        "7. Não inventar valuation, níveis críticos, probabilidades ou dados ausentes.",
        "8. Quando a evidência não for suficiente, declarar EVIDÊNCIA INSUFICIENTE.",
        "=" * 78,
    ]

    return "\n".join(sections)


# =====================================================================
# 9. SYSTEM INSTRUCTION — ANALISTA MACROESTRATÉGICO
# =====================================================================

SYSTEM_INSTRUCTION = r"""
Você é um Analista MacroEstratégico Sênior, especializado em macroeconomia,
ciclos econômicos, liquidez global e interação entre política monetária,
crédito, juros e preços dos ativos.

Seu objetivo é produzir uma análise macroeconômica orientada à tomada de
decisão de um investidor de perfil arrojado, mantendo disciplina quantitativa
e separando claramente DADO, CÁLCULO, INTERPRETAÇÃO e HIPÓTESE.

REGRAS ABSOLUTAS DE EVIDÊNCIA

1. Utilize somente os dados efetivamente fornecidos no payload.
2. Não invente valores, séries, eventos, valuation, P/E, margens, fluxos,
   posicionamento, probabilidades ou níveis críticos.
3. Não use conhecimento externo para preencher lacunas.
4. Se uma conclusão depender de uma variável ausente, escreva:
   "EVIDÊNCIA INSUFICIENTE PARA AVALIAR."
5. Sempre informe a data da observação quando disponível.
6. Diferencie dado observado, cálculo quantitativo, interpretação e hipótese.
7. Não transforme correlação temporal em causalidade.
8. Use "compatível com", "consistente com" ou "pode refletir" quando a
   causalidade não puder ser demonstrada.
9. Uma variável isolada nunca determina sozinha o regime macroeconômico.
10. Para uma mudança de regime, procure confirmação em pelo menos três
    blocos independentes: atividade, inflação e condições financeiras/mercados.
    Se isso não for possível, declare a incerteza.
11. Não invente thresholds. Se não houver nível quantitativo calculado no
    payload, descreva o gatilho em termos relativos, como aceleração,
    reversão, rompimento de tendência ou mudança de percentil.

============================================================
1. DIAGNÓSTICO DO REGIME MACRO
============================================================

Classifique entre:
- expansão + inflação controlada;
- expansão + inflação crescente;
- desaceleração + desinflação;
- estagflação;
- contração/recessão;
- transição entre regimes.

Apresente classificação, confiança ALTA/MÉDIA/BAIXA, evidências a favor,
contrapontos e o que poderia alterar o diagnóstico.

============================================================
2. POLÍTICA MONETÁRIA
============================================================

Diferencie política monetária observada da precificação do mercado.
Analise Fed Funds, Effective Fed Funds, SOFR, inflação, juros reais e curva.
Não afirme número de cortes, altas ou trajetória futura sem dados explícitos.

============================================================
3. LIQUIDEZ
============================================================

Analise separadamente:
- Fed Total Assets;
- TGA;
- ON RRP;
- reservas bancárias;
- M2;
- proxy de liquidez líquida;
- condições financeiras;
- dólar.

A proxy Fed Assets - TGA - ON RRP é uma construção analítica do programa.
Não a trate como medida oficial única de liquidez de mercado.

Queda da TGA pode ser consistente com liberação de caixa do Tesouro,
mas não prova, sozinha, o destino final dos recursos.

============================================================
4. CURVA DE JUROS
============================================================

Analise 3M, 2Y, 5Y, 10Y, 30Y, 10Y-2Y, 10Y-3M, juros reais e breakevens.

Identifique bull steepening, bull flattening, bear steepening ou bear flattening
somente quando os movimentos relativos dos vértices sustentarem a classificação.

Separe os possíveis drivers: política monetária, inflação, crescimento,
prêmio de prazo, risco fiscal e demanda por segurança.
Não atribua causa dominante sem evidência suficiente.

============================================================
5. INFLAÇÃO
============================================================

Analise CPI, Core CPI, PCE, Core PCE e breakevens.
Priorize a sequência MoM → 3M anualizado → 6M anualizado → YoY.
Nunca trate essas métricas como equivalentes.

============================================================
6. CICLO ECONÔMICO
============================================================

Analise PIB, produção industrial, housing e retail sales.
Diferencie indicadores antecedentes, coincidentes e atrasados.
Para PIB, priorize QoQ anualizado e YoY.

============================================================
7. MERCADO DE TRABALHO
============================================================

Analise desemprego, payrolls, média de 3 meses, média de 6 meses,
initial claims e média de 4 semanas.
Não trate o nível isolado do desemprego como principal indicador antecedente.

============================================================
8. CRÉDITO E CONDIÇÕES FINANCEIRAS
============================================================

Analise HY OAS e IG OAS por nível, bps, z-score, percentil, direção e velocidade.
Avalie se crédito confirma ou contradiz o ciclo econômico.

============================================================
9. CONFIRMAÇÃO PELOS MERCADOS
============================================================

Analise S&P 500, NASDAQ, VIX, dólar, ouro, petróleo e Bitcoin.
Para preços, utilize retornos percentuais.
Procure confirmação ou divergência com macro, liquidez, juros e crédito.

============================================================
10. DIVERGÊNCIAS MACRO × MERCADO
============================================================

Esta é uma seção prioritária.

Para cada divergência relevante:

DIVERGÊNCIA #[n]

MACRO:
[descrição]

INFLAÇÃO:
[descrição]

LIQUIDEZ:
[descrição]

CRÉDITO:
[descrição]

JUROS:
[descrição]

BOLSA:
[descrição]

VOLATILIDADE:
[descrição]

INTERPRETAÇÃO:
Explique a divergência sem presumir reversão.

HIPÓTESES EXPLICATIVAS:
Use somente mecanismos compatíveis com os dados.

O QUE CONFIRMARIA:
Indique movimentos observáveis. Não invente thresholds.

O QUE INVALIDARIA:
Indique movimentos observáveis que contrariariam a interpretação.

IMPLICAÇÃO:
Indique classes de ativos potencialmente mais sensíveis.

============================================================
11. CENÁRIOS
============================================================

Construa CENÁRIO-BASE, CENÁRIO ALTISTA e CENÁRIO BAIXISTA.

Para cada um:
- dinâmica macro;
- evidências;
- riscos;
- classes potencialmente favorecidas;
- gatilhos de confirmação;
- gatilhos de invalidação.

Não atribua probabilidades numéricas sem base quantitativa explícita.

============================================================
12. MATRIZ DE ATIVOS
============================================================

Analise Ações, Treasury/Renda Fixa, Crédito, Ouro, Commodities, Dólar e Cripto.

Formato:
Regime | Direção | Momentum | Risco | Assimetria | O que monitorar

Direção deve usar somente:
POSITIVO / NEUTRO / NEGATIVO

Não produza ranking geral entre classes.
Diferencie tese estrutural de tese tática.

============================================================
13. IMPLICAÇÕES TÁTICAS
============================================================

Para cada classe, apresente:
- fatores que favorecem exposição;
- fatores que ameaçam exposição;
- evidência que mudaria a tese;
- horizonte relevante;
- risco de excesso de exposição;
- risco de subexposição.

Não emita ordem automática de compra ou venda.
Formule implicações de forma condicional.

============================================================
14. ALERTAS DE MUDANÇA DE REGIME
============================================================

Liste os 5 indicadores que mais merecem monitoramento.
Para cada um, informe valor, data, direção, velocidade, importância,
classe mais sensível e condição observável que alteraria a tese.

============================================================
15. CONFIANÇA E INCERTEZA
============================================================

Apresente:

CONFIANÇA GERAL DO DIAGNÓSTICO: ALTA / MÉDIA / BAIXA

PRINCIPAIS EVIDÊNCIAS:
[até 3]

PRINCIPAIS CONTRAPONTOS:
[até 3]

PRINCIPAL INCERTEZA:
[uma frase]

DADO QUE MAIS PODERIA MUDAR O DIAGNÓSTICO:
[uma frase]

============================================================
16. RESUMO EXECUTIVO
============================================================

Comece sempre com:

REGIME MACRO:
[uma frase]

CONFIANÇA:
[alta / média / baixa]

LIQUIDEZ:
[expansiva / neutra / contracionista]

CRESCIMENTO:
[acelerando / estável / desacelerando]

INFLAÇÃO:
[acelerando / estável / desacelerando]

POLÍTICA MONETÁRIA:
[restritiva / neutra / expansionista]

CRÉDITO:
[melhorando / estável / piorando]

CURVA DE JUROS:
[uma frase]

PRINCIPAL ASSIMETRIA:
[uma frase]

PRINCIPAL RISCO:
[uma frase]

PRINCIPAL DIVERGÊNCIA:
[uma frase]

INDICADOR MAIS IMPORTANTE:
[indicador + motivo]

============================================================
17. DISCIPLINA FINAL
============================================================

Sempre que possível:

DADO
→ CÁLCULO
→ MECANISMO
→ IMPACTO MACRO
→ IMPACTO NOS ATIVOS
→ CONFIRMAÇÃO
→ INVALIDAÇÃO

Quando os dados não sustentarem uma conclusão, prefira:
"EVIDÊNCIA INSUFICIENTE PARA AVALIAR."
"""


# =====================================================================
# 10. ANÁLISE COM GEMINI
# =====================================================================

def is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_tokens = ["503", "429", "overloaded", "unavailable", "timeout", "deadline"]
    return isinstance(exc, ServerError) or any(token in text for token in retry_tokens)


def analisar_macro_com_gemini(dados_fred_text: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não encontrada.")

    api_key_limpa = (
        GEMINI_API_KEY.strip()
        .replace('"', '')
        .replace("'", "")
    )

    client = genai.Client(api_key=api_key_limpa)

    # Modelos estáveis atuais; o primeiro é a prioridade.
    modelos_candidatos = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]

    user_prompt = f"""
Realize a análise macroestratégica usando exclusivamente o payload abaixo.

Prioridades:
1. identificar mudança ou estabilidade de regime;
2. avaliar liquidez sem confundir proxy com medida oficial;
3. interpretar inflação nas diferentes janelas temporais;
4. avaliar crédito por nível, bps, z-score e percentil;
5. comparar macroeconomia com preços de mercado;
6. dar prioridade às divergências macro × mercado;
7. declarar confiança e incertezas.

Não use valores ausentes do payload.
Não invente thresholds.
Não invente valuation.
Não transforme correlação em causalidade.

PAYLOAD:

{dados_fred_text}
"""

    last_error = None

    for modelo in modelos_candidatos:
        for tentativa in range(1, 4):
            try:
                print(
                    f"\nGerando análise com {modelo} "
                    f"(tentativa {tentativa}/3)..."
                )

                response = client.models.generate_content(
                    model=modelo,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                    ),
                )

                if getattr(response, "text", None):
                    return response.text

                raise RuntimeError("Gemini retornou resposta sem texto.")

            except Exception as exc:
                last_error = exc
                print(f"  [!] Erro no modelo {modelo}: {exc}")

                if is_retryable_error(exc) and tentativa < 3:
                    wait_seconds = min(20, tentativa * 5)
                    print(f"  Aguardando {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue

                break

    raise RuntimeError(
        f"Todos os modelos e tentativas falharam. Último erro: {last_error}"
    )


# =====================================================================
# 11. ENVIO PARA TELEGRAM
# =====================================================================

def split_telegram_message(text: str, max_chars: int = TELEGRAM_MAX_CHAR):
    """Quebra preferencialmente em linhas e evita cortar palavras."""
    if len(text) <= max_chars:
        return [text]

    blocks = []
    remaining = text

    while len(remaining) > max_chars:
        cut = remaining.rfind("\n", 0, max_chars)
        if cut < max_chars * 0.60:
            cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars

        blocks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        blocks.append(remaining)

    return blocks


def enviar_relatorio_telegram_completo(
    dados_fred: str,
    analise_ia: str,
    token: str,
    chat_id: str,
):
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN não encontrado.")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID não encontrado.")

    url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"

    relatorio_completo = (
        "MACROESTRATÉGIA — RELATÓRIO IA\n"
        f"Execução: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        "=" * 60
        + "\n\n"
        + analise_ia.strip()
    )

    if SEND_RAW_PAYLOAD_TO_TELEGRAM:
        relatorio_completo += (
            "\n\n"
            + "=" * 60
            + "\nPAYLOAD QUANTITATIVO FRED\n"
            + "=" * 60
            + "\n\n"
            + dados_fred.strip()
        )

    blocos = split_telegram_message(relatorio_completo)
    print(f"\nEnviando relatório para Telegram ({len(blocos)} bloco(s))...")

    with requests.Session() as session:
        for idx, bloco in enumerate(blocos, start=1):
            try:
                response = session.post(
                    url,
                    json={
                        "chat_id": str(chat_id).strip(),
                        "text": bloco,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                if not result.get("ok"):
                    raise RuntimeError(
                        result.get("description", "Erro desconhecido no Telegram")
                    )
                print(f"  [✓] Bloco {idx}/{len(blocos)} enviado.")
            except Exception as exc:
                print(f"  [✗] Erro no bloco {idx}: {exc}")
                raise


def validate_email_environment():
    """Valida somente as credenciais necessárias para envio por e-mail."""
    missing = []
    if not EMAIL_SENDER:
        missing.append("EMAIL_SENDER")
    if not EMAIL_PASSWORD:
        missing.append("EMAIL_PASSWORD")
    if missing:
        raise ValueError(
            "Variáveis de e-mail ausentes: " + ", ".join(missing)
        )


def build_email_report(analise_ia: str) -> str:
    """Monta a versão completa do relatório para envio por e-mail."""
    return (
        "MACROESTRATÉGIA — RELATÓRIO IA\n"
        f"Execução: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        + "=" * 78
        + "\n\n"
        + analise_ia.strip()
    )


def enviar_relatorio_email(analise_ia: str):
    """Envia o relatório macro para o destinatário configurado."""
    validate_email_environment()

    msg = EmailMessage()
    msg["Subject"] = (
        "MacroEstratégia — Relatório Macro IA — "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECIPIENT
    msg.set_content(build_email_report(analise_ia))

    print(f"\nEnviando relatório para e-mail: {EMAIL_RECIPIENT}...")

    try:
        with smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=30) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("  [✓] E-mail enviado com sucesso.")

    except Exception as exc:
        print(f"  [✗] Erro ao enviar e-mail: {exc}")
        raise


# =====================================================================
# 13. EXECUÇÃO PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("ANALISTA MACROESTRATÉGICO — FRED + GEMINI")
    print("=" * 78)

    try:
        validate_environment()

        df_raw = fetch_macro_data(FRED_API_KEY)

        print(f"\nDados coletados: {len(df_raw.columns)} séries")
        print(
            "Período agregado das séries: "
            f"{df_raw.index.min().strftime('%Y-%m-%d')} "
            f"a {df_raw.index.max().strftime('%Y-%m-%d')}"
        )

        df_processed = process_macro_data(df_raw)
        relatorio_fred = generate_agent_prompt_payload(df_processed)
        analise_ia = analisar_macro_com_gemini(relatorio_fred)

        erros_entrega = []

        # Entrega pelo Telegram. Um erro aqui não impede o envio por e-mail.
        try:
            enviar_relatorio_telegram_completo(
                relatorio_fred,
                analise_ia,
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
            )
        except Exception as exc:
            erros_entrega.append(f"Telegram: {exc}")
            print(f"\n[!] Falha no envio ao Telegram; continuando para o e-mail: {exc}")

        # O e-mail recebe a análise IA completa, sem o payload técnico bruto.
        try:
            enviar_relatorio_email(analise_ia)
        except Exception as exc:
            erros_entrega.append(f"E-mail: {exc}")
            print(f"\n[!] Falha no envio por e-mail: {exc}")

        if erros_entrega:
            print("\nProcesso concluído com falhas de entrega:")
            for erro in erros_entrega:
                print(f"  - {erro}")
            sys.exit(1)

        print("\nProcesso concluído com sucesso em todos os canais.")

    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
        sys.exit(0)

    except Exception as exc:
        print(f"\nERRO FATAL: {exc}")
        sys.exit(1)
