import os
import sys
import time
import requests
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


# =====================================================================
# 2. CONFIGURAÇÕES
# =====================================================================

LOOKBACK_YEARS = 2

# Número máximo de caracteres por mensagem do Telegram.
# Mantido abaixo do limite oficial para permitir alguma margem.
TELEGRAM_MAX_CHAR = 3800


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

    "USSLIND": {
        "nome": "US Leading Index",
        "coluna": "Leading_Index",
        "grupo": "Ciclo Econômico",
        "unidade": "índice",
        "natureza": "indicador composto",
        "frequencia": "mensal",
        "maior_e_melhor": True,
        "interpretacao_alta": "Sinal de melhora nas perspectivas econômicas.",
        "interpretacao_baixa": "Sinal de deterioração das perspectivas.",
        "horizonte_principal": "3-12 meses",
        "comparacoes": ["30D", "90D", "365D"],
        "tipo_momentum": "variação percentual",
        "impacto_risco": "melhora tende a favorecer ativos cíclicos",
        "observacao": (
            "É um indicador antecedente; deve receber peso maior que "
            "indicadores econômicos atrasados quando houver divergência."
        ),
    },


    # ================================================================
    # MERCADO DE TRABALHO
    # ================================================================

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
# 4. COLETA DOS DADOS DO FRED
# =====================================================================

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
                observation_start=start_date
            )

            if series is not None and len(series) > 0:

                series = pd.Series(series)
                series.index = pd.to_datetime(series.index)
                series = pd.to_numeric(series, errors="coerce")
                series = series.dropna()

                data[col_name] = series

                print(
                    f"  [✓] {series_id:<20} -> {col_name}"
                )

            else:

                print(
                    f"  [!] {series_id:<20} -> sem dados"
                )

        except Exception as e:

            print(
                f"  [✗] {series_id:<20} -> erro: {e}"
            )

    if not data:
        raise RuntimeError("Nenhuma série foi coletada do FRED.")

    df = pd.DataFrame(data)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normaliza para frequência diária.
    # O forward fill permite comparar séries com frequências diferentes.
    df = df.resample("D").last().ffill()

    return df


# =====================================================================
# 5. FUNÇÕES AUXILIARES
# =====================================================================

def get_value_at_date(df: pd.DataFrame, column: str, days_ago: int):
    """
    Retorna o valor mais próximo disponível até determinada data.
    """

    if column not in df.columns:
        return np.nan

    target_date = df.index[-1] - pd.Timedelta(days=days_ago)

    series = df[column].dropna()

    if series.empty:
        return np.nan

    available = series.loc[:target_date]

    if available.empty:
        return np.nan

    return available.iloc[-1]


def calculate_zscore(series: pd.Series, window_days: int = 504):
    """
    Z-score utilizando aproximadamente 2 anos de dados diários.
    """

    series = series.dropna()

    if len(series) < 30:
        return pd.Series(index=series.index, dtype=float)

    rolling_mean = series.rolling(
        window=window_days,
        min_periods=30
    ).mean()

    rolling_std = series.rolling(
        window=window_days,
        min_periods=30
    ).std()

    zscore = (series - rolling_mean) / rolling_std

    return zscore


def format_number(value, decimals=2):

    if pd.isna(value):
        return "N/D"

    return f"{value:,.{decimals}f}"


def format_change(current, previous, decimals=2):

    if pd.isna(current) or pd.isna(previous):
        return "N/D"

    change = current - previous

    return f"{change:+,.{decimals}f}"


def determine_direction(current, previous):

    if pd.isna(current) or pd.isna(previous):
        return "N/D"

    if current > previous:
        return "SUBINDO"

    if current < previous:
        return "CAINDO"

    return "ESTÁVEL"


# =====================================================================
# 6. PROCESSAMENTO DOS INDICADORES
# =====================================================================

def process_macro_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # ---------------------------------------------------------------
    # LIQUIDEZ
    # ---------------------------------------------------------------

    if all(
        c in df.columns
        for c in [
            "Fed_Total_Assets_M",
            "TGA_Balance_M",
            "ON_RRP_B"
        ]
    ):

        fed_assets_b = df["Fed_Total_Assets_M"] / 1000.0
        tga_b = df["TGA_Balance_M"] / 1000.0
        rrp_b = df["ON_RRP_B"]

        df["Net_Liquidity_B"] = (
            fed_assets_b
            - tga_b
            - rrp_b
        )

        df["Net_Liquidity_30D_Change_B"] = (
            df["Net_Liquidity_B"]
            - df["Net_Liquidity_B"].shift(30)
        )

        df["Net_Liquidity_90D_Change_B"] = (
            df["Net_Liquidity_B"]
            - df["Net_Liquidity_B"].shift(90)
        )

    # ---------------------------------------------------------------
    # CRÉDITO — Z-SCORE
    # ---------------------------------------------------------------

    if "HY_Spread_Pct" in df.columns:

        df["HY_Spread_ZScore"] = calculate_zscore(
            df["HY_Spread_Pct"]
        )

    if "IG_Spread_Pct" in df.columns:

        df["IG_Spread_ZScore"] = calculate_zscore(
            df["IG_Spread_Pct"]
        )

    # ---------------------------------------------------------------
    # CURVA
    # ---------------------------------------------------------------

    if (
        "Yield_10Y" in df.columns
        and "Yield_2Y" in df.columns
    ):

        df["Yield_Curve_10Y2Y_Calc"] = (
            df["Yield_10Y"]
            - df["Yield_2Y"]
        )

    if (
        "Yield_10Y" in df.columns
        and "Yield_3M" in df.columns
    ):

        df["Yield_Curve_10Y3M_Calc"] = (
            df["Yield_10Y"]
            - df["Yield_3M"]
        )

    # ---------------------------------------------------------------
    # JUROS REAIS
    # ---------------------------------------------------------------

    if (
        "Yield_10Y" in df.columns
        and "Real_Yield_10Y" in df.columns
    ):

        df["Nominal_Real_Spread_10Y"] = (
            df["Yield_10Y"]
            - df["Real_Yield_10Y"]
        )

    return df


# =====================================================================
# 7. GERAÇÃO DO PAYLOAD PARA O GEMINI
# =====================================================================

def generate_agent_prompt_payload(df: pd.DataFrame) -> str:

    latest_date = df.index[-1]

    sections = []

    sections.append(
        "=" * 72
    )

    sections.append(
        f"DADOS MACROECONÔMICOS CONSOLIDADOS — FRED\n"
        f"Data de referência: {latest_date.strftime('%Y-%m-%d')}"
    )

    sections.append(
        "=" * 72
    )

    # ---------------------------------------------------------------
    # Função interna para montar blocos
    # ---------------------------------------------------------------

    def add_indicator(
        name,
        current_col,
        unit="",
        days=(30, 90, 365),
        decimals=2,
        zscore_col=None
    ):

        if current_col not in df.columns:
            return

        current = df[current_col].iloc[-1]

        values = []

        for d in days:
            previous = get_value_at_date(
                df,
                current_col,
                d
            )

            change = (
                current - previous
                if not pd.isna(previous)
                else np.nan
            )

            values.append(
                f"{d}D={format_change(change, 0, decimals)}"
            )

        direction_30 = determine_direction(
            current,
            get_value_at_date(
                df,
                current_col,
                30
            )
        )

        line = (
            f"- {name}: "
            f"{format_number(current, decimals)}{unit} | "
            f"{' | '.join(values)} | "
            f"Direção 30D: {direction_30}"
        )

        if zscore_col and zscore_col in df.columns:

            zscore = df[zscore_col].iloc[-1]

            if not pd.isna(zscore):

                line += (
                    f" | Z-score: {zscore:+.2f}"
                )

        sections.append(line)

    # ---------------------------------------------------------------
    # 1. LIQUIDEZ
    # ---------------------------------------------------------------

    sections.append("\n1. LIQUIDEZ DO SISTEMA FINANCEIRO")
    sections.append("-" * 72)

    add_indicator(
        "Fed Total Assets",
        "Fed_Total_Assets_M",
        " M",
        decimals=0
    )

    add_indicator(
        "TGA",
        "TGA_Balance_M",
        " M",
        decimals=0
    )

    add_indicator(
        "ON RRP",
        "ON_RRP_B",
        " B",
        decimals=2
    )

    add_indicator(
        "Reservas Bancárias",
        "Bank_Reserves_M",
        " M",
        decimals=0
    )

    add_indicator(
        "M2",
        "M2_B",
        " B",
        decimals=0
    )

    add_indicator(
        "Liquidez Líquida Fed - TGA - RRP",
        "Net_Liquidity_B",
        " B",
        decimals=2
    )

    add_indicator(
        "Variação Liquidez Líquida",
        "Net_Liquidity_30D_Change_B",
        " B",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 2. POLÍTICA MONETÁRIA
    # ---------------------------------------------------------------

    sections.append("\n2. POLÍTICA MONETÁRIA")
    sections.append("-" * 72)

    add_indicator(
        "Fed Funds",
        "Fed_Funds_Rate",
        "%",
        decimals=2
    )

    add_indicator(
        "Effective Fed Funds",
        "Effective_Fed_Funds_Rate",
        "%",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 3. CURVA DE JUROS
    # ---------------------------------------------------------------

    sections.append("\n3. CURVA DE JUROS")
    sections.append("-" * 72)

    add_indicator(
        "Treasury 3M",
        "Yield_3M",
        "%",
        decimals=2
    )

    add_indicator(
        "Treasury 2Y",
        "Yield_2Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Treasury 5Y",
        "Yield_5Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Treasury 10Y",
        "Yield_10Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Treasury 30Y",
        "Yield_30Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Curva 10Y-2Y",
        "Yield_Curve_10Y2Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Curva 10Y-3M",
        "Yield_Curve_10Y3M",
        "%",
        decimals=2
    )

    add_indicator(
        "Juro Real 5Y",
        "Real_Yield_5Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Juro Real 10Y",
        "Real_Yield_10Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Juro Real 30Y",
        "Real_Yield_30Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Breakeven 5Y",
        "Breakeven_Inflation_5Y",
        "%",
        decimals=2
    )

    add_indicator(
        "Breakeven 10Y",
        "Breakeven_Inflation_10Y",
        "%",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 4. INFLAÇÃO
    # ---------------------------------------------------------------

    sections.append("\n4. INFLAÇÃO")
    sections.append("-" * 72)

    add_indicator(
        "CPI",
        "CPI",
        "",
        decimals=2
    )

    add_indicator(
        "Core CPI",
        "Core_CPI",
        "",
        decimals=2
    )

    add_indicator(
        "PCE",
        "PCE",
        "",
        decimals=2
    )

    add_indicator(
        "Core PCE",
        "Core_PCE",
        "",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 5. CICLO ECONÔMICO
    # ---------------------------------------------------------------

    sections.append("\n5. CICLO ECONÔMICO")
    sections.append("-" * 72)

    add_indicator(
        "PIB Real",
        "Real_GDP",
        "",
        decimals=2
    )

    add_indicator(
        "Produção Industrial",
        "Industrial_Production",
        "",
        decimals=2
    )

    add_indicator(
        "Housing Starts",
        "Housing_Starts",
        "",
        decimals=0
    )

    add_indicator(
        "Retail Sales",
        "Retail_Sales",
        "",
        decimals=2
    )

    add_indicator(
        "Leading Index",
        "Leading_Index",
        "",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 6. MERCADO DE TRABALHO
    # ---------------------------------------------------------------

    sections.append("\n6. MERCADO DE TRABALHO")
    sections.append("-" * 72)

    add_indicator(
        "Desemprego",
        "Unemployment_Rate",
        "%",
        decimals=2
    )

    add_indicator(
        "Nonfarm Payrolls",
        "Nonfarm_Payrolls",
        " mil",
        decimals=0
    )

    add_indicator(
        "Initial Jobless Claims",
        "Initial_Jobless_Claims",
        "",
        decimals=0
    )

    # ---------------------------------------------------------------
    # 7. CRÉDITO
    # ---------------------------------------------------------------

    sections.append("\n7. CRÉDITO")
    sections.append("-" * 72)

    add_indicator(
        "High Yield Spread",
        "HY_Spread_Pct",
        "%",
        decimals=2,
        zscore_col="HY_Spread_ZScore"
    )

    add_indicator(
        "Investment Grade Spread",
        "IG_Spread_Pct",
        "%",
        decimals=2,
        zscore_col="IG_Spread_ZScore"
    )

    # ---------------------------------------------------------------
    # 8. CONDIÇÕES FINANCEIRAS
    # ---------------------------------------------------------------

    sections.append("\n8. CONDIÇÕES FINANCEIRAS")
    sections.append("-" * 72)

    add_indicator(
        "Chicago Fed NFCI",
        "Chicago_Financial_Conditions",
        "",
        decimals=3
    )

    # ---------------------------------------------------------------
    # 9. MERCADO DE AÇÕES
    # ---------------------------------------------------------------

    sections.append("\n9. MERCADOS DE AÇÕES")
    sections.append("-" * 72)

    add_indicator(
        "S&P 500",
        "SP500",
        "",
        decimals=2
    )

    add_indicator(
        "NASDAQ",
        "NASDAQ",
        "",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 10. VOLATILIDADE
    # ---------------------------------------------------------------

    sections.append("\n10. VOLATILIDADE")
    sections.append("-" * 72)

    add_indicator(
        "VIX",
        "VIX",
        "",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 11. DÓLAR
    # ---------------------------------------------------------------

    sections.append("\n11. DÓLAR")
    sections.append("-" * 72)

    add_indicator(
        "Dollar Broad Index",
        "Dollar_Broad_Index",
        "",
        decimals=2
    )

    add_indicator(
        "USD/EUR",
        "Dollar_Euro",
        "",
        decimals=4
    )

    add_indicator(
        "USD/JPY",
        "Dollar_Yen",
        "",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 12. COMMODITIES
    # ---------------------------------------------------------------

    sections.append("\n12. COMMODITIES")
    sections.append("-" * 72)

    add_indicator(
        "WTI",
        "WTI_Oil",
        " USD",
        decimals=2
    )

    add_indicator(
        "Ouro",
        "Gold_USD",
        " USD",
        decimals=2
    )

    # ---------------------------------------------------------------
    # 13. CRIPTO
    # ---------------------------------------------------------------

    sections.append("\n13. CRIPTO")
    sections.append("-" * 72)

    add_indicator(
        "Bitcoin",
        "Bitcoin_USD",
        " USD",
        decimals=2
    )

    sections.append("\n" + "=" * 72)

    sections.append(
        "INSTRUÇÃO: Utilize somente os dados efetivamente fornecidos acima. "
        "Não invente valores ausentes. Quando uma série estiver indisponível, "
        "indique N/D ou EVIDÊNCIA INSUFICIENTE."
    )

    sections.append("=" * 72)

    return "\n".join(sections)


# =====================================================================
# 8. SYSTEM INSTRUCTION — ANALISTA MACROESTRATÉGICO
# =====================================================================

SYSTEM_INSTRUCTION = """
Você é um Analista MacroEstratégico Sênior, especializado em macroeconomia,
ciclos econômicos, liquidez global e interação entre política monetária,
crédito, juros e preços dos ativos.

Seu objetivo é produzir uma análise macroeconômica orientada à tomada de
decisão de um investidor de perfil arrojado, evitando previsões categóricas
e separando claramente dados, interpretação e hipóteses.

IMPORTANTE:
- Utilize somente os dados efetivamente fornecidos no payload.
- Não invente valores, séries ou acontecimentos.
- Quando um dado necessário não estiver disponível, escreva "N/D" ou
  "EVIDÊNCIA INSUFICIENTE".
- Não trate uma única variável como determinante do cenário.
- Diferencie fato, interpretação e hipótese.
- Não faça previsões categóricas.
- Priorize mudanças de regime, momentum, divergências e assimetrias.

============================================================
1. PRINCÍPIOS DA ANÁLISE
============================================================

- Seja direto, técnico e objetivo.
- Priorize mudanças de regime, assimetrias e relações de causa e efeito.
- Analise nível, direção, velocidade de mudança e posição histórica.
- Diferencie indicadores leading, coincident e lagging.
- Sempre considere a data de referência dos dados.
- Dê maior peso a mudanças persistentes e confirmadas por diferentes
  classes de indicadores.
- Quando os indicadores forem conflitantes, destaque explicitamente
  a divergência.
- Não confunda correlação com causalidade.
- Não transforme uma divergência automaticamente em previsão de reversão.

============================================================
2. REGIME MACROECONÔMICO
============================================================

Determine qual regime melhor descreve o ambiente atual:

- expansão + inflação controlada;
- expansão + inflação crescente;
- desaceleração + desinflação;
- estagflação;
- contração/recessão;
- transição entre regimes.

Explique:

1. quais indicadores sustentam essa classificação;
2. quais indicadores a contradizem;
3. se o regime está se fortalecendo ou enfraquecendo;
4. quais sinais antecipariam uma mudança de regime.

Não classifique o regime com base em um único indicador.

============================================================
3. POLÍTICA MONETÁRIA
============================================================

Analise:

- Federal Funds;
- Effective Fed Funds;
- juros reais;
- inflação;
- expectativas de inflação;
- condições financeiras;
- relação entre crescimento e política monetária.

Diferencie:

POLÍTICA MONETÁRIA EFETIVA

de

POLÍTICA MONETÁRIA ESPERADA PELO MERCADO.

Identifique divergências entre dados econômicos, política monetária e
precificação dos ativos.

============================================================
4. LIQUIDEZ
============================================================

Analise:

- balanço do Fed;
- TGA;
- ON RRP;
- reservas bancárias;
- M2;
- liquidez líquida calculada;
- variação da liquidez;
- condições financeiras;
- dólar.

Não trate simplesmente o balanço do Fed como sinônimo de liquidez.

Determine se a liquidez está:

EXPANDINDO / NEUTRA / CONTRAINDO

Analise:

- velocidade da mudança;
- aceleração ou desaceleração;
- componentes responsáveis;
- possível impacto sobre ativos de risco.

Diferencie liquidez do Fed, liquidez bancária e liquidez percebida
pelos mercados de risco.

============================================================
5. CURVA DE JUROS
============================================================

Analise:

- 3M;
- 2Y;
- 5Y;
- 10Y;
- 30Y;
- 10Y-2Y;
- 10Y-3M;
- juros reais;
- breakevens.

Identifique se os movimentos são predominantemente explicados por:

- expectativa de política monetária;
- inflação;
- crescimento;
- prêmio de prazo;
- risco fiscal;
- demanda por ativos seguros.

Diferencie movimentos da parte curta e da parte longa da curva.

============================================================
6. CRÉDITO
============================================================

Analise:

- High Yield;
- Investment Grade;
- spreads;
- variações recentes;
- Z-score;
- condições financeiras.

Para os spreads, considere:

NÍVEL + DIREÇÃO + MOMENTUM + POSIÇÃO HISTÓRICA.

Dê atenção especial a mudanças rápidas nos spreads.

Avalie se o crédito está:

CONFIRMANDO ou CONTRADIZENDO o cenário macroeconômico.

============================================================
7. CICLO ECONÔMICO
============================================================

Analise:

- PIB;
- produção industrial;
- housing;
- retail sales;
- leading indicators;
- desemprego;
- payrolls;
- jobless claims.

Determine se o crescimento está:

ACELERANDO / ESTÁVEL / DESACELERANDO.

Diferencie indicadores antecedentes, contemporâneos e atrasados.

Dê atenção especial ao momentum.

============================================================
8. INFLAÇÃO
============================================================

Analise:

- CPI;
- Core CPI;
- PCE;
- Core PCE;
- breakevens;
- juros reais.

Diferencie:

PERSISTENTE × TRANSITÓRIA.

Avalie se a trajetória é compatível com política monetária:

RESTRITIVA / NEUTRA / EXPANSIONISTA.

============================================================
9. CONFIRMAÇÃO PELOS MERCADOS
============================================================

Compare o cenário macro com:

- S&P 500;
- NASDAQ;
- Treasury;
- dólar;
- ouro;
- petróleo;
- crédito;
- VIX;
- Bitcoin.

Procure principalmente divergências entre fundamentos macroeconômicos
e preços dos ativos.

Considere possíveis explicações:

- liquidez;
- posicionamento;
- expectativas futuras;
- valuation;
- prêmio de risco;
- fatores técnicos;
- fatores específicos do ativo.

============================================================
10. DIVERGÊNCIAS MACRO × MERCADO
============================================================

Esta é uma seção prioritária.

Identifique situações em que diferentes blocos de indicadores apresentam
sinais conflitantes.

Procure divergências entre:

- MACROECONOMIA;
- INFLAÇÃO;
- POLÍTICA MONETÁRIA;
- LIQUIDEZ;
- CRÉDITO;
- JUROS;
- BOLSA;
- VOLATILIDADE;
- DÓLAR;
- OURO/COMMODITIES;
- CRIPTO.

Para cada divergência relevante:

DIVERGÊNCIA #[n]

MACRO: [sinal]
INFLAÇÃO: [sinal]
POLÍTICA MONETÁRIA: [sinal]
LIQUIDEZ: [sinal]
CRÉDITO: [sinal]
JUROS: [sinal]
BOLSA: [sinal]
VOLATILIDADE: [sinal]

INTERPRETAÇÃO:
Explique objetivamente a divergência e possíveis mecanismos que a expliquem.

O QUE CONFIRMARIA A TESE:
Indique quais dados ou movimentos deveriam ocorrer.

O QUE INVALIDARIA A TESE:
Indique quais dados ou movimentos contrariariam a interpretação.

IMPLICAÇÃO:
Indique quais classes de ativos podem ser mais sensíveis à resolução
da divergência.

Não trate divergência como evidência automática de reversão.

Priorize divergências que:

1. estejam se ampliando;
2. apresentem grande diferença histórica;
3. envolvam diferentes tipos de indicadores;
4. possam sinalizar mudança de regime;
5. possam gerar movimentos relevantes entre classes de ativos.

============================================================
11. CENÁRIOS
============================================================

Construa:

CENÁRIO-BASE
CENÁRIO ALTISTA
CENÁRIO BAIXISTA

Para cada cenário, apresente:

- dinâmica macro;
- evidências;
- principais riscos;
- ativos potencialmente favorecidos;
- indicadores de confirmação;
- indicadores de invalidação.

Não atribua probabilidades numéricas sem base quantitativa suficiente.

============================================================
12. MATRIZ DE ATIVOS
============================================================

Analise:

- Ações;
- Treasury/Renda Fixa;
- Crédito;
- Ouro;
- Commodities;
- Dólar;
- Cripto.

Para cada classe:

Regime atual | Direção | Momentum | Risco | Assimetria | O que monitorar

Para Direção utilize:

POSITIVO / NEUTRO / NEGATIVO

Não produza ranking entre classes.

Diferencie:

TESE ESTRUTURAL

de

TESE TÁTICA.

============================================================
13. IMPLICAÇÕES PARA O INVESTIDOR
============================================================

Para cada classe:

1. O que favorece a exposição;
2. O que ameaça a exposição;
3. Qual evidência mudaria a tese;
4. Horizonte temporal relevante;
5. Principal risco de excesso de exposição;
6. Principal risco de subexposição.

Não transforme automaticamente uma condição macro favorável em
recomendação de compra ou uma condição desfavorável em recomendação
de venda.

Quando houver uma possível oportunidade tática, descreva:

"A assimetria favorece maior exposição caso X aconteça,
especialmente se Y confirmar."

Quando houver risco:

"O risco aumenta caso X aconteça e seja confirmado por Y."

============================================================
14. ALERTAS DE MUDANÇA DE REGIME
============================================================

Identifique os 5 indicadores que mais merecem monitoramento.

Para cada um:

- indicador;
- valor atual;
- data;
- direção;
- velocidade da mudança;
- nível crítico ou mudança relevante;
- por que importa;
- classe de ativo mais afetada;
- qual mudança alteraria a tese atual.

============================================================
15. QUALIDADE DOS DADOS
============================================================

Priorize dados oficiais e fontes primárias.

Diferencie:

- dado observado;
- estimativa;
- expectativa;
- interpretação.

Quando houver revisão relevante, destaque-a.

Se não houver dados suficientes:

"EVIDÊNCIA INSUFICIENTE."

Não preencha lacunas com suposições.

============================================================
16. RESUMO EXECUTIVO
============================================================

Comece sempre com:

REGIME MACRO:
[descrição em uma frase]

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
[principal característica]

PRINCIPAL ASSIMETRIA:
[uma frase]

PRINCIPAL RISCO:
[uma frase]

PRINCIPAL DIVERGÊNCIA:
[uma frase]

INDICADOR MAIS IMPORTANTE A MONITORAR:
[indicador + motivo]

============================================================
17. DISCIPLINA ANALÍTICA
============================================================

Sempre que possível, utilize a sequência:

DADO
→ MECANISMO
→ IMPACTO MACRO
→ IMPACTO NOS ATIVOS
→ O QUE CONFIRMARIA
→ O QUE INVALIDARIA

Destaque quando:

- o mercado estiver antecipando uma mudança ainda não visível
  nos dados econômicos;
- os dados econômicos estiverem mudando antes dos preços;
- houver mudança de nível sem mudança de tendência;
- houver mudança de tendência;
- houver divergência entre classes de ativos;
- a evidência estiver dividida.

O objetivo final é identificar:

MUDANÇAS DE REGIME
DIVERGÊNCIAS
ASSIMETRIAS
RISCOS DE CAUDA

produzindo uma análise útil para acompanhamento contínuo do ambiente
macroeconômico e planejamento tático de exposição entre classes de ativos.
"""


# =====================================================================
# 9. ANÁLISE COM GEMINI
# =====================================================================

def analisar_macro_com_gemini(dados_fred_text: str) -> str:

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não encontrada.")

    api_key_limpa = (
        GEMINI_API_KEY
        .strip()
        .replace('"', '')
        .replace("'", "")
    )

    client = genai.Client(
        api_key=api_key_limpa
    )

    modelos_candidatos = [
        "gemini-3.6-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    user_prompt = f"""
Realize a análise macroestratégica utilizando os dados abaixo.

Não utilize informações externas que não estejam presentes no payload.

DADOS:

{dados_fred_text}

Compare especialmente:

- situação atual;
- mudança em 30 dias;
- mudança em 90 dias;
- mudança em 12 meses;
- momentum;
- divergências;
- sinais de mudança de regime.

Dê prioridade à seção DIVERGÊNCIAS MACRO × MERCADO.

Ao final, produza o RESUMO EXECUTIVO e os 5 ALERTAS DE MUDANÇA DE REGIME
solicitados nas instruções do sistema.
"""

    for modelo in modelos_candidatos:

        for tentativa in range(1, 4):

            try:

                print(
                    f"\nGerando análise com {modelo} "
                    f"(Tentativa {tentativa}/3)..."
                )

                response = client.models.generate_content(

                    model=modelo,

                    contents=user_prompt,

                    config=types.GenerateContentConfig(

                        system_instruction=SYSTEM_INSTRUCTION,

                        temperature=0.2
                    )
                )

                if response.text:

                    return response.text

            except ServerError as e:

                print(
                    f"Servidor sobrecarregado (503) no modelo "
                    f"{modelo}: {e}"
                )

                if tentativa < 3:

                    tempo_espera = tentativa * 10

                    print(
                        f"Aguardando {tempo_espera}s..."
                    )

                    time.sleep(tempo_espera)

            except Exception as e:

                print(
                    f"Erro inesperado no modelo {modelo}: {e}"
                )

                break

    print(
        "\nTodos os modelos e tentativas esgotaram com erro."
    )

    sys.exit(1)


# =====================================================================
# 10. ENVIO PARA TELEGRAM
# =====================================================================

def enviar_relatorio_telegram_completo(
    dados_fred: str,
    analise_ia: str,
    token: str,
    chat_id: str
):

    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN não encontrado."
        )

    if not chat_id:
        raise ValueError(
            "TELEGRAM_CHAT_ID não encontrado."
        )

    url = (
        f"https://api.telegram.org/bot"
        f"{token.strip()}/sendMessage"
    )

    relatorio_completo = (
        f"{dados_fred}\n\n"
        f"{'=' * 60}\n"
        f"ANÁLISE MACROESTRATÉGICA & TÁTICA — IA\n"
        f"{'=' * 60}\n\n"
        f"{analise_ia}"
    )

    blocos = [
        relatorio_completo[
            i:i + TELEGRAM_MAX_CHAR
        ]
        for i in range(
            0,
            len(relatorio_completo),
            TELEGRAM_MAX_CHAR
        )
    ]

    print(
        f"\nEnviando relatório completo "
        f"({len(blocos)} bloco(s)) para o Telegram..."
    )

    for idx, bloco in enumerate(blocos):

        payload = {
            "chat_id": str(chat_id).strip(),
            "text": bloco
        }

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=30
            )

            res_data = response.json()

            if response.status_code == 200 and res_data.get("ok"):

                print(
                    f"  [✓] Bloco "
                    f"{idx + 1}/{len(blocos)} enviado."
                )

            else:

                print(
                    f"  [✗] Erro Telegram: "
                    f"{res_data.get('description')}"
                )

                sys.exit(1)

        except Exception as e:

            print(
                f"  [✗] Erro de conexão com Telegram: {e}"
            )

            sys.exit(1)


# =====================================================================
# 11. EXECUÇÃO PRINCIPAL
# =====================================================================

if __name__ == "__main__":

    print("\n" + "=" * 72)
    print("     ANALISTA MACROESTRATÉGICO — FRED + GEMINI")
    print("=" * 72)

    try:

        # -------------------------------------------------------------
        # ETAPA 1 — FRED
        # -------------------------------------------------------------

        df_raw = fetch_macro_data(
            FRED_API_KEY
        )

        print(
            f"\nDados coletados: "
            f"{len(df_raw.columns)} séries"
        )

        print(
            f"Período: "
            f"{df_raw.index.min().strftime('%Y-%m-%d')} "
            f"até "
            f"{df_raw.index.max().strftime('%Y-%m-%d')}"
        )

        # -------------------------------------------------------------
        # ETAPA 2 — PROCESSAMENTO
        # -------------------------------------------------------------

        df_processed = process_macro_data(
            df_raw
        )

        # -------------------------------------------------------------
        # ETAPA 3 — PAYLOAD
        # -------------------------------------------------------------

        relatorio_fred = generate_agent_prompt_payload(
            df_processed
        )

        # -------------------------------------------------------------
        # ETAPA 4 — GEMINI
        # -------------------------------------------------------------

        analise_ia = analisar_macro_com_gemini(
            relatorio_fred
        )

        # -------------------------------------------------------------
        # ETAPA 5 — TELEGRAM
        # -------------------------------------------------------------

        enviar_relatorio_telegram_completo(
            relatorio_fred,
            analise_ia,
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID
        )

        print("\nProcesso concluído com sucesso.")

    except KeyboardInterrupt:

        print("\nExecução interrompida pelo usuário.")
        sys.exit(0)

    except Exception as e:

        print(
            f"\nERRO FATAL: {e}"
        )

        sys.exit(1)
