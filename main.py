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
# 1. VARIÁVEIS DE AMBIENTE
# =====================================================================

FRED_API_KEY = os.environ.get("FRED_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# =====================================================================
# 2. CONFIGURAÇÕES
# =====================================================================

LOOKBACK_YEARS = 3
TELEGRAM_MAX_CHAR = 3800
REQUEST_TIMEOUT = 30

# Modelos estáveis/atuais, em ordem de prioridade.
# Gemini 3.8 Flash é atualmente o principal modelo Flash estável.
MODELOS_CANDIDATOS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]


# =====================================================================
# 3. METADADOS DAS SÉRIES DO FRED
# =====================================================================

SERIES_METADATA = {

    # ---------------------------------------------------------------
    # LIQUIDEZ
    # ---------------------------------------------------------------
    "WALCL": {
        "nome": "Fed Total Assets",
        "coluna": "Fed_Total_Assets_M",
        "grupo": "Liquidez",
        "unidade": "USD milhões",
        "natureza": "estoque",
        "frequencia": "semanal",
        "calc_tipo": "change_abs",
        "comparacoes": [30, 90, 365],
        "observacao": "Não confundir tamanho do balanço com liquidez disponível para ativos de risco.",
    },
    "WTREGEN": {
        "nome": "Treasury General Account (TGA)",
        "coluna": "TGA_Balance_M",
        "grupo": "Liquidez",
        "unidade": "USD milhões",
        "natureza": "estoque",
        "frequencia": "semanal",
        "calc_tipo": "change_abs",
        "comparacoes": [30, 90, 365],
        "observacao": "Aumento do TGA, em igualdade de condições, tende a retirar liquidez do sistema.",
    },
    "RRPONTSYD": {
        "nome": "ON RRP",
        "coluna": "ON_RRP_B",
        "grupo": "Liquidez",
        "unidade": "USD bilhões",
        "natureza": "estoque",
        "frequencia": "diária",
        "calc_tipo": "change_abs",
        "comparacoes": [30, 90],
        "observacao": "A queda do ON RRP pode liberar recursos para o sistema enquanto houver saldo relevante.",
    },
    "WRESBAL": {
        "nome": "Bank Reserves",
        "coluna": "Bank_Reserves_M",
        "grupo": "Liquidez",
        "unidade": "USD milhões",
        "natureza": "estoque",
        "frequencia": "semanal",
        "calc_tipo": "change_abs",
        "comparacoes": [30, 90, 365],
        "observacao": "Reservas maiores tendem a indicar maior folga de liquidez bancária, mas não garantem expansão do crédito.",
    },
    "M2SL": {
        "nome": "M2",
        "coluna": "M2_B",
        "grupo": "Liquidez",
        "unidade": "USD bilhões",
        "natureza": "estoque monetário",
        "frequencia": "mensal",
        "calc_tipo": "yoy_pct",
        "comparacoes": [90, 365],
        "observacao": "Relação com preços de ativos varia conforme o regime macroeconômico.",
    },

    # ---------------------------------------------------------------
    # POLÍTICA MONETÁRIA
    # ---------------------------------------------------------------
    "FEDFUNDS": {
        "nome": "Federal Funds Rate",
        "coluna": "Fed_Funds_Rate",
        "grupo": "Política Monetária",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "mensal",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Interpretar junto com inflação, crescimento e condições financeiras.",
    },
    "EFFR": {
        "nome": "Effective Federal Funds Rate",
        "coluna": "Effective_Fed_Funds_Rate",
        "grupo": "Política Monetária",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90],
        "observacao": "Taxa efetiva diária do mercado de Fed Funds.",
    },
    "SOFR": {
        "nome": "Secured Overnight Financing Rate (SOFR)",
        "coluna": "SOFR",
        "grupo": "Política Monetária / Funding",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [5, 30, 90],
        "observacao": "Referência importante para condições de funding de curto prazo.",
    },

    # ---------------------------------------------------------------
    # CURVA DE JUROS
    # ---------------------------------------------------------------
    "DGS3MO": {
        "nome": "Treasury 3M",
        "coluna": "Yield_3M",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Parte curta da curva; muito ligada à trajetória da política monetária.",
    },
    "DGS2": {
        "nome": "Treasury 2Y",
        "coluna": "Yield_2Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Sensível às expectativas de política monetária.",
    },
    "DGS5": {
        "nome": "Treasury 5Y",
        "coluna": "Yield_5Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Parte intermediária da curva.",
    },
    "DGS10": {
        "nome": "Treasury 10Y",
        "coluna": "Yield_10Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Interpretar junto com juros reais e breakevens.",
    },
    "DGS30": {
        "nome": "Treasury 30Y",
        "coluna": "Yield_30Y",
        "grupo": "Curva de Juros",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Muito sensível a prêmio de prazo, inflação e risco fiscal.",
    },
    "T10Y2Y": {
        "nome": "Curva 10Y-2Y",
        "coluna": "Yield_Curve_10Y2Y",
        "grupo": "Curva de Juros",
        "unidade": "pontos percentuais",
        "natureza": "spread de taxas",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Não interpretar inclinação positiva ou negativa como sinal automático de alta ou queda dos ativos.",
    },
    "T10Y3M": {
        "nome": "Curva 10Y-3M",
        "coluna": "Yield_Curve_10Y3M",
        "grupo": "Curva de Juros",
        "unidade": "pontos percentuais",
        "natureza": "spread de taxas",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Interpretar em conjunto com crescimento e condições financeiras.",
    },

    # ---------------------------------------------------------------
    # JUROS REAIS
    # ---------------------------------------------------------------
    "DFII5": {
        "nome": "Real Yield 5Y",
        "coluna": "Real_Yield_5Y",
        "grupo": "Juros Reais",
        "unidade": "%",
        "natureza": "taxa real",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Relevante para ativos de duration.",
    },
    "DFII10": {
        "nome": "Real Yield 10Y",
        "coluna": "Real_Yield_10Y",
        "grupo": "Juros Reais",
        "unidade": "%",
        "natureza": "taxa real",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Uma das variáveis mais relevantes para valuation e ouro.",
    },
    "DFII30": {
        "nome": "Real Yield 30Y",
        "coluna": "Real_Yield_30Y",
        "grupo": "Juros Reais",
        "unidade": "%",
        "natureza": "taxa real",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Importante para duration longa.",
    },

    # ---------------------------------------------------------------
    # EXPECTATIVAS DE INFLAÇÃO
    # ---------------------------------------------------------------
    "T5YIE": {
        "nome": "Breakeven Inflation 5Y",
        "coluna": "Breakeven_Inflation_5Y",
        "grupo": "Expectativas de Inflação",
        "unidade": "%",
        "natureza": "expectativa implícita",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Breakeven contém outros componentes além da expectativa pura de inflação.",
    },
    "T10YIE": {
        "nome": "Breakeven Inflation 10Y",
        "coluna": "Breakeven_Inflation_10Y",
        "grupo": "Expectativas de Inflação",
        "unidade": "%",
        "natureza": "expectativa implícita",
        "frequencia": "diária",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Interpretar conjuntamente com juros reais e Treasury 10Y.",
    },

    # ---------------------------------------------------------------
    # INFLAÇÃO
    # ---------------------------------------------------------------
    "CPIAUCSL": {
        "nome": "CPI",
        "coluna": "CPI",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "calc_tipo": "inflation_index",
        "comparacoes": [30, 90, 365],
        "observacao": "O nível do índice não é a taxa de inflação; calcular MoM e YoY.",
    },
    "CPILFESL": {
        "nome": "Core CPI",
        "coluna": "Core_CPI",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "calc_tipo": "inflation_index",
        "comparacoes": [30, 90, 365],
        "observacao": "Dar atenção especial ao momentum da inflação subjacente.",
    },
    "PCEPI": {
        "nome": "PCE",
        "coluna": "PCE",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "calc_tipo": "inflation_index",
        "comparacoes": [30, 90, 365],
        "observacao": "Usado pelo Fed como medida de inflação ampla.",
    },
    "PCEPILFE": {
        "nome": "Core PCE",
        "coluna": "Core_PCE",
        "grupo": "Inflação",
        "unidade": "índice",
        "natureza": "índice de preços",
        "frequencia": "mensal",
        "calc_tipo": "inflation_index",
        "comparacoes": [30, 90, 365],
        "observacao": "Medida especialmente relevante para a função de reação do Fed.",
    },

    # ---------------------------------------------------------------
    # ATIVIDADE ECONÔMICA
    # ---------------------------------------------------------------
    "GDPC1": {
        "nome": "Real GDP",
        "coluna": "Real_GDP",
        "grupo": "Ciclo Econômico",
        "unidade": "USD bilhões encadeados",
        "natureza": "fluxo",
        "frequencia": "trimestral",
        "calc_tipo": "gdp_growth",
        "comparacoes": [90, 365],
        "observacao": "Indicador atrasado; dar peso maior a indicadores antecedentes e coincidentes para mudanças recentes.",
    },
    "INDPRO": {
        "nome": "Industrial Production",
        "coluna": "Industrial_Production",
        "grupo": "Ciclo Econômico",
        "unidade": "índice",
        "natureza": "índice de atividade",
        "frequencia": "mensal",
        "calc_tipo": "yoy_pct",
        "comparacoes": [30, 90, 365],
        "observacao": "Indicador coincidente do setor industrial.",
    },
    "HOUST": {
        "nome": "Housing Starts",
        "coluna": "Housing_Starts",
        "grupo": "Ciclo Econômico",
        "unidade": "milhares de unidades anualizadas",
        "natureza": "fluxo",
        "frequencia": "mensal",
        "calc_tipo": "yoy_pct",
        "comparacoes": [30, 90, 365],
        "observacao": "Série volátil; avaliar tendência e não uma única leitura.",
    },
    "RSAFS": {
        "nome": "Retail Sales",
        "coluna": "Retail_Sales",
        "grupo": "Ciclo Econômico",
        "unidade": "USD milhões",
        "natureza": "fluxo nominal",
        "frequencia": "mensal",
        "calc_tipo": "yoy_pct",
        "comparacoes": [30, 90, 365],
        "observacao": "É nominal; deve ser interpretado junto com inflação.",
    },

    # ---------------------------------------------------------------
    # MERCADO DE TRABALHO
    # ---------------------------------------------------------------
    "UNRATE": {
        "nome": "Unemployment Rate",
        "coluna": "Unemployment_Rate",
        "grupo": "Mercado de Trabalho",
        "unidade": "%",
        "natureza": "taxa",
        "frequencia": "mensal",
        "calc_tipo": "change_pp",
        "comparacoes": [30, 90, 365],
        "observacao": "Indicador atrasado; combinar com payrolls e jobless claims.",
    },
    "PAYEMS": {
        "nome": "Nonfarm Payrolls",
        "coluna": "Nonfarm_Payrolls",
        "grupo": "Mercado de Trabalho",
        "unidade": "milhares",
        "natureza": "estoque/emprego",
        "frequencia": "mensal",
        "calc_tipo": "payrolls",
        "comparacoes": [30, 90, 365],
        "observacao": "Dar preferência à média de 3 meses e à variação mensal.",
    },
    "ICSA": {
        "nome": "Initial Jobless Claims",
        "coluna": "Initial_Jobless_Claims",
        "grupo": "Mercado de Trabalho",
        "unidade": "pedidos",
        "natureza": "fluxo",
        "frequencia": "semanal",
        "calc_tipo": "claims",
        "comparacoes": [30, 90],
        "observacao": "Utilizar média móvel de 4 semanas para reduzir ruído.",
    },

    # ---------------------------------------------------------------
    # CRÉDITO
    # ---------------------------------------------------------------
    "BAMLH0A0HYM2": {
        "nome": "High Yield OAS",
        "coluna": "HY_Spread_Pct",
        "grupo": "Crédito",
        "unidade": "%",
        "natureza": "spread de crédito",
        "frequencia": "diária",
        "calc_tipo": "spread",
        "comparacoes": [5, 30, 90, 365],
        "observacao": "Aumento rápido indica deterioração das condições de crédito.",
    },
    "BAMLC0A0CM": {
        "nome": "Investment Grade OAS",
        "coluna": "IG_Spread_Pct",
        "grupo": "Crédito",
        "unidade": "%",
        "natureza": "spread de crédito",
        "frequencia": "diária",
        "calc_tipo": "spread",
        "comparacoes": [5, 30, 90, 365],
        "observacao": "Utilizar junto com HY para detectar deterioração ampla ou concentrada.",
    },

    # ---------------------------------------------------------------
    # CONDIÇÕES FINANCEIRAS
    # ---------------------------------------------------------------
    "NFCI": {
        "nome": "Chicago Fed NFCI",
        "coluna": "Chicago_Financial_Conditions",
        "grupo": "Condições Financeiras",
        "unidade": "índice",
        "natureza": "índice composto",
        "frequencia": "semanal",
        "calc_tipo": "change_index",
        "comparacoes": [30, 90, 365],
        "observacao": "Valores positivos significam condições mais apertadas que a média histórica.",
    },
    "NFCICREDIT": {
        "nome": "NFCI Credit Subindex",
        "coluna": "NFCI_Credit",
        "grupo": "Condições Financeiras",
        "unidade": "índice",
        "natureza": "índice composto",
        "frequencia": "semanal",
        "calc_tipo": "change_index",
        "comparacoes": [30, 90, 365],
        "observacao": "Ajuda a separar condições de crédito do restante das condições financeiras.",
    },
    "NFCIRISK": {
        "nome": "NFCI Risk Subindex",
        "coluna": "NFCI_Risk",
        "grupo": "Condições Financeiras",
        "unidade": "índice",
        "natureza": "índice composto",
        "frequencia": "semanal",
        "calc_tipo": "change_index",
        "comparacoes": [30, 90, 365],
        "observacao": "Ajuda a identificar mudanças no componente de risco das condições financeiras.",
    },

    # ---------------------------------------------------------------
    # AÇÕES
    # ---------------------------------------------------------------
    "SP500": {
        "nome": "S&P 500",
        "coluna": "SP500",
        "grupo": "Mercado de Ações",
        "unidade": "pontos",
        "natureza": "índice de preço",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [5, 30, 90, 365],
        "observacao": "Combinar com juros reais, crédito, liquidez e valuation quando disponíveis.",
    },
    "NASDAQCOM": {
        "nome": "NASDAQ Composite",
        "coluna": "NASDAQ",
        "grupo": "Mercado de Ações",
        "unidade": "pontos",
        "natureza": "índice de preço",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [5, 30, 90, 365],
        "observacao": "Particularmente sensível a juros reais e liquidez.",
    },

    # ---------------------------------------------------------------
    # VOLATILIDADE
    # ---------------------------------------------------------------
    "VIXCLS": {
        "nome": "VIX",
        "coluna": "VIX",
        "grupo": "Volatilidade",
        "unidade": "índice",
        "natureza": "volatilidade implícita",
        "frequencia": "diária",
        "calc_tipo": "vix",
        "comparacoes": [5, 30, 90],
        "observacao": "Aceleração do VIX é mais informativa que seu nível isolado.",
    },

    # ---------------------------------------------------------------
    # DÓLAR
    # ---------------------------------------------------------------
    "DTWEXBGS": {
        "nome": "Broad Dollar Index",
        "coluna": "Dollar_Broad_Index",
        "grupo": "Dólar",
        "unidade": "índice",
        "natureza": "índice cambial",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [5, 30, 90, 365],
        "observacao": "Dólar forte pode apertar condições financeiras globais.",
    },
    "DEXUSEU": {
        "nome": "USD/EUR",
        "coluna": "Dollar_Euro",
        "grupo": "Dólar",
        "unidade": "USD por EUR",
        "natureza": "câmbio",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [30, 90, 365],
        "observacao": "Usar como informação complementar ao índice amplo do dólar.",
    },
    "DEXJPUS": {
        "nome": "USD/JPY",
        "coluna": "Dollar_Yen",
        "grupo": "Dólar",
        "unidade": "JPY por USD",
        "natureza": "câmbio",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [30, 90, 365],
        "observacao": "Pode ajudar a identificar mudanças no carry e na liquidez global.",
    },

    # ---------------------------------------------------------------
    # COMMODITIES
    # ---------------------------------------------------------------
    "DCOILWTICO": {
        "nome": "WTI Crude Oil",
        "coluna": "WTI_Oil",
        "grupo": "Commodities",
        "unidade": "USD/barril",
        "natureza": "preço",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [30, 90, 365],
        "observacao": "Alta pode refletir oferta restrita ou demanda forte; o mecanismo importa.",
    },
    "GOLDAMGBD228NLBM": {
        "nome": "Gold",
        "coluna": "Gold_USD",
        "grupo": "Commodities",
        "unidade": "USD/onça",
        "natureza": "preço",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [30, 90, 365],
        "observacao": "Interpretar conjuntamente com juros reais, dólar e risco sistêmico.",
    },

    # ---------------------------------------------------------------
    # CRIPTO
    # ---------------------------------------------------------------
    "CBBTCUSD": {
        "nome": "Bitcoin",
        "coluna": "Bitcoin_USD",
        "grupo": "Cripto",
        "unidade": "USD",
        "natureza": "preço",
        "frequencia": "diária",
        "calc_tipo": "price_return",
        "comparacoes": [7, 30, 90, 365],
        "observacao": "Alta volatilidade; não inferir mudança macroestrutural de movimentos curtos isolados.",
    },
}


SERIES_MAP = {
    fred_id: metadata["coluna"]
    for fred_id, metadata in SERIES_METADATA.items()
}


# =====================================================================
# 4. FUNÇÕES DE COLETA
# =====================================================================

def fetch_macro_data(api_key: str, lookback_years: int = LOOKBACK_YEARS):
    """Coleta as séries sem forçar uma frequência comum."""

    if not api_key:
        raise ValueError("FRED_API_KEY não encontrada.")

    fred = Fred(api_key=api_key.strip())

    start_date = (
        datetime.today() - timedelta(days=365 * lookback_years)
    ).strftime("%Y-%m-%d")

    series_data = {}

    print("\nColetando séries temporais do FRED...\n")

    for series_id, col_name in SERIES_MAP.items():

        try:
            series = fred.get_series(
                series_id,
                observation_start=start_date,
            )

            if series is None or len(series) == 0:
                print(f"  [!] {series_id:<20} -> sem dados")
                continue

            series = pd.Series(series, dtype="float64")
            series.index = pd.to_datetime(series.index)
            series = pd.to_numeric(series, errors="coerce").dropna()
            series = series[~series.index.duplicated(keep="last")]
            series = series.sort_index()

            series_data[col_name] = series

            last_date = series.index[-1].strftime("%Y-%m-%d")
            print(
                f"  [✓] {series_id:<20} -> {col_name:<32} "
                f"última: {last_date}"
            )

        except Exception as e:
            print(f"  [✗] {series_id:<20} -> erro: {e}")

    if not series_data:
        raise RuntimeError("Nenhuma série foi coletada do FRED.")

    return series_data


# =====================================================================
# 5. FUNÇÕES TEMPORAIS E ESTATÍSTICAS
# =====================================================================

def latest_value(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan, None
    return float(series.iloc[-1]), series.index[-1]


def value_on_or_before(series: pd.Series, target_date):
    """Último valor disponível em ou antes da data alvo."""

    if series is None:
        return np.nan, None

    series = series.dropna()
    if series.empty:
        return np.nan, None

    target_date = pd.Timestamp(target_date)
    available = series.loc[series.index <= target_date]

    if available.empty:
        return np.nan, None

    return float(available.iloc[-1]), available.index[-1]


def historical_value(series: pd.Series, days_ago: int):
    if series is None or series.empty:
        return np.nan, None

    latest_date = series.dropna().index[-1]
    return value_on_or_before(
        series,
        latest_date - pd.Timedelta(days=days_ago),
    )


def percentage_change(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return (current / previous - 1.0) * 100.0


def percentage_change_at(series: pd.Series, days_ago: int):
    current, _ = latest_value(series)
    previous, _ = historical_value(series, days_ago)
    return percentage_change(current, previous)


def absolute_change_at(series: pd.Series, days_ago: int):
    current, _ = latest_value(series)
    previous, _ = historical_value(series, days_ago)

    if pd.isna(current) or pd.isna(previous):
        return np.nan

    return current - previous


def change_pp_at(series: pd.Series, days_ago: int):
    return absolute_change_at(series, days_ago)


def yoy_growth(series: pd.Series):
    current, _ = latest_value(series)
    previous, _ = historical_value(series, 365)
    return percentage_change(current, previous)


def mom_growth(series: pd.Series):
    current, _ = latest_value(series)
    previous, _ = historical_value(series, 30)
    return percentage_change(current, previous)


def annualized_growth_from_months(series: pd.Series, months: int):
    """Crescimento anualizado aproximado usando meses observados."""

    current, _ = latest_value(series)

    if pd.isna(current):
        return np.nan

    target_date = series.index[-1] - pd.DateOffset(months=months)
    previous, _ = value_on_or_before(series, target_date)

    if pd.isna(previous) or previous <= 0:
        return np.nan

    periods_per_year = 12 / months
    return ((current / previous) ** periods_per_year - 1.0) * 100.0


def rolling_mean(series: pd.Series, periods: int):
    clean = series.dropna()
    if len(clean) < periods:
        return np.nan
    return float(clean.iloc[-periods:].mean())


def rolling_max(series: pd.Series, periods: int):
    clean = series.dropna()
    if len(clean) < periods:
        return np.nan
    return float(clean.iloc[-periods:].max())


def rolling_zscore(series: pd.Series, observations: int = 504, min_obs: int = 126):
    """Z-score rolling sobre observações reais da série, sem preencher datas."""

    clean = series.dropna().astype(float)

    if len(clean) < min_obs:
        return np.nan

    window = clean.iloc[-observations:]

    if len(window) < min_obs:
        return np.nan

    std = window.std(ddof=1)

    if std == 0 or pd.isna(std):
        return np.nan

    return float((window.iloc[-1] - window.mean()) / std)


def percentile_rank(series: pd.Series, observations: int = 504):
    clean = series.dropna().astype(float)

    if len(clean) < 30:
        return np.nan

    window = clean.iloc[-observations:]
    current = window.iloc[-1]

    return float((window <= current).mean() * 100.0)


# =====================================================================
# 6. PROCESSAMENTO DOS DADOS
# =====================================================================

def build_net_liquidity(series_data):
    """
    Calcula a proxy:

        Liquidez Líquida = Fed Assets - TGA - ON RRP

    O cálculo é feito nas datas semanais do WALCL, que é semanal e termina
    na quarta-feira. TGA é alinhado à mesma data e ON RRP usa o último
    valor disponível até a data.
    """

    assets = series_data.get("Fed_Total_Assets_M")
    tga = series_data.get("TGA_Balance_M")
    rrp = series_data.get("ON_RRP_B")

    if assets is None or tga is None or rrp is None:
        return None

    rows = []

    for date, asset_value in assets.dropna().items():
        tga_value, tga_date = value_on_or_before(tga, date)
        rrp_value, rrp_date = value_on_or_before(rrp, date)

        if pd.isna(tga_value) or pd.isna(rrp_value):
            continue

        liquidity = (
            asset_value / 1000.0
            - tga_value / 1000.0
            - rrp_value
        )

        rows.append({
            "date": date,
            "Fed_Assets_B": asset_value / 1000.0,
            "TGA_B": tga_value / 1000.0,
            "ON_RRP_B": rrp_value,
            "Net_Liquidity_B": liquidity,
            "TGA_Date": tga_date,
            "RRP_Date": rrp_date,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("date").sort_index()

    return df


def process_macro_data(series_data):
    """Calcula métricas derivadas sem destruir a frequência original."""

    processed = {
        "series": series_data,
        "net_liquidity": build_net_liquidity(series_data),
        "metrics": {},
    }

    # ---------------------------------------------------------------
    # Liquidez
    # ---------------------------------------------------------------

    liquidity_df = processed["net_liquidity"]

    if liquidity_df is not None and not liquidity_df.empty:
        net = liquidity_df["Net_Liquidity_B"]

        processed["metrics"]["Net_Liquidity_Current_B"] = float(net.iloc[-1])
        processed["metrics"]["Net_Liquidity_30D_Change_B"] = absolute_change_at(net, 30)
        processed["metrics"]["Net_Liquidity_90D_Change_B"] = absolute_change_at(net, 90)

    # ---------------------------------------------------------------
    # Z-score de crédito
    # ---------------------------------------------------------------

    for column, metric_name in [
        ("HY_Spread_Pct", "HY_Spread_ZScore"),
        ("IG_Spread_Pct", "IG_Spread_ZScore"),
    ]:
        if column in series_data:
            processed["metrics"][metric_name] = rolling_zscore(
                series_data[column]
            )
            processed[metric_name + "_Percentile"] = percentile_rank(
                series_data[column]
            )

    return processed


# =====================================================================
# 7. FORMATAÇÃO DOS INDICADORES
# =====================================================================

def fmt(value, decimals=2):
    if pd.isna(value):
        return "N/D"
    return f"{value:,.{decimals}f}"


def signed(value, decimals=2):
    if pd.isna(value):
        return "N/D"
    return f"{value:+,.{decimals}f}"


def obs_date(series):
    _, date = latest_value(series)
    return date.strftime("%Y-%m-%d") if date is not None else "N/D"


def direction_text(calc_type, current, comparison):
    if pd.isna(current) or pd.isna(comparison):
        return "N/D"

    if current == comparison:
        return "ESTÁVEL"

    if calc_type in {"price_return", "yoy_pct", "inflation_index", "gdp_growth", "payrolls"}:
        return "ALTA" if current > comparison else "QUEDA"

    if calc_type in {"change_pp", "change_abs", "spread", "change_index", "claims", "vix"}:
        return "ALTA" if current > comparison else "QUEDA"

    return "N/D"


def add_generic_indicator(lines, series, metadata):
    """Monta bloco interpretável de acordo com o tipo econômico da série."""

    if series is None or series.empty:
        return

    name = metadata["nome"]
    calc_type = metadata["calc_tipo"]
    current, date = latest_value(series)
    if pd.isna(current) or date is None:
        return

    date_text = date.strftime("%Y-%m-%d")

    # ---------------------------------------------------------------
    # Preços / índices financeiros
    # ---------------------------------------------------------------
    if calc_type == "price_return":
        returns = []
        for d in metadata.get("comparacoes", []):
            r = percentage_change_at(series, d)
            returns.append(f"{d}D={signed(r, 2)}%")

        lines.append(
            f"- {name}: {fmt(current)} {metadata['unidade']} | "
            + " | ".join(returns)
            + f" | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # Taxas / spreads / índices de condições financeiras
    # ---------------------------------------------------------------
    if calc_type in {"change_pp", "change_abs", "change_index"}:
        changes = []
        for d in metadata.get("comparacoes", []):
            ch = absolute_change_at(series, d)
            suffix = " pp" if calc_type == "change_pp" else ""
            changes.append(f"{d}D={signed(ch, 3)}{suffix}")

        lines.append(
            f"- {name}: {fmt(current)} {metadata['unidade']} | "
            + " | ".join(changes)
            + f" | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # Inflação por índice de preços
    # ---------------------------------------------------------------
    if calc_type == "inflation_index":
        mom = mom_growth(series)
        qoq_ann = annualized_growth_from_months(series, 3)
        six_ann = annualized_growth_from_months(series, 6)
        yoy = yoy_growth(series)

        lines.append(
            f"- {name}: índice={fmt(current, 3)} | "
            f"MoM={signed(mom, 2)}% | "
            f"3M anualizado={signed(qoq_ann, 2)}% | "
            f"6M anualizado={signed(six_ann, 2)}% | "
            f"YoY={signed(yoy, 2)}% | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # PIB
    # ---------------------------------------------------------------
    if calc_type == "gdp_growth":
        clean = series.dropna()
        if len(clean) >= 2:
            qoq = ((clean.iloc[-1] / clean.iloc[-2]) ** 4 - 1.0) * 100.0
        else:
            qoq = np.nan

        yoy = yoy_growth(series)

        lines.append(
            f"- {name}: nível={fmt(current, 1)} | "
            f"QoQ anualizado={signed(qoq, 2)}% | "
            f"YoY={signed(yoy, 2)}% | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # Payrolls
    # ---------------------------------------------------------------
    if calc_type == "payrolls":
        clean = series.dropna()
        monthly_change = (
            clean.iloc[-1] - clean.iloc[-2]
            if len(clean) >= 2 else np.nan
        )

        avg_3m = (
            clean.iloc[-3:].diff().mean()
            if len(clean) >= 4 else np.nan
        )

        yoy = yoy_growth(series)

        lines.append(
            f"- {name}: nível={fmt(current, 0)} mil | "
            f"variação mensal={signed(monthly_change, 0)} mil | "
            f"média 3M={signed(avg_3m, 0)} mil | "
            f"YoY={signed(yoy, 2)}% | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # Jobless claims
    # ---------------------------------------------------------------
    if calc_type == "claims":
        avg_4w = rolling_mean(series, 4)
        yoy = yoy_growth(series)

        lines.append(
            f"- {name}: atual={fmt(current, 0)} | "
            f"média 4S={fmt(avg_4w, 0)} | "
            f"YoY={signed(yoy, 2)}% | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # Crédito / spreads
    # ---------------------------------------------------------------
    if calc_type == "spread":
        changes = []
        for d in metadata.get("comparacoes", []):
            ch = absolute_change_at(series, d)
            bps = ch * 100.0 if not pd.isna(ch) else np.nan
            changes.append(f"{d}D={signed(bps, 1)} bps")

        z = rolling_zscore(series)
        pct = percentile_rank(series)

        lines.append(
            f"- {name}: {fmt(current, 2)}% | "
            + " | ".join(changes)
            + f" | Z-score 2A={signed(z, 2)}"
            + f" | Percentil 2A={fmt(pct, 1)}"
            + f" | obs={date_text}"
        )
        return

    # ---------------------------------------------------------------
    # VIX
    # ---------------------------------------------------------------
    if calc_type == "vix":
        avg_20 = rolling_mean(series, 20)
        max_30 = rolling_max(series, 30)
        ch_5 = absolute_change_at(series, 5)
        ch_30 = absolute_change_at(series, 30)

        lines.append(
            f"- {name}: {fmt(current, 2)} | "
            f"média 20D={fmt(avg_20, 2)} | "
            f"máx 30D={fmt(max_30, 2)} | "
            f"Δ5D={signed(ch_5, 2)} | "
            f"Δ30D={signed(ch_30, 2)} | obs={date_text}"
        )
        return

    # Fallback
    lines.append(
        f"- {name}: {fmt(current)} {metadata['unidade']} | obs={date_text}"
    )


# =====================================================================
# 8. PAYLOAD CONSOLIDADO PARA O GEMINI
# =====================================================================

def generate_agent_prompt_payload(processed):

    series_data = processed["series"]
    metrics = processed["metrics"]

    lines = []

    lines.append("=" * 76)
    lines.append("DADOS MACROECONÔMICOS CONSOLIDADOS — FRED")
    lines.append(
        f"Data de execução: {datetime.today().strftime('%Y-%m-%d %H:%M') }"
    )
    lines.append("Cada indicador informa sua própria data de observação.")
    lines.append("=" * 76)

    # ---------------------------------------------------------------
    # Liquidez construída
    # ---------------------------------------------------------------

    lines.append("\n1. LIQUIDEZ")
    lines.append("-" * 76)

    liquidity_df = processed["net_liquidity"]

    if liquidity_df is not None and not liquidity_df.empty:
        latest = liquidity_df.iloc[-1]
        latest_date = liquidity_df.index[-1].strftime("%Y-%m-%d")

        lines.append(
            f"- Liquidez Líquida (Fed Assets - TGA - ON RRP): "
            f"{fmt(latest['Net_Liquidity_B'], 2)} B | obs={latest_date}"
        )
        lines.append(
            f"  Δ30D={signed(metrics.get('Net_Liquidity_30D_Change_B'), 2)} B | "
            f"Δ90D={signed(metrics.get('Net_Liquidity_90D_Change_B'), 2)} B"
        )
        lines.append(
            f"  Componentes: Fed Assets={fmt(latest['Fed_Assets_B'], 2)} B | "
            f"TGA={fmt(latest['TGA_B'], 2)} B | "
            f"ON RRP={fmt(latest['ON_RRP_B'], 2)} B"
        )
        lines.append(
            "  Nota: métrica construída pelo programa; não é uma série oficial única do FRED."
        )
    else:
        lines.append("- Liquidez Líquida: EVIDÊNCIA INSUFICIENTE.")

    # Séries de liquidez complementares
    for fred_id in ["WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "M2SL"]:
        metadata = SERIES_METADATA[fred_id]
        add_generic_indicator(lines, series_data.get(metadata["coluna"]), metadata)

    # ---------------------------------------------------------------
    # Demais grupos
    # ---------------------------------------------------------------

    groups = [
        ("2. POLÍTICA MONETÁRIA", ["FEDFUNDS", "EFFR", "SOFR"]),
        (
            "3. CURVA DE JUROS",
            ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M"],
        ),
        ("4. JUROS REAIS", ["DFII5", "DFII10", "DFII30"]),
        ("5. EXPECTATIVAS DE INFLAÇÃO", ["T5YIE", "T10YIE"]),
        ("6. INFLAÇÃO", ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]),
        ("7. CICLO ECONÔMICO", ["GDPC1", "INDPRO", "HOUST", "RSAFS"]),
        ("8. MERCADO DE TRABALHO", ["UNRATE", "PAYEMS", "ICSA"]),
        ("9. CRÉDITO", ["BAMLH0A0HYM2", "BAMLC0A0CM"]),
        (
            "10. CONDIÇÕES FINANCEIRAS",
            ["NFCI", "NFCICREDIT", "NFCIRISK"],
        ),
        ("11. AÇÕES", ["SP500", "NASDAQCOM"]),
        ("12. VOLATILIDADE", ["VIXCLS"]),
        ("13. DÓLAR", ["DTWEXBGS", "DEXUSEU", "DEXJPUS"]),
        ("14. COMMODITIES", ["DCOILWTICO", "GOLDAMGBD228NLBM"]),
        ("15. CRIPTO", ["CBBTCUSD"]),
    ]

    for title, ids in groups:
        lines.append(f"\n{title}")
        lines.append("-" * 76)

        for fred_id in ids:
            metadata = SERIES_METADATA[fred_id]
            series = series_data.get(metadata["coluna"])

            if series is None or series.empty:
                lines.append(
                    f"- {metadata['nome']}: N/D (série não coletada)."
                )
                continue

            add_generic_indicator(lines, series, metadata)

    lines.append("\n" + "=" * 76)
    lines.append(
        "INSTRUÇÃO DE INTEGRIDADE: use somente os dados acima. "
        "Não invente valores ausentes, datas, eventos ou expectativas."
    )
    lines.append(
        "Quando um indicador não estiver disponível ou não puder sustentar "
        "uma conclusão, escreva EVIDÊNCIA INSUFICIENTE."
    )
    lines.append("=" * 76)

    return "\n".join(lines)


# =====================================================================
# 9. SYSTEM INSTRUCTION
# =====================================================================

SYSTEM_INSTRUCTION = r"""
Você é um Analista MacroEstratégico Sênior, especializado em macroeconomia,
ciclos econômicos, liquidez, política monetária, crédito, juros e interação
entre preços dos ativos.

OBJETIVO:
Produzir uma análise macroeconômica técnica, objetiva e orientada ao
acompanhamento de um investidor de perfil arrojado.

IMPORTANTE:
- Utilize somente os dados efetivamente fornecidos no payload.
- Não invente valores, datas, séries ou eventos.
- Diferencie fato, interpretação e hipótese.
- Não faça previsões categóricas.
- Não trate um indicador isolado como determinante do regime.
- Considere nível, direção, momentum e contexto histórico.
- Quando houver conflito entre indicadores, destaque a divergência.
- Quando não houver evidência suficiente, escreva "EVIDÊNCIA INSUFICIENTE".

============================================================
1. DIAGNÓSTICO DO REGIME MACRO
============================================================

Classifique descritivamente o regime entre:

- expansão + inflação controlada;
- expansão + inflação crescente;
- desaceleração + desinflação;
- estagflação;
- contração/recessão;
- transição entre regimes.

Explique quais evidências sustentam e quais contradizem a leitura.
Dê atenção especial aos indicadores antecedentes e ao momentum.

============================================================
2. POLÍTICA MONETÁRIA
============================================================

Analise:
- Federal Funds;
- Effective Fed Funds;
- SOFR;
- juros reais;
- inflação;
- expectativas de inflação;
- condições financeiras.

Diferencie política efetivamente observada de expectativas implícitas,
quando estas estiverem disponíveis.

Não invente expectativas de mercado que não estejam no payload.

============================================================
3. LIQUIDEZ
============================================================

Analise:
- Fed Assets;
- TGA;
- ON RRP;
- reservas bancárias;
- M2;
- Liquidez Líquida construída pelo programa.

Para a Liquidez Líquida, use:

Fed Assets - TGA - ON RRP

Trate essa métrica como proxy construída, não como série oficial do FRED.

Determine se a liquidez está:
EXPANDINDO / NEUTRA / CONTRAINDO

Avalie também aceleração/desaceleração e quais componentes explicam a mudança.

============================================================
4. CURVA DE JUROS
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

Explique se os movimentos parecem relacionados a:
- política monetária;
- crescimento;
- inflação;
- prêmio de prazo;
- risco fiscal;
- demanda por segurança.

Diferencie parte curta da parte longa da curva.

============================================================
5. INFLAÇÃO
============================================================

Use especialmente:
- CPI;
- Core CPI;
- PCE;
- Core PCE;
- breakevens;
- juros reais.

Avalie:
- MoM;
- 3M anualizado;
- 6M anualizado;
- YoY.

Diferencie persistência de desaceleração transitória.

============================================================
6. CICLO ECONÔMICO
============================================================

Analise:
- PIB;
- produção industrial;
- housing;
- retail sales;
- desemprego;
- payrolls;
- jobless claims.

Determine se o crescimento está:
ACELERANDO / ESTÁVEL / DESACELERANDO

Diferencie indicadores antecedentes, coincidentes e atrasados.

============================================================
7. CRÉDITO E CONDIÇÕES FINANCEIRAS
============================================================

Analise:
- HY OAS;
- IG OAS;
- Z-score;
- percentil histórico;
- NFCI;
- NFCI Credit;
- NFCI Risk.

Destaque especialmente:
- aumento rápido dos spreads;
- divergência entre HY e IG;
- divergência entre crédito e bolsa;
- deterioração das condições financeiras sem estresse aparente na bolsa.

============================================================
8. CONFIRMAÇÃO PELOS MERCADOS
============================================================

Analise:
- S&P 500;
- NASDAQ;
- VIX;
- dólar;
- ouro;
- petróleo;
- Bitcoin.

Verifique se os preços confirmam ou contradizem o cenário macro.

Não confunda preço forte com fundamento necessariamente positivo.

============================================================
9. DIVERGÊNCIAS MACRO × MERCADO
============================================================

Esta é uma seção prioritária.

Procure conflitos entre:
- crescimento;
- inflação;
- política monetária;
- liquidez;
- crédito;
- juros;
- bolsa;
- volatilidade;
- dólar;
- ouro/commodities;
- cripto.

Para cada divergência relevante use:

DIVERGÊNCIA #[n]

MACRO: [sinal predominante]
INFLAÇÃO: [sinal predominante]
LIQUIDEZ: [sinal predominante]
CRÉDITO: [sinal predominante]
JUROS: [sinal predominante]
BOLSA: [sinal predominante]
VOLATILIDADE: [sinal predominante]

INTERPRETAÇÃO:
[explique a divergência]

O QUE CONFIRMARIA:
[variáveis/movimentos]

O QUE INVALIDARIA:
[variáveis/movimentos]

IMPLICAÇÃO:
[classes de ativos potencialmente mais sensíveis]

Uma divergência é um sinal de atenção, não uma previsão automática de reversão.

============================================================
10. CENÁRIOS
============================================================

Construa:

CENÁRIO-BASE
CENÁRIO ALTISTA
CENÁRIO BAIXISTA

Para cada cenário apresente:
- dinâmica macro;
- evidências;
- riscos;
- ativos potencialmente favorecidos;
- indicadores de confirmação;
- indicadores de invalidação.

Não atribua probabilidades numéricas sem base quantitativa suficiente.

============================================================
11. MATRIZ DE ATIVOS
============================================================

Analise:
- Ações;
- Treasury/Renda Fixa;
- Crédito;
- Ouro;
- Commodities;
- Dólar;
- Cripto.

Para cada uma:

Regime | Direção | Momentum | Risco | Assimetria | O que monitorar

Para Direção use apenas:
POSITIVO / NEUTRO / NEGATIVO

Não produza ranking entre classes.

============================================================
12. IMPLICAÇÕES TÁTICAS
============================================================

Para cada classe apresente:

- o que favorece a exposição;
- o que ameaça a exposição;
- qual evidência mudaria a tese;
- horizonte relevante;
- principal risco de excesso de exposição;
- principal risco de subexposição.

Não transforme automaticamente uma condição macro em compra ou venda.

Quando houver assimetria, descreva a condição que a confirma.

============================================================
13. ALERTAS DE MUDANÇA DE REGIME
============================================================

Identifique os 5 indicadores mais importantes para monitorar.

Para cada um:
- valor;
- data;
- direção;
- momentum;
- por que importa;
- classe mais sensível;
- mudança que alteraria a tese.

============================================================
14. RESUMO EXECUTIVO
============================================================

Comece sempre com:

REGIME MACRO:
[uma frase]

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
15. DISCIPLINA ANALÍTICA
============================================================

Sempre que possível utilize:

DADO
→ MECANISMO
→ IMPACTO MACRO
→ IMPACTO NOS ATIVOS
→ O QUE CONFIRMARIA
→ O QUE INVALIDARIA

Destaque quando:
- o mercado antecipar algo ainda ausente nos dados;
- os dados mudarem antes dos preços;
- houver mudança de nível sem mudança de tendência;
- houver mudança de tendência;
- houver divergência entre classes de ativos;
- a evidência estiver dividida.

Não faça narrativa pós-fato sem explicitar que se trata de interpretação.
"""


# =====================================================================
# 10. CHAMADA AO GEMINI
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

    client = genai.Client(api_key=api_key_limpa)

    user_prompt = f"""
Realize a análise macroestratégica utilizando exclusivamente os dados abaixo.

Observe especialmente:
1. o que mudou recentemente;
2. momentum de 30, 90 e 365 dias quando disponível;
3. mudanças de regime;
4. divergências macro × mercado;
5. liquidez;
6. curva de juros;
7. crédito;
8. inflação;
9. ciclo econômico;
10. confirmação ou contradição pelos mercados.

DADOS:

{dados_fred_text}

Priorize a seção DIVERGÊNCIAS MACRO × MERCADO e termine com o RESUMO
EXECUTIVO e os 5 ALERTAS DE MUDANÇA DE REGIME.
"""

    ultimo_erro = None

    for modelo in MODELOS_CANDIDATOS:

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
                        temperature=0.2,
                    ),
                )

                if getattr(response, "text", None):
                    print(f"[✓] Análise gerada com {modelo}.")
                    return response.text

                raise RuntimeError("O modelo retornou resposta vazia.")

            except ServerError as e:
                ultimo_erro = e
                print(f"[!] Erro de servidor no {modelo}: {e}")

                if tentativa < 3:
                    espera = tentativa * 10
                    print(f"Aguardando {espera}s...")
                    time.sleep(espera)

            except Exception as e:
                ultimo_erro = e
                texto_erro = str(e).lower()

                # Erros de modelo inexistente/inválido não justificam 3 retries.
                if any(token in texto_erro for token in [
                    "404",
                    "not found",
                    "invalid argument",
                    "invalid model",
                ]):
                    print(f"[!] Modelo {modelo} indisponível: {e}")
                    break

                # Rate limit: retry com espera crescente.
                if "429" in texto_erro or "resource exhausted" in texto_erro:
                    print(f"[!] Rate limit em {modelo}: {e}")
                    if tentativa < 3:
                        espera = tentativa * 15
                        print(f"Aguardando {espera}s...")
                        time.sleep(espera)
                    continue

                print(f"[!] Erro no {modelo}: {e}")
                break

    raise RuntimeError(
        f"Não foi possível gerar a análise com os modelos configurados. "
        f"Último erro: {ultimo_erro}"
    )


# =====================================================================
# 11. TELEGRAM
# =====================================================================

def split_text(text: str, max_chars: int = TELEGRAM_MAX_CHAR):
    """Divide texto respeitando linhas quando possível."""

    if len(text) <= max_chars:
        return [text]

    blocos = []
    restante = text

    while len(restante) > max_chars:
        corte = restante.rfind("\n", 0, max_chars)

        if corte < int(max_chars * 0.60):
            corte = max_chars

        blocos.append(restante[:corte])
        restante = restante[corte:].lstrip("\n")

    if restante:
        blocos.append(restante)

    return blocos


def enviar_telegram(texto: str, token: str, chat_id: str):

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN não encontrado.")

    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID não encontrado.")

    url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
    blocos = split_text(texto)

    print(f"\nEnviando relatório para o Telegram: {len(blocos)} bloco(s).")

    for idx, bloco in enumerate(blocos, start=1):

        payload = {
            "chat_id": str(chat_id).strip(),
            "text": bloco,
        }

        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        try:
            result = response.json()
        except ValueError:
            result = {}

        if response.status_code != 200 or not result.get("ok"):
            raise RuntimeError(
                f"Erro Telegram no bloco {idx}: "
                f"HTTP {response.status_code} - {result.get('description', response.text)}"
            )

        print(f"  [✓] Bloco {idx}/{len(blocos)} enviado.")


# =====================================================================
# 12. RELATÓRIO FINAL
# =====================================================================

def montar_relatorio_telegram(analise_ia: str) -> str:

    data_execucao = datetime.today().strftime("%d/%m/%Y %H:%M")

    return (
        "MACROESTRATÉGIA — RELATÓRIO IA\n"
        f"Execução: {data_execucao}\n"
        f"{'=' * 58}\n\n"
        f"{analise_ia.strip()}"
    )


# =====================================================================
# 13. EXECUÇÃO
# =====================================================================

def main():

    print("\n" + "=" * 76)
    print("       ANALISTA MACROESTRATÉGICO — FRED + GEMINI")
    print("=" * 76)

    try:

        # -------------------------------------------------------------
        # FRED
        # -------------------------------------------------------------
        series_data = fetch_macro_data(FRED_API_KEY)

        print(
            f"\nSéries coletadas: {len(series_data)} / "
            f"{len(SERIES_METADATA)}"
        )

        # -------------------------------------------------------------
        # Processamento
        # -------------------------------------------------------------
        processed = process_macro_data(series_data)

        # -------------------------------------------------------------
        # Payload
        # -------------------------------------------------------------
        payload = generate_agent_prompt_payload(processed)

        # -------------------------------------------------------------
        # Gemini
        # -------------------------------------------------------------
        analise_ia = analisar_macro_com_gemini(payload)

        # -------------------------------------------------------------
        # Telegram
        # -------------------------------------------------------------
        relatorio = montar_relatorio_telegram(analise_ia)

        enviar_telegram(
            relatorio,
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID,
        )

        print("\n[✓] Processo concluído com sucesso.")

    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
        sys.exit(0)

    except Exception as e:
        print(f"\n[✗] ERRO FATAL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
