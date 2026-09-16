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

SERIES_MAP = {

    # ---------------------------------------------------------------
    # LIQUIDEZ
    # ---------------------------------------------------------------
    "WALCL": "Fed_Total_Assets_M",
    "WTREGEN": "TGA_Balance_M",
    "RRPONTSYD": "ON_RRP_B",
    "WRESBAL": "Bank_Reserves_M",
    "M2SL": "M2_B",

    # ---------------------------------------------------------------
    # POLÍTICA MONETÁRIA
    # ---------------------------------------------------------------
    "FEDFUNDS": "Fed_Funds_Rate",
    "EFFR": "Effective_Fed_Funds_Rate",

    # ---------------------------------------------------------------
    # CURVA DE JUROS NOMINAL
    # ---------------------------------------------------------------
    "DGS3MO": "Yield_3M",
    "DGS2": "Yield_2Y",
    "DGS5": "Yield_5Y",
    "DGS10": "Yield_10Y",
    "DGS30": "Yield_30Y",

    # Curvas específicas
    "T10Y2Y": "Yield_Curve_10Y2Y",
    "T10Y3M": "Yield_Curve_10Y3M",

    # ---------------------------------------------------------------
    # JUROS REAIS E EXPECTATIVAS DE INFLAÇÃO
    # ---------------------------------------------------------------
    "DFII5": "Real_Yield_5Y",
    "DFII10": "Real_Yield_10Y",
    "DFII30": "Real_Yield_30Y",

    "T5YIE": "Breakeven_Inflation_5Y",
    "T10YIE": "Breakeven_Inflation_10Y",

    # ---------------------------------------------------------------
    # INFLAÇÃO
    # ---------------------------------------------------------------
    "CPIAUCSL": "CPI",
    "CPILFESL": "Core_CPI",
    "PCEPI": "PCE",
    "PCEPILFE": "Core_PCE",

    # ---------------------------------------------------------------
    # ATIVIDADE ECONÔMICA
    # ---------------------------------------------------------------
    "GDPC1": "Real_GDP",
    "INDPRO": "Industrial_Production",
    "HOUST": "Housing_Starts",
    "RSAFS": "Retail_Sales",

    # Indicador antecedente
    "USSLIND": "Leading_Index",

    # ---------------------------------------------------------------
    # MERCADO DE TRABALHO
    # ---------------------------------------------------------------
    "UNRATE": "Unemployment_Rate",
    "PAYEMS": "Nonfarm_Payrolls",
    "ICSA": "Initial_Jobless_Claims",

    # ---------------------------------------------------------------
    # CRÉDITO
    # ---------------------------------------------------------------
    "BAMLH0A0HYM2": "HY_Spread_Pct",
    "BAMLC0A0CM": "IG_Spread_Pct",

    # ---------------------------------------------------------------
    # CONDIÇÕES FINANCEIRAS
    # ---------------------------------------------------------------
    "NFCI": "Chicago_Financial_Conditions",

    # ---------------------------------------------------------------
    # MERCADO DE AÇÕES
    # ---------------------------------------------------------------
    "SP500": "SP500",
    "NASDAQCOM": "NASDAQ",

    # ---------------------------------------------------------------
    # VOLATILIDADE
    # ---------------------------------------------------------------
    "VIXCLS": "VIX",

    # ---------------------------------------------------------------
    # DÓLAR
    # ---------------------------------------------------------------
    "DTWEXBGS": "Dollar_Broad_Index",
    "DEXUSEU": "Dollar_Euro",
    "DEXJPUS": "Dollar_Yen",

    # ---------------------------------------------------------------
    # COMMODITIES
    # ---------------------------------------------------------------
    "DCOILWTICO": "WTI_Oil",

    # Preço do ouro em USD/oz.
    "GOLDAMGBD228NLBM": "Gold_USD",

    # ---------------------------------------------------------------
    # CRIPTO
    # ---------------------------------------------------------------
    "CBBTCUSD": "Bitcoin_USD",
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
