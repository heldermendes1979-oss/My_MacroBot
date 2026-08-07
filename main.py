import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from google import genai
from google.genai import types

# =====================================================================
# 1. LEITURA DE VARIÁVEIS DE AMBIENTE SEGURAS (SECRETS DA NUVEM)
# =====================================================================
FRED_API_KEY = os.environ.get("FRED_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# =====================================================================
# 2. COLETA E PROCESSAMENTO DOS DADOS DO FRED
# =====================================================================
def fetch_macro_data(api_key: str, lookback_years: int = 2) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    start_date = (datetime.today() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
    
    series_map = {
        'WALCL': 'Fed_Total_Assets_M',
        'WTREGEN': 'TGA_Balance_M',
        'RRPONTSYD': 'ON_RRP_B',
        'BAMLH0A0HYM2': 'HY_Spread_Pct',
        'BAMLC0A0CM': 'IG_Spread_Pct',
        'T10Y2Y': 'Yield_Curve_10Y2Y',
        'DFII10': 'Real_Yield_10Y_TIPS',
        'FEDFUNDS': 'Fed_Funds_Rate'
    }
    
    data = {}
    print("Coletando séries temporais do FRED...")
    for series_id, col_name in series_map.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            data[col_name] = s
            print(f"  [✓] Coletado: {series_id} -> {col_name}")
        except Exception as e:
            print(f"  [✗] Erro ao coletar {series_id}: {e}")
            
    df = pd.DataFrame(data).ffill().dropna(how='all')
    return df

def process_liquidity_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fed_assets_b = df['Fed_Total_Assets_M'] / 1000.0
    tga_b = df['TGA_Balance_M'] / 1000.0
    rrp_b = df['ON_RRP_B']
    
    df['Net_Liquidity_B'] = fed_assets_b - tga_b - rrp_b
    df['Net_Liquidity_30D_Change_B'] = df['Net_Liquidity_B'].diff(periods=30)
    
    hy_mean = df['HY_Spread_Pct'].mean()
    hy_std = df['HY_Spread_Pct'].std()
    df['HY_Spread_ZScore'] = (df['HY_Spread_Pct'] - hy_mean) / hy_std
    
    return df

def generate_agent_prompt_payload(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    prev_month = df.iloc[-30] if len(df) >= 30 else df.iloc[0]
    
    payload = f"""
====================================================================
DADOS MACROECONÔMICOS CONSOLIDADOS (FRED) - {datetime.today().strftime('%Y-%m-%d')}
====================================================================

1. LIQUIDEZ LÍQUIDA DO FED:
   - Ativos Totais (Fed Balance Sheet): ${latest['Fed_Total_Assets_M']/1000:,.2f} Tri
   - Conta Geral do Tesouro (TGA):      ${latest['TGA_Balance_M']/1000:,.2f} Tri
   - Reverse Repo (RRP Overnight):      ${latest['ON_RRP_B']:,.2f} B
   -----------------------------------------------------------------
   - LIQUIDEZ LÍQUIDA ATUAL:            ${latest['Net_Liquidity_B']:,.2f} B
   - Variação em 30 Dias:              ${latest['Net_Liquidity_30D_Change_B']:+,.2f} B
   - Tendência de Liquidez:            {"EXPANSÃO (Apetite a Risco)" if latest['Net_Liquidity_30D_Change_B'] > 0 else "CONTRAÇÃO (Cautela/Risk-Off)"}

2. SPREADS DE CRÉDITO E ESTRESSE FINANCEIRO:
   - High Yield Option-Adjusted Spread: {latest['HY_Spread_Pct']:.2f}% (Há 30 dias: {prev_month['HY_Spread_Pct']:.2f}%)
   - Investment Grade Spread:          {latest['IG_Spread_Pct']:.2f}%
   - Z-Score Estresse High Yield:       {latest['HY_Spread_ZScore']:+.2f} ({'Estresse Elevado' if latest['HY_Spread_ZScore'] > 1 else 'Nível Normal'})

3. ESTRUTURA DE JUROS EUA:
   - Fed Funds Rate:                   {latest['Fed_Funds_Rate']:.2f}%
   - Curva 10Y - 2Y:                    {latest['Yield_Curve_10Y2Y']:+.2f}% ({'Invertida' if latest['Yield_Curve_10Y2Y'] < 0 else 'Normal/Desinvertida'})
   - Yield Real 10Y (TIPS):             {latest['Real_Yield_10Y_TIPS']:.2f}%
====================================================================
"""
    return payload

# =====================================================================
# 3. ANÁLISE COM O GEMINI
# =====================================================================
SYSTEM_INSTRUCTION = """
Você é um Analista MacroEstratégico Sênior focado em apoio à tomada de decisão de um investidor com perfil Arrojado.
Analise os dados do FRED e gere uma análise executiva e acionável.

DIRETRIZES:
1. Tom de voz direto, técnico e focado em assimetrias.
2. Analise a Liquidez Líquida do Fed, Spreads de Crédito e Curva de Juros.
3. Destaque o viés (Bullish, Bearish ou Neutro) para Ações, Renda Fixa, Commodities e Cripto.
4. Forneça recomendações táticas claras de rebalanceamento.
"""

def analisar_macro_com_gemini(dados_fred_text: str) -> str:
    print("\nGerando análise com Gemini (gemini-3.6-flash)...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=f"Aqui estão os dados atualizados do FRED para sua análise:\n\n{dados_fred_text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2
        )
    )
    return response.text

# =====================================================================
# 4. DISPARO PARA O TELEGRAM
# =====================================================================
def enviar_relatorio_telegram_completo(dados_fred: str, analise_ia: str, token: str, chat_id: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    relatorio_completo = (
        f"{dados_fred}\n\n"
        f"====================================================\n"
        f"        ANÁLISE MACROESTRATÉGICA & TÁTICA (IA)\n"
        f"====================================================\n\n"
        f"{analise_ia}"
    )
    
    MAX_CHAR = 3800
    blocos = [relatorio_completo[i:i + MAX_CHAR] for i in range(0, len(relatorio_completo), MAX_CHAR)]
    
    print(f"Enviando relatório completo ({len(blocos)} bloco(s)) para o Telegram...")
    
    for idx, bloco in enumerate(blocos):
        payload = {
            "chat_id": chat_id,
            "text": bloco
        }
        
        response = requests.post(url, json=payload)
        res_data = response.json()
        
        if response.status_code == 200 and res_data.get("ok"):
            print(f"  [✓] Bloco {idx + 1}/{len(blocos)} enviado com sucesso!")
        else:
            print(f"  [✗] Erro no bloco {idx + 1}: {res_data.get('description')}")

# =====================================================================
# 5. EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    df_raw = fetch_macro_data(FRED_API_KEY)
    df_processed = process_liquidity_and_metrics(df_raw)
    relatorio_fred = generate_agent_prompt_payload(df_processed)
    
    analise_ia = analisar_macro_com_gemini(relatorio_fred)
    
    enviar_relatorio_telegram_completo(relatorio_fred, analise_ia, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
