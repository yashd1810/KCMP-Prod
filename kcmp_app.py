import streamlit as st
import pandas as pd, numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ta

DEFAULTS = { 'market_cap_min':1000, 'roce_min':15, 'sales_growth_min':12, 'profit_growth_min':10,
             'pe_max':30, 'de_max':0.5, 'promoter_holding_min':50, 'price_min':50 }

st.set_page_config(page_title="KC Momentum Picks", layout="wide")
st.title("KC Momentum Picks (KCMP) – NSE Stock Screener")

with st.sidebar:
    st.header("Fundamental Filters")
    market_cap_min = st.number_input("Market Cap ≥ (₹ Cr)", value=DEFAULTS['market_cap_min'])
    roce_min = st.number_input("ROCE ≥ (%)", value=DEFAULTS['roce_min'])
    sales_growth_min = st.number_input("Sales Growth YoY ≥ (%)", value=DEFAULTS['sales_growth_min'])
    profit_growth_min = st.number_input("Profit Growth YoY ≥ (%)", value=DEFAULTS['profit_growth_min'])
    pe_max = st.number_input("P/E ≤", value=DEFAULTS['pe_max'])
    de_max = st.number_input("Debt/Equity ≤", value=DEFAULTS['de_max'])
    promoter_holding_min = st.number_input("Promoter Holding ≥ (%)", value=DEFAULTS['promoter_holding_min'])
    price_min = st.number_input("Price ≥ (₹)", value=DEFAULTS['price_min'])
    include_bfsi = st.checkbox("Include BFSI stocks", value=True)
    if st.button("Reset to Defaults"):
        st.experimental_rerun()

if st.button("Run Screening"):
    st.info("Fetching stock universe & fundamentals…")
    nifty500_url = "https://www1.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df_universe = pd.read_csv(nifty500_url)
    except Exception as e:
        st.error("Failed to fetch NSE list: "+str(e))
        st.stop()
    symbols = df_universe['Symbol'].tolist()
    results = []

    for sym in symbols:
        ticker = yf.Ticker(sym + ".NS")
        info = ticker.info
        if info is None or info=={}:
            continue
        price = info.get("currentPrice") or info.get("previousClose")
        market_cap = (info.get("marketCap") or 0)/1e7  # to Cr
        pe = info.get("trailingPE") or np.nan
        sector = (info.get("sector") or '').lower()
        industry = (info.get("industry") or '').lower()
        is_bfsi = any(k in industry for k in ['bank', 'finance', 'financial', 'insurance']) or sector=='financial services'

        # derive additional metrics; some maybe None
        roce = info.get("returnOnEquity")
        if roce is not None:
            roce *= 100
        sales_growth = info.get("revenueGrowth")
        if sales_growth is not None:
            sales_growth *= 100
        profit_growth = info.get("earningsQuarterlyGrowth")
        if profit_growth is not None:
            profit_growth *= 100
        de = info.get("debtToEquity")
        promoter_holding = info.get("heldPercentInsiders")
        if promoter_holding is not None:
            promoter_holding *= 100

        # apply filters
        pass_fundamental = True
        if market_cap < market_cap_min:
            pass_fundamental = False
        if price is None or price < price_min:
            pass_fundamental = False
        if np.isnan(pe) or pe > pe_max:
            pass_fundamental = False
        if promoter_holding is None or promoter_holding < promoter_holding_min:
            pass_fundamental = False

        if not include_bfsi and is_bfsi:
            pass_fundamental = False

        if include_bfsi or not is_bfsi:
            # apply non-BFSI extended filters when applicable
            if not is_bfsi:
                if roce is None or roce < roce_min:
                    pass_fundamental = False
                if sales_growth is None or sales_growth < sales_growth_min:
                    pass_fundamental = False
                if profit_growth is None or profit_growth < profit_growth_min:
                    pass_fundamental = False
                if de is None or de > de_max:
                    pass_fundamental = False

        if pass_fundamental:
            results.append({'Symbol': sym,
                            'Price': price,
                            'MarketCapCr': round(market_cap,2),
                            'PE': round(pe,2) if not np.isnan(pe) else None,
                            'ROCE': round(roce,2) if roce is not None else None,
                            'SalesG%': round(sales_growth,2) if sales_growth is not None else None,
                            'ProfitG%': round(profit_growth,2) if profit_growth is not None else None,
                            'DE': round(de,2) if de is not None else None,
                            'Promoter%': round(promoter_holding,1) if promoter_holding is not None else None,
                            'IsBFSI': is_bfsi})

    df_fund = pd.DataFrame(results)
    st.success(f"Fundamental filter passed: {len(df_fund)} stocks")
    st.dataframe(df_fund, use_container_width=True)

    if len(df_fund):
        st.info("Computing momentum metrics (last 15 trading days)…")
        end = datetime.today()
        start = end - timedelta(days=30)
        sym_list = [s+".NS" for s in df_fund['Symbol']]
        ohlcv = yf.download(sym_list, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='1d', group_by='ticker', progress=False, threads=True)

        nifty = yf.download('^NSEI', start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
        if len(nifty) < 15:
            st.warning("Not enough Nifty data for ADX. Skipping momentum stage.")
            st.stop()
        adx_series = ta.trend.adx(nifty['High'], nifty['Low'], nifty['Close'], window=14)
        adx = adx_series.dropna().iloc[-1]
        regime = "TRENDING" if adx >= 20 else "SIDEWAYS"
        st.write(f"Market Regime: **{regime}** (Nifty50 ADX = {adx:.2f})")

        final_rows = []
        for sym in df_fund['Symbol']:
            data = ohlcv[sym+'.NS'].dropna()
            if data.empty or len(data) < 8:
                continue
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-8]) / data['Close'].iloc[-8] * 100
            vol_today = data['Volume'].iloc[-1]
            vol_avg10 = data['Volume'].iloc[-11:-1].mean()
            vol_multiple = vol_today / vol_avg10 if vol_avg10 else 0

            if regime == "TRENDING":
                vol_spike = vol_multiple >= 1.5
                pc_pass = price_change >= 5
            else:
                vol_spike = vol_multiple >= 1.1
                pc_pass = price_change >= 0

            obv = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            obv_trend_up = obv.iloc[-1] > obv.iloc[-8]

            if vol_spike and pc_pass and obv_trend_up:
                final_rows.append({'Symbol': sym,
                                   'PriceChange%_7d': round(price_change,2),
                                   'VolSpike×': round(vol_multiple,2),
                                   'OBVTrend': 'Up'})
        df_final = pd.DataFrame(final_rows)
        st.success(f"Momentum filter passed: {len(df_final)} stocks")
        st.dataframe(df_final, use_container_width=True)
