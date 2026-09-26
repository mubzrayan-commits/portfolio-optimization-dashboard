import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio Optimization Dashboard")
st.caption("Portfolio analytics, asset allocation and optimization engine")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Portfolio Inputs")

assets_text = st.sidebar.text_input(
    "Assets",
    "AAPL, MSFT, GOOGL, AMZN"
)

capital = st.sidebar.number_input(
    "Investment Capital ($)",
    min_value=1000.0,
    value=100000.0,
    step=1000.0
)

method = st.sidebar.selectbox(
    "Optimization Method",
    ["Maximum Sharpe", "Minimum Volatility"]
)

period = st.sidebar.selectbox(
    "Historical Data",
    ["1y", "2y", "5y"]
)

# -----------------------------
# ASSET LIST
# -----------------------------
assets = [
    x.strip().upper()
    for x in assets_text.split(",")
    if x.strip()
]

assets = list(dict.fromkeys(assets))

if len(assets) < 2:
    st.error("Enter at least two assets.")
    st.stop()

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(tickers, selected_period):
    data = yf.download(
        tickers,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(axis=1, how="all")
    data = data.dropna()

    return data


try:
    prices = load_data(assets, period)

    available = [a for a in assets if a in prices.columns]

    if len(available) < 2:
        st.error("Not enough valid assets were found.")
        st.stop()

    prices = prices[available]
    returns = prices.pct_change().dropna()

except Exception as e:
    st.error(f"Unable to download market data: {e}")
    st.stop()

# -----------------------------
# PORTFOLIO FUNCTIONS
# -----------------------------
annual_returns = returns.mean() * 252
covariance = returns.cov() * 252

n = len(available)

def portfolio_return(weights):
    return np.dot(weights, annual_returns.values)

def portfolio_volatility(weights):
    return np.sqrt(
        weights.T @ covariance.values @ weights
    )

def negative_sharpe(weights):
    volatility = portfolio_volatility(weights)

    if volatility == 0:
        return 999999

    return -portfolio_return(weights) / volatility

constraints = {
    "type": "eq",
    "fun": lambda weights: np.sum(weights) - 1
}

bounds = [(0, 1) for _ in range(n)]
initial_weights = np.ones(n) / n

if method == "Maximum Sharpe":
    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
else:
    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

if not result.success:
    st.error("Optimization failed.")
    st.stop()

weights = result.x

expected_return = portfolio_return(weights)
volatility = portfolio_volatility(weights)

sharpe = (
    expected_return / volatility
    if volatility > 0
    else 0
)

# -----------------------------
# RESULTS
# -----------------------------
st.subheader("Optimal Portfolio")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Expected Return",
    f"{expected_return * 100:.2f}%"
)

col2.metric(
    "Annual Volatility",
    f"{volatility * 100:.2f}%"
)

col3.metric(
    "Sharpe Ratio",
    f"{sharpe:.2f}"
)

col4.metric(
    "Capital",
    f"${capital:,.0f}"
)

# -----------------------------
# ALLOCATION
# -----------------------------
allocation = pd.DataFrame({
    "Asset": available,
    "Weight": weights,
    "Allocation ($)": weights * capital
})

allocation["Weight"] = allocation["Weight"] * 100

st.subheader("Asset Allocation")

st.dataframe(
    allocation.style.format({
        "Weight": "{:.2f}%",
        "Allocation ($)": "${:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    allocation.set_index("Asset")["Weight"]
)

# -----------------------------
# MARKET DATA
# -----------------------------
with st.expander("Market Data"):
    st.dataframe(
        prices.tail(10),
        use_container_width=True
    )

# -----------------------------
# RISK / RETURN
# -----------------------------
st.subheader("Asset Risk & Return")

asset_stats = pd.DataFrame({
    "Asset": available,
    "Annual Return": annual_returns.values * 100,
    "Annual Volatility": np.sqrt(
        np.diag(covariance.values)
    ) * 100
})

st.dataframe(
    asset_stats.style.format({
        "Annual Return": "{:.2f}%",
        "Annual Volatility": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# CORRELATION
# -----------------------------
st.subheader("Correlation Matrix")

st.dataframe(
    returns.corr().style.format("{:.2f}"),
    use_container_width=True
)

st.caption(
    "Optimization uses historical market data and long-only portfolio weights."
)