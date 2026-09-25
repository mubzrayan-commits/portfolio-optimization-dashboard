
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Portfolio Optimization Dashboard")
st.write("Optimize asset allocation using Maximum Sharpe or Minimum Volatility.")

default_assets = ["AAPL", "MSFT", "GOOGL", "AMZN"]

assets_text = st.text_input(
    "Assets",
    value=", ".join(default_assets),
    help="Enter ticker symbols separated by commas."
)

capital = st.number_input(
    "Investment Capital ($)",
    min_value=100.0,
    value=100000.0,
    step=1000.0
)

method = st.selectbox(
    "Optimization Method",
    ["Maximum Sharpe", "Minimum Volatility"]
)

def portfolio_dashboard(asset_text, capital, method):
    assets = [x.strip().upper() for x in asset_text.split(",") if x.strip()]
    assets = list(dict.fromkeys(assets))

    if len(assets) < 2:
        st.error("Please enter at least two assets.")
        return

    try:
        data = yf.download(
            assets,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )["Close"]

        if isinstance(data, pd.Series):
            data = data.to_frame()

        data = data.dropna(axis=1, how="all").dropna()

        available = [a for a in assets if a in data.columns]
        data = data[available]

        if len(available) < 2:
            st.error("Not enough valid ticker symbols were found.")
            return

        returns = data.pct_change().dropna()

        if returns.empty:
            st.error("Not enough historical price data.")
            return

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        n = len(available)

        def portfolio_return(weights):
            return np.dot(weights, mean_returns.values)

        def portfolio_volatility(weights):
            return np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix.values, weights))
            )

        def negative_sharpe(weights):
            volatility = portfolio_volatility(weights)
            if volatility == 0:
                return 1e6
            return -portfolio_return(weights) / volatility

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        initial = np.ones(n) / n

        if method == "Maximum Sharpe":
            result = minimize(
                negative_sharpe,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
        else:
            result = minimize(
                portfolio_volatility,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

        if not result.success:
            st.error("Optimization failed. Please try again.")
            return

        weights = result.x
        expected_return = portfolio_return(weights)
        volatility = portfolio_volatility(weights)
        sharpe = expected_return / volatility if volatility != 0 else 0

        allocation = pd.DataFrame({
            "Asset": available,
            "Weight": weights,
            "Investment ($)": weights * capital
        })

        st.subheader("Optimal Portfolio")

        display = allocation.copy()
        display["Weight"] = display["Weight"].map(lambda x: f"{x:.2%}")
        display["Investment ($)"] = display["Investment ($)"].map(
            lambda x: f"${x:,.2f}"
        )

        st.dataframe(display, hide_index=True, use_container_width=True)

        st.metric("Expected Annual Return", f"{expected_return:.2%}")
        st.metric("Annual Volatility", f"{volatility:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.write(f"**Investment Capital:** ${capital:,.2f}")

        st.subheader("Allocation Chart")
        chart_data = allocation.set_index("Asset")["Weight"]
        st.bar_chart(chart_data)

        st.caption("Historical data is retrieved from Yahoo Finance. This dashboard is for analysis and does not constitute investment advice.")

    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Optimize Portfolio", type="primary"):
    portfolio_dashboard(assets_text, capital, method)
