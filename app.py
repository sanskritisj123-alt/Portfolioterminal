import warnings
warnings.filterwarnings("ignore")

import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf

from scipy.optimize import minimize


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DATABASE
# ============================================================

DB_FILE = "portfolio.db"

REQUIRED_COLUMNS = [
    "Ticker",
    "Quantity",
    "Purchase Price",
    "Purchase Date"
]

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC"
}


def get_connection():
    return sqlite3.connect(DB_FILE)


def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_name TEXT NOT NULL UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            purchase_price REAL NOT NULL,
            purchase_date TEXT NOT NULL,
            FOREIGN KEY (portfolio_id)
            REFERENCES portfolios(id)
            ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# ============================================================
# SAVE PORTFOLIO
# ============================================================

def save_portfolio_to_db(name, df):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            "SELECT id FROM portfolios WHERE portfolio_name = ?",
            (name,)
        )

        result = cursor.fetchone()

        if result:

            portfolio_id = result[0]

            # Replace existing holdings
            cursor.execute(
                "DELETE FROM holdings WHERE portfolio_id = ?",
                (portfolio_id,)
            )

        else:

            cursor.execute(
                """
                INSERT INTO portfolios (portfolio_name)
                VALUES (?)
                """,
                (name,)
            )

            portfolio_id = cursor.lastrowid

        for _, row in df.iterrows():

            cursor.execute(
                """
                INSERT INTO holdings
                (
                    portfolio_id,
                    ticker,
                    quantity,
                    purchase_price,
                    purchase_date
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(row["Ticker"]).upper().strip(),
                    float(row["Quantity"]),
                    float(row["Purchase Price"]),
                    str(row["Purchase Date"])
                )
            )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# GET SAVED PORTFOLIOS
# ============================================================

def get_saved_portfolios():

    conn = get_connection()

    query = """
        SELECT portfolio_name
        FROM portfolios
        ORDER BY portfolio_name
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df["portfolio_name"].tolist()


# ============================================================
# LOAD PORTFOLIO
# ============================================================

def load_portfolio_from_db(name):

    conn = get_connection()

    query = """
        SELECT
            ticker AS Ticker,
            quantity AS Quantity,
            purchase_price AS "Purchase Price",
            purchase_date AS "Purchase Date"
        FROM holdings
        WHERE portfolio_id = (
            SELECT id
            FROM portfolios
            WHERE portfolio_name = ?
        )
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(name,)
    )

    conn.close()

    if not df.empty:

        df["Purchase Date"] = pd.to_datetime(
            df["Purchase Date"],
            errors="coerce"
        )

    return df


# ============================================================
# DELETE PORTFOLIO
# ============================================================

def delete_portfolio_from_db(name):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM holdings
        WHERE portfolio_id = (
            SELECT id
            FROM portfolios
            WHERE portfolio_name = ?
        )
        """,
        (name,)
    )

    cursor.execute(
        """
        DELETE FROM portfolios
        WHERE portfolio_name = ?
        """,
        (name,)
    )

    conn.commit()
    conn.close()


# ============================================================
# SESSION STATE
# ============================================================

if "portfolio_df" not in st.session_state:

    st.session_state.portfolio_df = pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )


if "portfolio_name" not in st.session_state:

    st.session_state.portfolio_name = "My Portfolio"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_inr(value):

    if pd.isna(value):
        return "₹0.00"

    return f"₹{value:,.2f}"


def format_pct(value):

    if pd.isna(value):
        return "0.00%"

    return f"{value:.2f}%"


def clean_ticker(ticker):

    return str(ticker).upper().strip()


# ============================================================
# NORMALIZE PORTFOLIO
# ============================================================

def normalize_portfolio(df):

    df = df.copy()

    rename_map = {

        "ticker": "Ticker",
        "TICKER": "Ticker",

        "quantity": "Quantity",
        "QUANTITY": "Quantity",

        "purchase price": "Purchase Price",
        "Purchase price": "Purchase Price",
        "purchase_price": "Purchase Price",
        "PURCHASE PRICE": "Purchase Price",

        "purchase date": "Purchase Date",
        "Purchase date": "Purchase Date",
        "purchase_date": "Purchase Date",
        "PURCHASE DATE": "Purchase Date"
    }

    df.rename(
        columns=rename_map,
        inplace=True
    )

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    df["Ticker"] = (
        df["Ticker"]
        .apply(clean_ticker)
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Purchase Price"] = pd.to_numeric(
        df["Purchase Price"],
        errors="coerce"
    )

    df["Purchase Date"] = pd.to_datetime(
        df["Purchase Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    df = df[
        (df["Quantity"] > 0)
        &
        (df["Purchase Price"] > 0)
    ]

    return df


# ============================================================
# FETCH MARKET DATA
# ============================================================

@st.cache_data(ttl=900)
def fetch_market_data(tickers):

    data = {}

    for ticker in tickers:

        try:

            stock = yf.Ticker(ticker)

            hist = stock.history(
                period="2y",
                auto_adjust=True
            )

            if not hist.empty:

                data[ticker] = hist

        except Exception:

            continue

    return data


# ============================================================
# BENCHMARK DATA
# ============================================================

@st.cache_data(ttl=900)
def fetch_benchmark_data(ticker):

    try:

        data = yf.download(
            ticker,
            period="2y",
            auto_adjust=True,
            progress=False
        )

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )

        return data

    except Exception:

        return pd.DataFrame()


# ============================================================
# HOLDINGS CALCULATION
# ============================================================

def calculate_holdings(portfolio_df):

    if portfolio_df.empty:

        return pd.DataFrame()

    tickers = (
        portfolio_df["Ticker"]
        .unique()
    )

    market_data = fetch_market_data(
        tuple(tickers)
    )

    records = []

    for ticker in tickers:

        rows = portfolio_df[
            portfolio_df["Ticker"] == ticker
        ]

        quantity = (
            rows["Quantity"]
            .sum()
        )

        invested = (
            rows["Quantity"]
            *
            rows["Purchase Price"]
        ).sum()

        current_price = np.nan

        if ticker in market_data:

            hist = market_data[ticker]

            if not hist.empty:

                current_price = (
                    hist["Close"].iloc[-1]
                )

        if pd.isna(current_price):

            current_value = np.nan
            pnl = np.nan
            return_pct = np.nan

        else:

            current_value = (
                quantity *
                current_price
            )

            pnl = (
                current_value -
                invested
            )

            return_pct = (
                pnl / invested
                if invested != 0
                else np.nan
            )

        records.append({

            "Ticker": ticker,

            "Quantity": quantity,

            "Invested Value": invested,

            "Current Price": current_price,

            "Current Value": current_value,

            "P&L": pnl,

            "Return %": return_pct
        })

    holdings = pd.DataFrame(
        records
    )

    total_current = (
        holdings["Current Value"]
        .sum(skipna=True)
    )

    if total_current != 0:

        holdings["Portfolio Weight"] = (
            holdings["Current Value"]
            /
            total_current
        )

    else:

        holdings["Portfolio Weight"] = np.nan

    return holdings


# ============================================================
# PORTFOLIO RETURNS
# ============================================================

def build_portfolio_returns(holdings):

    if holdings.empty:

        return pd.Series(
            dtype=float
        )

    weights = (
        holdings
        .set_index("Ticker")
        ["Portfolio Weight"]
        .dropna()
    )

    if weights.empty:

        return pd.Series(
            dtype=float
        )

    market_data = fetch_market_data(
        tuple(weights.index)
    )

    prices = pd.DataFrame()

    for ticker in weights.index:

        if ticker in market_data:

            prices[ticker] = (
                market_data[ticker]["Close"]
            )

    if prices.empty:

        return pd.Series(
            dtype=float
        )

    returns = prices.pct_change()

    portfolio_returns = pd.Series(
        0.0,
        index=returns.index
    )

    for ticker in weights.index:

        if ticker in returns.columns:

            portfolio_returns += (
                returns[ticker]
                .fillna(0)
                *
                weights[ticker]
            )

    return portfolio_returns.dropna()


# ============================================================
# PERFORMANCE FUNCTIONS
# ============================================================

def annualized_return(returns):

    returns = returns.dropna()

    if len(returns) < 2:

        return np.nan

    cumulative = (
        1 + returns
    ).prod()

    years = len(returns) / 252

    return (
        cumulative ** (1 / years)
        - 1
    )


def annualized_volatility(returns):

    returns = returns.dropna()

    if len(returns) < 2:

        return np.nan

    return (
        returns.std()
        *
        np.sqrt(252)
    )


def sharpe_ratio(
    returns,
    risk_free=0.06
):

    annual_return = (
        annualized_return(returns)
    )

    volatility = (
        annualized_volatility(returns)
    )

    if (
        pd.isna(volatility)
        or
        volatility == 0
    ):

        return np.nan

    return (
        annual_return -
        risk_free
    ) / volatility


def sortino_ratio(
    returns,
    risk_free=0.06
):

    annual_return = (
        annualized_return(returns)
    )

    downside = (
        returns[returns < 0]
    )

    if downside.empty:

        return np.nan

    downside_deviation = (
        downside.std()
        *
        np.sqrt(252)
    )

    if downside_deviation == 0:

        return np.nan

    return (
        annual_return -
        risk_free
    ) / downside_deviation


def max_drawdown(returns):

    wealth = (
        1 + returns
    ).cumprod()

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth /
        running_max -
        1
    )

    return drawdown.min()


def value_at_risk(
    returns,
    confidence=0.95
):

    return np.percentile(
        returns.dropna(),
        (1 - confidence) * 100
    )


def conditional_var(
    returns,
    confidence=0.95
):

    var = value_at_risk(
        returns,
        confidence
    )

    losses = returns[
        returns <= var
    ]

    if losses.empty:

        return var

    return losses.mean()


def beta(
    portfolio_returns,
    benchmark_returns
):

    df = pd.concat(
        [
            portfolio_returns,
            benchmark_returns
        ],
        axis=1
    ).dropna()

    if len(df) < 2:

        return np.nan

    covariance = np.cov(
        df.iloc[:, 0],
        df.iloc[:, 1]
    )[0, 1]

    variance = np.var(
        df.iloc[:, 1]
    )

    if variance == 0:

        return np.nan

    return covariance / variance


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_weights(
    returns,
    method="max_sharpe"
):

    mean_returns = (
        returns.mean()
        *
        252
    )

    cov_matrix = (
        returns.cov()
        *
        252
    )

    n = len(
        mean_returns
    )

    initial_weights = (
        np.ones(n) / n
    )

    bounds = tuple(
        (0, 1)
        for _ in range(n)
    )

    constraints = {
        "type": "eq",
        "fun": lambda x:
            np.sum(x) - 1
    }

    def objective(weights):

        portfolio_return = np.dot(
            weights,
            mean_returns.values
        )

        volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    cov_matrix.values,
                    weights
                )
            )
        )

        if method == "max_sharpe":

            if volatility == 0:

                return 999

            return -(
                portfolio_return /
                volatility
            )

        return volatility

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:

        return None

    return pd.Series(
        result.x,
        index=mean_returns.index
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Portfolio Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Portfolio Input",
        "Dashboard",
        "Holdings",
        "Performance",
        "Risk Analysis",
        "Benchmark",
        "Optimization",
        "Efficient Frontier",
        "Correlation",
        "Stock Analysis"
    ]
)


# ============================================================
# PORTFOLIO INPUT PAGE
# ============================================================

if page == "Portfolio Input":

    st.title(
        "📂 Portfolio Management"
    )

    st.info(
        "Saved portfolios are stored permanently "
        "in portfolio.db."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📂 Upload Portfolio",
            "✏️ Manual Entry",
            "💾 Saved Portfolios"
        ]
    )


    # ========================================================
    # TAB 1 - UPLOAD
    # ========================================================

    with tab1:

        st.subheader(
            "📂 Upload Portfolio"
        )

        portfolio_name_upload = st.text_input(
            "Portfolio Name",
            value=st.session_state.portfolio_name,
            key="upload_portfolio_name"
        )

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
            key="portfolio_uploader"
        )

        if uploaded_file:

            try:

                if uploaded_file.name.endswith(
                    ".csv"
                ):

                    df = pd.read_csv(
                        uploaded_file
                    )

                else:

                    df = pd.read_excel(
                        uploaded_file
                    )

                df = normalize_portfolio(
                    df
                )

                # Save to current session
                st.session_state.portfolio_df = df

                st.session_state.portfolio_name = (
                    portfolio_name_upload
                )

                st.success(
                    "✅ Portfolio uploaded successfully!"
                )

                st.subheader(
                    "Uploaded Portfolio"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.divider()

                # ====================================================
                # SAVE UPLOADED PORTFOLIO
                # ====================================================

                if st.button(
                    "💾 Save Uploaded Portfolio Permanently",
                    type="primary",
                    use_container_width=True,
                    key="save_uploaded_portfolio"
                ):

                    try:

                        save_portfolio_to_db(
                            portfolio_name_upload,
                            df
                        )

                        st.success(
                            f"✅ Portfolio "
                            f"'{portfolio_name_upload}' "
                            "saved permanently!"
                        )

                    except Exception as e:

                        st.error(
                            f"Could not save portfolio: {e}"
                        )

            except Exception as e:

                st.error(
                    f"❌ Error reading file: {e}"
                )


    # ========================================================
    # TAB 2 - MANUAL ENTRY
    # ========================================================

    with tab2:

        st.subheader(
            "✏️ Manual Portfolio Entry"
        )

        portfolio_name_manual = st.text_input(
            "Portfolio Name",
            value=st.session_state.portfolio_name,
            key="manual_portfolio_name"
        )

        edited_df = st.data_editor(
            st.session_state.portfolio_df,
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button(
            "Update Current Portfolio",
            type="primary"
        ):

            try:

                cleaned = normalize_portfolio(
                    edited_df
                )

                st.session_state.portfolio_df = (
                    cleaned
                )

                st.session_state.portfolio_name = (
                    portfolio_name_manual
                )

                st.success(
                    "Current portfolio updated."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


        st.divider()

        if st.button(
            "💾 Save Manual Portfolio Permanently",
            use_container_width=True
        ):

            if (
                st.session_state
                .portfolio_df
                .empty
            ):

                st.warning(
                    "Please add holdings first."
                )

            else:

                try:

                    save_portfolio_to_db(
                        portfolio_name_manual,
                        st.session_state.portfolio_df
                    )

                    st.success(
                        f"Portfolio "
                        f"'{portfolio_name_manual}' "
                        "saved permanently!"
                    )

                except Exception as e:

                    st.error(
                        f"Could not save portfolio: {e}"
                    )


    # ========================================================
    # TAB 3 - SAVED PORTFOLIOS
    # ========================================================

    with tab3:

        st.subheader(
            "💾 Saved Portfolios"
        )

        saved_names = (
            get_saved_portfolios()
        )

        if not saved_names:

            st.info(
                "No saved portfolios yet."
            )

        else:

            selected_portfolio = st.selectbox(
                "Select Saved Portfolio",
                saved_names
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "📂 Load Portfolio",
                    type="primary",
                    use_container_width=True
                ):

                    loaded_df = (
                        load_portfolio_from_db(
                            selected_portfolio
                        )
                    )

                    if not loaded_df.empty:

                        st.session_state.portfolio_df = (
                            loaded_df
                        )

                        st.session_state.portfolio_name = (
                            selected_portfolio
                        )

                        st.success(
                            f"✅ '{selected_portfolio}' "
                            "loaded successfully!"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Portfolio contains no holdings."
                        )

            with col2:

                if st.button(
                    "🗑️ Delete Portfolio",
                    use_container_width=True
                ):

                    delete_portfolio_from_db(
                        selected_portfolio
                    )

                    st.success(
                        "Portfolio deleted."
                    )

                    st.rerun()


    # ========================================================
    # CURRENT PORTFOLIO PREVIEW
    # ========================================================

    st.divider()

    st.subheader(
        f"Current Portfolio: "
        f"{st.session_state.portfolio_name}"
    )

    if st.session_state.portfolio_df.empty:

        st.info(
            "No portfolio loaded."
        )

    else:

        st.dataframe(
            st.session_state.portfolio_df,
            use_container_width=True
        )

        holdings_preview = (
            calculate_holdings(
                st.session_state.portfolio_df
            )
        )

        if not holdings_preview.empty:

            total_investment = (
                st.session_state
                .portfolio_df["Quantity"]
                *
                st.session_state
                .portfolio_df["Purchase Price"]
            ).sum()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Holding Rows",
                    len(
                        st.session_state.portfolio_df
                    )
                )

            with col2:

                st.metric(
                    "Unique Stocks",
                    st.session_state
                    .portfolio_df[
                        "Ticker"
                    ].nunique()
                )

            with col3:

                st.metric(
                    "Total Investment",
                    format_inr(
                        total_investment
                    )
                )


    st.divider()

    csv_data = (
        st.session_state.portfolio_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Current Portfolio CSV",
        data=csv_data,
        file_name=(
            f"{st.session_state.portfolio_name}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    if st.button(
        "🗑️ Clear Current Portfolio"
    ):

        st.session_state.portfolio_df = (
            pd.DataFrame(
                columns=REQUIRED_COLUMNS
            )
        )

        st.success(
            "Current portfolio cleared."
        )

        st.rerun()

    st.stop()


# ============================================================
# CHECK PORTFOLIO
# ============================================================

portfolio_df = (
    st.session_state.portfolio_df
)

if portfolio_df.empty:

    st.title(
        "📊 Portfolio Intelligence"
    )

    st.warning(
        "No portfolio is loaded."
    )

    st.info(
        "Go to Portfolio Input → "
        "Upload Portfolio or Saved Portfolios."
    )

    st.stop()


holdings = calculate_holdings(
    portfolio_df
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "📊 Portfolio Dashboard"
    )

    st.caption(
        f"Portfolio: "
        f"{st.session_state.portfolio_name}"
    )

    total_invested = (
        holdings["Invested Value"]
        .sum()
    )

    total_current = (
        holdings["Current Value"]
        .sum()
    )

    total_pnl = (
        total_current -
        total_invested
    )

    total_return = (
        total_pnl /
        total_invested
        if total_invested != 0
        else np.nan
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Invested Value",
            format_inr(total_invested)
        )

    with col2:

        st.metric(
            "Current Value",
            format_inr(total_current)
        )

    with col3:

        st.metric(
            "Total P&L",
            format_inr(total_pnl)
        )

    with col4:

        st.metric(
            "Return",
            format_pct(
                total_return * 100
            )
        )


    st.divider()

    st.subheader(
        "Portfolio Allocation"
    )

    fig = px.pie(
        holdings,
        names="Ticker",
        values="Current Value",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Portfolio Holdings"
    )

    display_df = holdings.copy()

    display_df[
        "Invested Value"
    ] = display_df[
        "Invested Value"
    ].apply(format_inr)

    display_df[
        "Current Value"
    ] = display_df[
        "Current Value"
    ].apply(format_inr)

    display_df[
        "P&L"
    ] = display_df[
        "P&L"
    ].apply(format_inr)

    display_df[
        "Return %"
    ] = display_df[
        "Return %"
    ].apply(
        lambda x:
        format_pct(x * 100)
    )

    display_df[
        "Portfolio Weight"
    ] = display_df[
        "Portfolio Weight"
    ].apply(
        lambda x:
        format_pct(x * 100)
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )


# ============================================================
# HOLDINGS
# ============================================================

elif page == "Holdings":

    st.title(
        "📋 Holdings"
    )

    st.dataframe(
        holdings,
        use_container_width=True
    )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "Performance":

    st.title(
        "📈 Performance Analysis"
    )

    returns = (
        build_portfolio_returns(
            holdings
        )
    )

    if returns.empty:

        st.warning(
            "Unable to calculate portfolio returns."
        )

    else:

        annual_return = (
            annualized_return(
                returns
            )
        )

        volatility = (
            annualized_volatility(
                returns
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Annualized Return",
                format_pct(
                    annual_return * 100
                )
            )

        with col2:

            st.metric(
                "Annualized Volatility",
                format_pct(
                    volatility * 100
                )
            )


        cumulative = (
            1 + returns
        ).cumprod() - 1

        st.subheader(
            "Cumulative Portfolio Return"
        )

        fig = px.line(
            x=cumulative.index,
            y=cumulative.values * 100,
            labels={
                "x": "Date",
                "y": "Cumulative Return (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader(
            "Distribution of Daily Returns"
        )

        ret_df = pd.DataFrame({
            "Daily Return":
            returns.values * 100
        })

        fig = px.histogram(
            ret_df,
            x="Daily Return",
            nbins=50,
            title="Distribution of Daily Returns"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.title(
        "⚠️ Risk Analysis"
    )

    returns = (
        build_portfolio_returns(
            holdings
        )
    )

    if returns.empty:

        st.warning(
            "Unable to calculate risk metrics."
        )

    else:

        volatility = (
            annualized_volatility(
                returns
            )
        )

        sharpe = (
            sharpe_ratio(
                returns
            )
        )

        sortino = (
            sortino_ratio(
                returns
            )
        )

        drawdown = (
            max_drawdown(
                returns
            )
        )

        var95 = (
            value_at_risk(
                returns
            )
        )

        cvar95 = (
            conditional_var(
                returns
            )
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Volatility",
                format_pct(
                    volatility * 100
                )
            )

        with col2:

            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}"
                if not pd.isna(sharpe)
                else "N/A"
            )

        with col3:

            st.metric(
                "Sortino Ratio",
                f"{sortino:.2f}"
                if not pd.isna(sortino)
                else "N/A"
            )


        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Maximum Drawdown",
                format_pct(
                    drawdown * 100
                )
            )

        with col2:

            st.metric(
                "95% VaR",
                format_pct(
                    var95 * 100
                )
            )

        with col3:

            st.metric(
                "95% CVaR",
                format_pct(
                    cvar95 * 100
                )
            )


        wealth = (
            1 + returns
        ).cumprod()

        running_max = (
            wealth.cummax()
        )

        drawdowns = (
            wealth /
            running_max -
            1
        )

        st.subheader(
            "Drawdown Chart"
        )

        fig = px.area(
            x=drawdowns.index,
            y=drawdowns.values * 100,
            labels={
                "x": "Date",
                "y": "Drawdown (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# BENCHMARK
# ============================================================

elif page == "Benchmark":

    st.title(
        "📊 Benchmark Comparison"
    )

    selected_benchmark = st.selectbox(
        "Select Benchmark",
        list(BENCHMARKS.keys())
    )

    benchmark_ticker = (
        BENCHMARKS[
            selected_benchmark
        ]
    )

    benchmark_data = (
        fetch_benchmark_data(
            benchmark_ticker
        )
    )

    portfolio_returns = (
        build_portfolio_returns(
            holdings
        )
    )

    if (
        benchmark_data.empty
        or
        portfolio_returns.empty
    ):

        st.warning(
            "Unable to retrieve benchmark data."
        )

    else:

        benchmark_returns = (
            benchmark_data["Close"]
            .pct_change()
            .dropna()
        )

        combined = pd.concat(
            [
                portfolio_returns,
                benchmark_returns
            ],
            axis=1,
            join="inner"
        )

        combined.columns = [
            "Portfolio",
            "Benchmark"
        ]

        portfolio_cumulative = (
            1 +
            combined["Portfolio"]
        ).cumprod() - 1

        benchmark_cumulative = (
            1 +
            combined["Benchmark"]
        ).cumprod() - 1

        comparison = pd.DataFrame({

            "Portfolio":
            portfolio_cumulative * 100,

            "Benchmark":
            benchmark_cumulative * 100
        })

        fig = px.line(
            comparison,
            labels={
                "value":
                "Cumulative Return (%)"
            },
            title=(
                f"Portfolio vs "
                f"{selected_benchmark}"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        portfolio_beta = beta(
            combined["Portfolio"],
            combined["Benchmark"]
        )

        st.metric(
            "Beta",
            f"{portfolio_beta:.2f}"
            if not pd.isna(
                portfolio_beta
            )
            else "N/A"
        )


# ============================================================
# OPTIMIZATION
# ============================================================

elif page == "Optimization":

    st.title(
        "⚙️ Portfolio Optimization"
    )

    tickers = (
        holdings["Ticker"]
        .tolist()
    )

    if len(tickers) < 2:

        st.warning(
            "At least two stocks are required."
        )

    else:

        market_data = (
            fetch_market_data(
                tuple(tickers)
            )
        )

        prices = pd.DataFrame()

        for ticker in tickers:

            if ticker in market_data:

                prices[ticker] = (
                    market_data[ticker]["Close"]
                )

        returns = (
            prices
            .pct_change()
            .dropna()
        )

        if returns.empty:

            st.warning(
                "Insufficient market data."
            )

        else:

            method = st.selectbox(
                "Optimization Method",
                [
                    "Maximum Sharpe Ratio",
                    "Minimum Volatility"
                ]
            )

            method_key = (
                "max_sharpe"
                if method ==
                "Maximum Sharpe Ratio"
                else
                "min_volatility"
            )

            optimized = (
                optimize_weights(
                    returns,
                    method_key
                )
            )

            if optimized is None:

                st.error(
                    "Optimization failed."
                )

            else:

                result_df = pd.DataFrame({

                    "Ticker":
                    optimized.index,

                    "Optimized Weight":
                    optimized.values * 100
                })

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                fig = px.bar(
                    result_df,
                    x="Ticker",
                    y="Optimized Weight",
                    title=(
                        "Optimized "
                        "Portfolio Weights"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# EFFICIENT FRONTIER
# ============================================================

elif page == "Efficient Frontier":

    st.title(
        "📈 Efficient Frontier"
    )

    tickers = (
        holdings["Ticker"]
        .tolist()
    )

    if len(tickers) < 2:

        st.warning(
            "At least two stocks are required."
        )

    else:

        market_data = (
            fetch_market_data(
                tuple(tickers)
            )
        )

        prices = pd.DataFrame()

        for ticker in tickers:

            if ticker in market_data:

                prices[ticker] = (
                    market_data[ticker]["Close"]
                )

        returns = (
            prices
            .pct_change()
            .dropna()
        )

        if returns.empty:

            st.warning(
                "Insufficient market data."
            )

        else:

            mean_returns = (
                returns.mean() * 252
            )

            cov_matrix = (
                returns.cov() * 252
            )

            n = len(tickers)

            frontier = []

            for _ in range(100):

                weights = (
                    np.random.random(n)
                )

                weights /= (
                    weights.sum()
                )

                portfolio_return = np.dot(
                    weights,
                    mean_returns
                )

                portfolio_volatility = np.sqrt(
                    np.dot(
                        weights.T,
                        np.dot(
                            cov_matrix,
                            weights
                        )
                    )
                )

                frontier.append([
                    portfolio_volatility,
                    portfolio_return
                ])

            frontier_df = pd.DataFrame(
                frontier,
                columns=[
                    "Volatility",
                    "Return"
                ]
            )

            frontier_df[
                "Volatility"
            ] *= 100

            frontier_df[
                "Return"
            ] *= 100

            fig = px.scatter(
                frontier_df,
                x="Volatility",
                y="Return",
                title="Efficient Frontier",
                labels={
                    "Volatility":
                    "Volatility (%)",

                    "Return":
                    "Annualized Return (%)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# CORRELATION
# ============================================================

elif page == "Correlation":

    st.title(
        "🔗 Correlation Matrix"
    )

    tickers = (
        holdings["Ticker"]
        .tolist()
    )

    market_data = (
        fetch_market_data(
            tuple(tickers)
        )
    )

    prices = pd.DataFrame()

    for ticker in tickers:

        if ticker in market_data:

            prices[ticker] = (
                market_data[ticker]["Close"]
            )

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    if returns.empty:

        st.warning(
            "Unable to calculate correlation."
        )

    else:

        correlation = (
            returns.corr()
        )

        fig = px.imshow(
            correlation,
            text_auto=True,
            title="Stock Return Correlation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif page == "Stock Analysis":

    st.title(
        "🔎 Stock Analysis"
    )

    ticker = st.selectbox(
        "Select Stock",
        holdings["Ticker"].tolist()
    )

    market_data = (
        fetch_market_data(
            (ticker,)
        )
    )

    if ticker not in market_data:

        st.error(
            "Unable to retrieve stock data."
        )

    else:

        stock_data = (
            market_data[ticker]
        )

        st.subheader(
            f"{ticker} Price Chart"
        )

        fig = px.line(
            stock_data,
            x=stock_data.index,
            y="Close",
            title=(
                f"{ticker} Historical Price"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        latest_price = (
            stock_data["Close"].iloc[-1]
        )

        first_price = (
            stock_data["Close"].iloc[0]
        )

        stock_return = (
            latest_price /
            first_price -
            1
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Current Price",
                format_inr(
                    latest_price
                )
            )

        with col2:

            st.metric(
                "2-Year Return",
                format_pct(
                    stock_return * 100
                )
            )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "📊 Portfolio Intelligence"
)

st.sidebar.caption(
    "💾 Data stored in portfolio.db"
)