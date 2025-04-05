import streamlit as st

# MUST be the VERY FIRST Streamlit command
st.set_page_config(page_title="Deep Value ATH Tracker", layout="wide")
import streamlit as st

# 1. Page config MUST be first
st.set_page_config(page_title="Protected App")

# 2. Password check
if "PASSWORD" not in st.secrets:
    st.error("Server error: Password not configured")
    st.stop()

password = st.text_input("Enter password:", type="password")
if password != st.secrets.PASSWORD:
    st.error("Wrong password")
    st.stop()

# ===== PASSWORD PROTECTION =====
# def check_password():
#     """Verify password from st.secrets with proper error handling."""
#     # Create empty container we can fill later
#     password_container = st.empty()
# 
#     # Check if secrets are available
#     if not hasattr(st, "secrets"):
#         password_container.warning("Running in dev mode - password bypassed")
#         return True
# 
#     if "PASSWORD" not in st.secrets:
#         password_container.error("Password not configured in secrets.toml")
#         st.stop()
# 
#     # Show password input
#     password_input = password_container.text_input(
#         "Enter Password:",
#         type="password",
#         key="pw_input"
#     )
#     # 
# 
# 
#     if password_input != st.secrets["PASSWORD"]:
#         if password_input:  # Only show error after first attempt
#             password_container.error("Wrong password")
#         st.stop()
#     return True




# Check password before proceeding
if not check_password():
    st.stop()

# Clear the password UI after successful authentication
st.empty()

# ===== ORIGINAL IMPORTS =====
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time

# Hardcoded default tickers
DEFAULT_TICKERS = [
    "SOL-USD", "HUT", "CLSK", "SHOP", "MSTR",
    "BTC-USD", "SOFI", "META", "PLTR",
    "PYPL", "TSLA", "AMD", "AMZN", "GOOGL",
    "QQQ", "MSFT", "AAPL", "NVDA", "EURHUF=X"
]

# Initialize session state
if 'tracked_tickers' not in st.session_state:
    st.session_state.tracked_tickers = DEFAULT_TICKERS.copy()
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None


@st.cache_data(ttl=3600)
def fetch_live_data(tickers):
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get current price
            current = info.get('currentPrice') or info.get('regularMarketPrice') or 0

            # Get 5-year highs/lows
            try:
                hist_5yr = stock.history(period="5y")
                ath = hist_5yr['High'].max() if not hist_5yr.empty else current * 1.5
                low_5yr = hist_5yr['Low'].min() if not hist_5yr.empty else current * 0.5
            except:
                ath = current * 1.5
                low_5yr = current * 0.5

            # Calculate metrics
            multiple_to_ath = ath / current if current > 0 else 1
            daily_change = info.get('regularMarketChangePercent', 0)

            data.append({
                "Ticker": ticker.replace('=X', ''),
                "FullTicker": ticker,
                "Company": str(info.get('shortName', ticker)),
                "Current": float(current),
                "5yr_ATH": float(ath),
                "5yr_Low": float(low_5yr),
                "Multiple_to_ATH": float(multiple_to_ath),
                "From_ATH%": ((current / ath) - 1) * 100,
                "From_5yrLow%": ((current / low_5yr) - 1) * 100,
                "Daily_Change%": float(daily_change),
                "Market_Cap": float(info.get('marketCap', 0)),
                "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            st.error(f"Error with {ticker}: {str(e)}")
            continue

    return pd.DataFrame(data).dropna(subset=['Current'])


def calculate_metrics(df):
    if df.empty:
        return df

    df['Deep_Value_Score'] = (
            (df['From_ATH%'].abs() * 0.7) +
            (df['From_5yrLow%'] * 0.3)
    )

    # Format with 1 decimal place
    pct_cols = ['From_ATH%', 'From_5yrLow%', 'Daily_Change%']
    for col in pct_cols:
        df[col] = df[col].round(1)

    df['Multiple_to_ATH'] = df['Multiple_to_ATH'].round(1)

    return df.sort_values('Deep_Value_Score', ascending=False)


def get_historical_data(ticker, months=12):
    """Get historical price data and calculate distance from ATH"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{months}mo")

        if hist.empty:
            return None

        # Calculate ATH up to each point in time
        hist['ATH'] = hist['High'].cummax()
        hist['From_ATH%'] = (hist['Close'] / hist['ATH'] - 1) * 100
        hist['Multiple_to_ATH'] = hist['ATH'] / hist['Close']
        hist['Date'] = hist.index.strftime('%Y-%m-%d')

        return hist[['Date', 'Close', 'ATH', 'From_ATH%', 'Multiple_to_ATH']].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error getting history for {ticker}: {str(e)}")
        return None


def color_multiple(val):
    """Color function for Multiple_to_ATH column"""
    if val > 2.0:
        return 'color: red'
    elif val > 1.5:
        return 'color: orange'
    else:
        return 'color: green'


def main():
    st.title("💰 Deep Value ATH Tracker")
    st.markdown("""
    **Strategy**: Find stocks FAR from 5-year highs (upside) and NEAR 5-year lows (downside protection)  
    *Select any stock from the dropdown for detailed historical analysis*
    """)

    # Ticker management
    with st.sidebar:
        st.header("Ticker Management")
        new_ticker = st.text_input("Add ticker (AAPL, BTC-USD, EURHUF=X):")

        if st.button("Add") and new_ticker:
            clean_ticker = new_ticker.upper().strip()
            if clean_ticker not in [t.upper() for t in st.session_state.tracked_tickers]:
                st.session_state.tracked_tickers.append(clean_ticker)
                st.rerun()

        if st.button("Reset to Defaults"):
            st.session_state.tracked_tickers = DEFAULT_TICKERS.copy()
            st.rerun()

        st.write("Current Tickers:")
        for i, ticker in enumerate(st.session_state.tracked_tickers[:25]):
            cols = st.columns([4, 1])
            cols[0].write(ticker)
            if cols[1].button("×", key=f"del_{i}"):
                st.session_state.tracked_tickers.remove(ticker)
                st.rerun()

    # Main data loading
    with st.spinner("Loading market data..."):
        df = fetch_live_data(st.session_state.tracked_tickers)
        df = calculate_metrics(df)

    if df.empty:
        st.error("No data loaded - try different tickers")
        return

    # Main metrics
    st.subheader("Portfolio Snapshot")
    cols = st.columns(4)
    cols[0].metric("Avg. Multiple to ATH", f"{df['Multiple_to_ATH'].mean():.1f}x")
    cols[1].metric("Avg. % Below ATH", f"{df['From_ATH%'].mean():.1f}%")
    cols[2].metric("Avg. % Above 5yr Low", f"{df['From_5yrLow%'].mean():.1f}%")
    cols[3].metric("Best Opportunity", df.iloc[0]['Ticker'])

    # Top opportunities
    st.subheader("🔥 Top 5 Deep Value Opportunities")
    top5 = df.head(5)
    for _, row in top5.iterrows():
        with st.expander(f"{row['Ticker']}: {row['Company']} - {row['Multiple_to_ATH']:.1f}x to ATH"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Current", f"${row['Current']:,.2f}")
            c2.metric("5yr ATH", f"${row['5yr_ATH']:,.2f}", f"{row['From_ATH%']:.1f}% below")
            c3.metric("5yr Low", f"${row['5yr_Low']:,.2f}", f"{row['From_5yrLow%']:.1f}% above")

            progress_value = min(100,
                                 max(0, (row['Current'] - row['5yr_Low']) / (row['5yr_ATH'] - row['5yr_Low']) * 100))
            st.progress(int(progress_value))
            st.write(
                f"**Potential Return**: {row['Multiple_to_ATH']:.1f}x (${1000 * row['Multiple_to_ATH']:,.0f} from $1k investment)")

    # Main data table with selection dropdown
    st.subheader("Detailed Analysis")

    # Format display dataframe
    display_df = df[[
        'Ticker', 'Company', 'Current', '5yr_ATH', 'Multiple_to_ATH',
        'From_ATH%', '5yr_Low', 'From_5yrLow%', 'Deep_Value_Score'
    ]].copy()

    display_df['Current'] = display_df['Current'].apply(lambda x: f"${x:,.2f}")
    display_df['5yr_ATH'] = display_df['5yr_ATH'].apply(lambda x: f"${x:,.2f}")
    display_df['5yr_Low'] = display_df['5yr_Low'].apply(lambda x: f"${x:,.2f}")

    # Display table with selection dropdown
    st.dataframe(
        display_df.sort_values('Multiple_to_ATH', ascending=False),
        height=800,
        use_container_width=True
    )

    # Ticker selection dropdown
    selected_ticker = st.selectbox(
        "Select a stock for detailed historical analysis:",
        df['Ticker'].unique()
    )

    if selected_ticker:
        st.session_state.selected_ticker = df[df['Ticker'] == selected_ticker]['FullTicker'].iloc[0]

    # Show detailed historical view when a ticker is selected
    if st.session_state.selected_ticker:
        selected_ticker = st.session_state.selected_ticker
        st.divider()
        st.subheader(f"📈 Historical Analysis: {selected_ticker}")

        # Get historical data
        hist_data = get_historical_data(selected_ticker)

        if hist_data is not None:
            # Plot distance from ATH over time
            fig = px.line(
                hist_data,
                x='Date',
                y='From_ATH%',
                title=f"{selected_ticker} - % From ATH Over Time",
                labels={'From_ATH%': '% From ATH', 'Date': 'Date'}
            )
            fig.update_layout(showlegend=False)

            # Add Multiple to ATH as secondary axis
            fig.add_scatter(
                x=hist_data['Date'],
                y=hist_data['Multiple_to_ATH'],
                yaxis="y2",
                name="Multiple to ATH",
                line=dict(color='green')
            )

            fig.update_layout(
                yaxis2=dict(
                    title="Multiple to ATH",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show historical data table with colored multiples
            st.write("Historical Data:")
            hist_display = hist_data.copy()
            hist_display['Close'] = hist_display['Close'].apply(lambda x: f"${x:,.2f}")
            hist_display['ATH'] = hist_display['ATH'].apply(lambda x: f"${x:,.2f}")
            hist_display['From_ATH%'] = hist_display['From_ATH%'].apply(lambda x: f"{x:.1f}%")
            hist_display['Multiple_to_ATH'] = hist_display['Multiple_to_ATH'].round(2)

            # Apply color formatting
            styled_df = hist_display[['Date', 'Close', 'ATH', 'From_ATH%', 'Multiple_to_ATH']].style.applymap(
                color_multiple,
                subset=['Multiple_to_ATH']
            )

            st.dataframe(
                styled_df,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning(f"Could not load historical data for {selected_ticker}")


if __name__ == "__main__":
    main()
