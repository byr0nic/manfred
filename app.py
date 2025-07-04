import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# Load data
@st.cache_data
def load_data(upload):
    df = pd.read_csv(upload)
    df['DATE/TIME'] = pd.to_datetime(df['DATE/TIME'], format='%d %b %Y %H:%M:%S')
    df = df[df['TYPE'].isin(['Close Bet', 'Stop Loss'])].copy()
    df['STAKE'] = pd.to_numeric(df['STAKE'].astype(str).str.replace(',', ''), errors='coerce')
    df['PRICE'] = pd.to_numeric(df['PRICE'].astype(str).str.replace(',', ''), errors='coerce')
    df['AMOUNT (GBP)'] = pd.to_numeric(df['AMOUNT (GBP)'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['Net P&L'] = df['AMOUNT (GBP)']
    df['Trade Outcome'] = df['Net P&L'].apply(lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Break-even')
    df['DATE'] = df['DATE/TIME'].dt.date
    df['HOUR'] = df['DATE/TIME'].dt.hour
    df['DATETIME_HOUR'] = df['DATE/TIME'].dt.floor('H')
    df['PRODUCT'] = df['PRODUCT'].str.strip()
    return df

# Sidebar upload
st.sidebar.title("Upload Trade History CSV")
upload = st.sidebar.file_uploader("Upload CMC History CSV", type=["csv"])

if upload:
    df = load_data(upload)
    st.title("📈 Trading Performance Dashboard")

    # Filters
    st.sidebar.subheader("Filters")
    min_date, max_date = df['DATE/TIME'].min().date(), df['DATE/TIME'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    stake_range = st.sidebar.slider("Stake Range", float(df['STAKE'].min()), float(df['STAKE'].max()), (float(df['STAKE'].min()), float(df['STAKE'].max())))
    result_filter = st.sidebar.multiselect("Trade Outcome", options=df['Trade Outcome'].unique(), default=df['Trade Outcome'].unique())

    # Apply filters
    if len(date_range) == 2:
        df = df[(df['DATE/TIME'].dt.date >= date_range[0]) & (df['DATE/TIME'].dt.date <= date_range[1])]
    df = df[(df['STAKE'] >= stake_range[0]) & (df['STAKE'] <= stake_range[1])]
    df = df[df['Trade Outcome'].isin(result_filter)]

    # Simulations
    st.sidebar.subheader("Simulations")
    use_stop = st.sidebar.checkbox("Apply stop-loss on losses")
    stop_level = st.sidebar.number_input("Stop-loss threshold (£)", min_value=0, max_value=10000, value=200, step=10)

    use_takeprofit = st.sidebar.checkbox("Apply take-profit on gains")
    takeprofit_level = st.sidebar.number_input("Take-profit threshold (£)", min_value=0, max_value=10000, value=300, step=10)

    use_trailing = st.sidebar.checkbox("Apply trailing stop-loss on gains")
    trailing_gap = st.sidebar.number_input("Trailing stop gap (£)", min_value=0, max_value=10000, value=150, step=10)

    def apply_simulation(row):
        pnl = row['Net P&L']
        if use_stop and pnl < -stop_level:
            pnl = -stop_level
        if use_takeprofit and pnl > takeprofit_level:
            pnl = takeprofit_level
        if use_trailing and pnl > trailing_gap:
            pnl = pnl - trailing_gap
        return pnl

    df['Net P&L (Adj)'] = df.apply(apply_simulation, axis=1)
    pnl_col = 'Net P&L (Adj)' if (use_stop or use_takeprofit or use_trailing) else 'Net P&L'

    # Summary stats
    total = len(df)
    wins = (df[pnl_col] > 0).sum()
    losses = (df[pnl_col] < 0).sum()
    win_rate = wins / total * 100 if total else 0
    avg_pnl = df[pnl_col].mean()
    avg_win = df[df[pnl_col] > 0][pnl_col].mean()
    avg_loss = df[df[pnl_col] < 0][pnl_col].mean()
    risk_reward = avg_win / abs(avg_loss) if avg_loss else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", total)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Net P&L", f"£{df[pnl_col].sum():.2f}")

    st.markdown("---")

    # Charts
    figs = []

    st.subheader("Win/Loss Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x=df[pnl_col].apply(lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Break-even'), palette='Set2', ax=ax1)
    st.pyplot(fig1)
    figs.append(fig1)

    st.subheader("Daily Net P&L")
    daily = df.groupby('DATE')[pnl_col].sum()
    fig2, ax2 = plt.subplots()
    daily.plot(kind='bar', ax=ax2)
    st.pyplot(fig2)
    figs.append(fig2)

    st.subheader("Trades by Hour")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='HOUR', palette='coolwarm', ax=ax3)
    st.pyplot(fig3)
    figs.append(fig3)

    st.subheader("Net P&L by Product")
    product_pnl = df.groupby('PRODUCT')[pnl_col].sum().sort_values()
    fig4, ax4 = plt.subplots()
    product_pnl.plot(kind='barh', ax=ax4)
    st.pyplot(fig4)
    figs.append(fig4)

    st.subheader("Intraday Cumulative P&L")
    intraday = df.groupby(['DATE', 'DATETIME_HOUR'])[pnl_col].sum().reset_index()
    intraday['Cumulative P&L'] = intraday.groupby('DATE')[pnl_col].cumsum()
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    for date in intraday['DATE'].unique():
        subset = intraday[intraday['DATE'] == date]
        ax5.plot(subset['DATETIME_HOUR'], subset['Cumulative P&L'], marker='o', label=str(date))
    ax5.axhline(0, color='gray', linestyle='--')
    ax5.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig5)
    figs.append(fig5)

    # PDF export button
    st.markdown("---")
    if st.button("📄 Export All Charts to PDF"):
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
        buffer.seek(0)
        st.download_button("Download PDF Report", buffer, file_name="trading_report.pdf", mime="application/pdf")

        st.subheader("Manual vs. Stop-Loss Exits")
    fig6, ax6 = plt.subplots()
    loss_only = df[df[pnl_col] < 0]
    sns.countplot(data=loss_only, x='TYPE', palette='pastel', ax=ax6)
    ax6.set_title("Exit Method Distribution (Losses Only)")
    st.pyplot(fig6)
    figs.append(fig6)

    st.subheader("Exit Method Performance Comparison")
    method_perf = df.groupby('TYPE')[pnl_col].agg(['count', 'mean', 'sum']).rename(columns={'count': 'Trades', 'mean': 'Avg P&L', 'sum': 'Total P&L'})
    st.dataframe(method_perf.style.format({'Avg P&L': '£{:.2f}', 'Total P&L': '£{:.2f}'}))

    st.caption("Upload a new file or adjust filters via the sidebar to refresh analysis.")
