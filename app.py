import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

def format_ordinal_date(date_obj):
    day = date_obj.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {date_obj.strftime('%b %y')}"

# Load data
@st.cache_data

def load_data(upload):
    raw_df = pd.read_csv(upload)
    raw_df['DATE/TIME'] = pd.to_datetime(raw_df['DATE/TIME'], format='%d %b %Y %H:%M:%S')
    df = raw_df[raw_df['TYPE'].isin(['Close Bet', 'Stop Loss'])].copy()
    df['BEGIN DATE/TIME'] = raw_df['DATE/TIME'].shift(-1).loc[df.index]
    df['Direction'] = raw_df['TYPE'].shift(-1).loc[df.index]
    df['Direction'] = df['Direction'].apply(
        lambda x: 'Sell' if pd.notna(x) and 'Sell' in x else 'Buy' if pd.notna(x) and 'Buy' in x else x
    )
    df['Trade Duration (s)'] = (df['DATE/TIME'] - df['BEGIN DATE/TIME']).dt.total_seconds().abs().astype('Int64')
    df.rename(columns={'DATE/TIME': 'CLOSE DATE/TIME'}, inplace=True)
    df['DATE/TIME'] = df['BEGIN DATE/TIME']
    df.drop(columns=['BEGIN DATE/TIME'], inplace=True)
    df['STAKE'] = pd.to_numeric(df['STAKE'].astype(str).str.replace(',', ''), errors='coerce')
    df['PRICE'] = pd.to_numeric(df['PRICE'].astype(str).str.replace(',', ''), errors='coerce')
    df['AMOUNT (GBP)'] = pd.to_numeric(df['AMOUNT (GBP)'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['Net P&L'] = df['AMOUNT (GBP)']
    df['Trade Outcome'] = df['Net P&L'].apply(lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Break-even')
    df['DATE'] = df['DATE/TIME'].dt.date
    df['HOUR'] = df['DATE/TIME'].dt.hour
    df['DATETIME_HOUR'] = df['DATE/TIME'].dt.floor('h')
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Outlier Filtering")
    use_outlier_filtering = st.sidebar.checkbox("Enable outlier filtering")
    bottom_pct = st.sidebar.slider("Remove bottom X% trades", 0, 50, 0, step=1)
    top_pct = st.sidebar.slider("Remove top X% trades", 0, 50, 0, step=1)

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

    # Apply outlier trimming
    if use_outlier_filtering and (bottom_pct > 0 or top_pct > 0):
        pnl_sorted = df.sort_values('Net P&L (Adj)')
        n = len(pnl_sorted)
        bottom_n = int(n * bottom_pct / 100)
        top_n = int(n * top_pct / 100)
        df = pnl_sorted.iloc[bottom_n:n - top_n if top_n > 0 else n]
    pnl_col = 'Net P&L (Adj)' if (use_stop or use_takeprofit or use_trailing) else 'Net P&L'

    # Trade Duration Summary
    st.subheader("Trade Duration Summary")
    st.markdown("**Note:** Duration reflects time between opening and closing a position.")
    duration_seconds = df['Trade Duration (s)'].dropna()
    st.write("Min Duration:", f"{duration_seconds.min():,.0f} secs")
    st.write("Max Duration:", f"{duration_seconds.max():,.0f} secs")
    st.write("Average Duration:", f"{duration_seconds.mean():,.1f} secs")

    dur_min, dur_max = st.slider("Filter trades by duration (seconds)", 0, int(duration_seconds.max()), (0, int(duration_seconds.max())))
    duration_filtered = duration_seconds[(duration_seconds >= dur_min) & (duration_seconds <= dur_max)]

    fig_dur, ax_dur = plt.subplots()
    sns.histplot(duration_filtered, bins=30, kde=True, ax=ax_dur)
    ax_dur.set_xlabel("Trade Duration (seconds)")
    ax_dur.set_ylabel("Number of Trades")
    st.pyplot(fig_dur)
