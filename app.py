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
    df['Direction'] = df['Direction'].apply(lambda x: 'Sell' if pd.notna(x) and 'Sell' in x else 'Buy' if pd.notna(x) and 'Buy' in x else x)
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
st.sidebar.title("📈 M.A.N.F.R.E.D.")
upload = st.sidebar.file_uploader("Upload CMC History CSV", type=["csv"])

if upload:
    df = load_data(upload)

    # Filters
    st.sidebar.title("Filters")
    min_date, max_date = df['DATE/TIME'].min().date(), df['DATE/TIME'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    stake_range = st.sidebar.slider("Stake Range", float(df['STAKE'].min()), float(df['STAKE'].max()), (float(df['STAKE'].min()), float(df['STAKE'].max())))
    result_filter = st.sidebar.multiselect("Trade Outcome", options=df['Trade Outcome'].unique(), default=df['Trade Outcome'].unique())
    direction_filter = st.sidebar.multiselect("Direction", options=df['Direction'].dropna().unique(), default=df['Direction'].dropna().unique())
    duration_unit_sidebar = st.sidebar.radio("Trade Duration Unit", options=["Seconds", "Minutes"], horizontal=True)
    duration_seconds = df['Trade Duration (s)'].dropna()
    durations_sidebar = duration_seconds if duration_unit_sidebar == "Seconds" else duration_seconds.div(60)
    dur_min, dur_max = st.sidebar.slider(f"Trade Duration ({'seconds' if duration_unit_sidebar == 'Seconds' else 'minutes'})", 0, int(durations_sidebar.max()), (0, int(durations_sidebar.max())))

    # Apply filters
    df_original = df.copy()  # Save pre-limit version for plotting comparison

    if len(date_range) == 2:
        df = df[(df['DATE/TIME'].dt.date >= date_range[0]) & (df['DATE/TIME'].dt.date <= date_range[1])]
    df = df[(df['STAKE'] >= stake_range[0]) & (df['STAKE'] <= stake_range[1])]
    if duration_unit_sidebar == 'Seconds':
        df = df[(df['Trade Duration (s)'] >= dur_min) & (df['Trade Duration (s)'] <= dur_max)]
    else:
        df = df[(df['Trade Duration (s)'] >= dur_min * 60) & (df['Trade Duration (s)'] <= dur_max * 60)]
    df = df[df['Trade Outcome'].isin(result_filter)]
    df = df[df['Direction'].isin(direction_filter)]

    use_outlier_filtering = st.sidebar.checkbox("Enable outlier filtering")
    bottom_pct = st.sidebar.slider("Remove bottom X% trades", 0, 50, 0, step=1)
    top_pct = st.sidebar.slider("Remove top X% trades", 0, 50, 0, step=1)

    # Remove Filters Button (reset filters to defaults)
    if st.sidebar.button("Reset Filters"):
        st.experimental_set_query_params()  # clear all URL query params
        st.session_state.clear()  # clear all session state
        st.rerun()

    # Simulations
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulations")
    use_stop = st.sidebar.checkbox("Apply stop-loss on losses")
    stop_level = st.sidebar.number_input("Stop-loss threshold (£)", min_value=0, max_value=10000, value=200, step=10)

    use_takeprofit = st.sidebar.checkbox("Apply take-profit on gains")
    takeprofit_level = st.sidebar.number_input("Take-profit threshold (£)", min_value=0, max_value=10000, value=300, step=10)

    use_trailing = False
    # use_trailing = st.sidebar.checkbox("Apply trailing stop-loss on gains")
    # trailing_gap = st.sidebar.number_input("Trailing stop gap (£)", min_value=0, max_value=10000, value=150, step=10)

    # st.sidebar.subheader("Win/Loss Trimming")
    trim_wins_pct = st.sidebar.slider("Trim % of winning trades", 0, 50, 0, step=1)
    trim_losses_pct = st.sidebar.slider("Trim % of losing trades", 0, 50, 0, step=1)

    limit_trades = st.sidebar.checkbox("Enable trade limits")
    max_trades_per_day = st.sidebar.number_input("Max trades per day", min_value=1, max_value=100, value=10, step=1)

    limit_hours = st.sidebar.checkbox("Enable time limits")
    max_hours_per_day = st.sidebar.number_input("Max trading hours per day", min_value=1, max_value=24, value=4, step=1)

    if limit_trades or limit_hours:
        df = df.sort_values(by=['DATE', 'DATE/TIME'])
        df['Trade #'] = df.groupby('DATE').cumcount() + 1
        df['Hour of Day'] = df['DATE/TIME'].dt.hour
        first_hour = df.groupby('DATE')['Hour of Day'].transform('min')
        df['Hours Since First'] = df['Hour of Day'] - first_hour

        if limit_trades:
            df = df[df['Trade #'] <= max_trades_per_day]
        if limit_hours:
            df = df[df['Hours Since First'] < max_hours_per_day]

    st.sidebar.markdown("---")
    use_partial_reintroduction = st.sidebar.checkbox("Include excluded trades")
    reintroduction_pct = st.sidebar.slider("Weight of excluded trades (%)", 0, 100, 50, step=5)

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

    # Apply win/loss trimming
    if trim_wins_pct > 0:
        win_df = df[df[pnl_col] > 0].sort_values(pnl_col, ascending=False)
        n_trim_wins = int(len(win_df) * trim_wins_pct / 100)
        df = df.drop(win_df.head(n_trim_wins).index)
    if trim_losses_pct > 0:
        loss_df = df[df[pnl_col] < 0].sort_values(pnl_col)
        n_trim_losses = int(len(loss_df) * trim_losses_pct / 100)
        df = df.drop(loss_df.head(n_trim_losses).index)

    # Apply reintroduction of excluded trades at reduced weight
    if use_partial_reintroduction:
        included_indices = df.index
        reintroduced = df_original[~df_original.index.isin(included_indices)].copy()
        # Ensure Net P&L (Adj) exists for reintroduced rows if simulations are used
        if pnl_col == 'Net P&L (Adj)':
            reintroduced['Net P&L (Adj)'] = reintroduced.apply(apply_simulation, axis=1)
        if pnl_col in reintroduced.columns:
            reintroduced[pnl_col] *= reintroduction_pct / 100
            df = pd.concat([df, reintroduced], ignore_index=True)
        else:
            st.warning(f"Column '{pnl_col}' missing from reintroduced trades; skipping reintroduction.")

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
    net_pnl_value = df[pnl_col].sum()
    net_pnl_str = f"(£{abs(net_pnl_value):,.2f})" if net_pnl_value < 0 else f"£{net_pnl_value:,.2f}"
    net_pnl_color = "#FF4B4B" if net_pnl_value < 0 else "#28A745"
    col3.metric("Net P&L", net_pnl_str)

    # Trade Data Table (Collapsible)
    with st.expander("🔍 Show All Filtered Trades"):
        # Drop columns with only NaN, empty strings or '-' values
        def is_meaningful(col):
            return not (
                col.isna().all()
                or (col.astype(str).str.strip() == "").all()
                or (col.astype(str).str.strip() == "-").all()
            )
        display_df = df.loc[:, df.apply(is_meaningful)]
        st.dataframe(display_df.sort_values(by='DATE/TIME', ascending=False), use_container_width=True)

    # st.markdown("---")

    # Charts
    figs = []

    st.subheader("Win/Loss Distribution")
    fig1, ax1 = plt.subplots()
    df['outcome'] = df[pnl_col].apply(lambda x: 'win' if x > 0 else 'loss' if x < 0 else 'break-even')
    sns.countplot(data=df, x='outcome', palette='Set2', ax=ax1)
    st.pyplot(fig1)
    figs.append(fig1)

    st.subheader("Daily Net P&L")
    # Daily Net P&L with Overlaid Trade Count
    daily_pnl = df.groupby('DATE')[pnl_col].sum().sort_index()
    daily_trades = df.groupby('DATE').size().sort_index()    
    fig2, ax2 = plt.subplots()
    daily_pnl.index = daily_pnl.index.to_series().apply(format_ordinal_date)
    daily_trades.index = daily_trades.index.to_series().apply(format_ordinal_date)
    # Bar chart for P&L
    bars = ax2.bar(daily_pnl.index, daily_pnl.values, color='skyblue', label='Net P&L')
    ax2.set_ylabel("P&L (£)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"(£{abs(x)/1000:.1f}k)" if x < -999 else
                     f"(£{int(round(abs(x)))})" if x < 0 else
                     f"£{x/1000:.1f}k" if x > 999 else
                     f"£{int(round(x))}"
    ))
    # Add secondary y-axis for trade count
    ax2b = ax2.twinx()
    ax2b.plot(daily_trades.index, daily_trades.values, color='orange', marker='o', linestyle='-', label='trades')
    ax2b.set_ylabel("trades")    
    st.pyplot(fig2)

    formatted_daily = (
        daily_pnl.reset_index()
        .rename(columns={pnl_col: 'Net P&L (£)'})
        .assign(DATE=pd.to_datetime(daily_pnl.index))
        .sort_values('DATE')
        .assign(DATE=lambda x: x['DATE'].apply(format_ordinal_date))
        .reset_index(drop=True)
    )
    st.dataframe(
        formatted_daily.style.format({'Net P&L (£)': lambda x: f"(£{abs(x):,.2f})" if x < 0 else f"£{x:,.2f}"}).applymap(
            lambda v: 'color: red' if isinstance(v, str) and v.startswith('(£') else ''
        ),
        column_config={"DATE": "Date"},
        use_container_width=True,
        hide_index=True
    )
    figs.append(fig2)
    fig_compare, ax_compare = plt.subplots(figsize=(10, 5))

    if not df_original.empty and {'DATE', 'DATETIME_HOUR', pnl_col}.issubset(df_original.columns):
        df_original_grouped = (
            df_original.groupby(['DATE', 'DATETIME_HOUR'])[pnl_col]
            .sum().reset_index()
        )
        df_original_grouped['Cumulative P&L'] = df_original_grouped.groupby('DATE')[pnl_col].cumsum()
    else:
        df_original_grouped = pd.DataFrame()
    if not df_original_grouped.empty and 'DATE' in df_original_grouped.columns:
        for date in df_original_grouped['DATE'].unique():
            subset = df_original_grouped[df_original_grouped['DATE'] == date]
            ax_compare.plot(subset['DATETIME_HOUR'], subset['Cumulative P&L'],
                            color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        ax_compare.plot([], [], color='gray', linestyle='--', label='original')
    df_grouped = df.groupby(['DATE', 'DATETIME_HOUR'])[pnl_col].sum().reset_index()
    df_grouped['Cumulative P&L'] = df_grouped.groupby('DATE')[pnl_col].cumsum()
    for date in df_grouped['DATE'].unique():
        subset = df_grouped[df_grouped['DATE'] == date]
        label = format_ordinal_date(pd.to_datetime(date))
        ax_compare.plot(subset['DATETIME_HOUR'], subset['Cumulative P&L'], marker='o', label=label)
    ax_compare.axhline(0, color='gray', linestyle='--')
    ax_compare.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_compare.set_ylabel('')
    ax_compare.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"(£{abs(x)/1000:.1f}k)" if x < -999 else f"(£{int(round(abs(x)))})" if x < 0 else f"£{x/1000:.1f}k" if x > 999 else f"£{int(round(x))}"))
    ax_compare.set_xticklabels([])
    st.pyplot(fig_compare)

    st.subheader("Trades by Hour")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='HOUR', palette='coolwarm', ax=ax3)
    ax3.set_xlabel('hour')
    st.pyplot(fig3)
    figs.append(fig3)
    
    st.subheader("Trade Duration Summary")
    # Recalculate display durations
    duration_unit = st.radio("Display Duration In:", options=["Seconds", "Minutes"], horizontal=True)
    durations = df['Trade Duration (s)'] if duration_unit == "Seconds" else df['Trade Duration (s)'].div(60)
    st.write("Min Duration:", f"{durations.min():,.0f} {'secs' if duration_unit == 'Seconds' else 'mins'}")
    st.write("Max Duration:", f"{durations.max():,.0f} {'secs' if duration_unit == 'Seconds' else 'mins'}")
    st.write("Average Duration:", f"{durations.mean():,.1f} {'secs' if duration_unit == 'Seconds' else 'mins'}")
    # Calculate £ per unit time
    total_duration = df['Trade Duration (s)'].sum()
    gbp_per_second = df[pnl_col].sum() / total_duration if total_duration else 0
    gbp_per_minute = gbp_per_second * 60
    if duration_unit == "Seconds":
        st.write("£ per Second:", f"\u00a3{gbp_per_second:.4f}/s")
    else:
        st.write("£ per Minute:", f"\u00a3{gbp_per_minute:.2f}/min")
    fig_dur, ax_dur = plt.subplots()
    sns.histplot(durations, bins=30, kde=True, ax=ax_dur)
    ax_dur.set_xlabel(f"duration ({'seconds' if duration_unit == 'Seconds' else 'minutes'})")
    ax_dur.set_ylabel("trades")
    st.pyplot(fig_dur)
    # Duration Buckets
    bins = [0, 15, 60, 120, 300, 900, 1800, float('inf')]
    labels = ["<15s", "15-60s", "1-2m", "2-5m", "5-15m", "15-30m", ">30m"]
    df['Duration Bucket'] = pd.cut(df['Trade Duration (s)'], bins=bins, labels=labels, right=False)

    st.subheader("Trade Distribution by Duration")
    fig_dur_dist, ax_dur_dist = plt.subplots()
    dur_counts = df['Duration Bucket'].value_counts(normalize=True).sort_index() * 100
    sns.barplot(x=dur_counts.index, y=dur_counts.values, palette="Blues", ax=ax_dur_dist)
    ax_dur_dist.set_ylabel("% of trades")
    ax_dur_dist.set_xlabel("duration")
    st.pyplot(fig_dur_dist)

    # Heatmap: % of Trades per Duration Bucket by Hour of Day
    duration_hour_counts = df.pivot_table(
        index='Duration Bucket',
        columns='HOUR',
        values=pnl_col,
        aggfunc='count',
        fill_value=0
    )
    duration_hour_pct = duration_hour_counts.div(duration_hour_counts.sum(axis=0), axis=1) * 100
    fig_ht, ax_ht = plt.subplots(figsize=(12, 5))
    sns.heatmap(duration_hour_pct.round().astype(int), annot=False, cmap="YlGnBu", ax=ax_ht, cbar_kws={'label': '% of trades'})
    ax_ht.set_xlabel("hour")
    ax_ht.set_ylabel("duration")
    st.pyplot(fig_ht)

    st.subheader("Trade Duration Performance")
    fig_dur_pnl, ax_dur_pnl = plt.subplots()
    dur_pnl = df.groupby('Duration Bucket')[pnl_col].mean().reindex(labels)
    bars = ax_dur_pnl.bar(dur_pnl.index, dur_pnl.values, color=sns.color_palette("RdYlGn", len(dur_pnl)))
    ax_dur_pnl.set_ylabel("P&L")
    ax_dur_pnl.set_xlabel("duration")
    ax_dur_pnl.set_xticks(range(len(labels)))
    ax_dur_pnl.set_xticklabels(labels)
    for bar, val in zip(bars, dur_pnl.values):
        if pd.isna(val):
            continue
        label = (
            f"(£{abs(val)/1000:.1f}k)" if val < -999 else
            f"(£{int(round(abs(val)))})" if val < 0 else
            f"£{val/1000:.1f}k" if val > 999 else
            f"£{int(round(val))}"
        )
        ax_dur_pnl.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                        ha='center', va='bottom', fontsize=8)
    st.pyplot(fig_dur_pnl)

    st.subheader("Weekdays Performance")
    daily_wl = df.groupby('DATE')[pnl_col].sum().reset_index()
    daily_wl['Outcome'] = daily_wl[pnl_col].apply(lambda x: 'winning' if x > 0 else 'losing' if x < 0 else 'flat')
    daily_wl['weekday'] = pd.to_datetime(daily_wl['DATE']).dt.strftime('%A')
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    toggle_heatmap_metric = st.radio("Metric", options=["Average P&L", "Day Count"], horizontal=True)
    if toggle_heatmap_metric == "Average P&L":
        breakdown = daily_wl.groupby(['Outcome', 'weekday'])[pnl_col].mean().unstack(fill_value=0)
        breakdown = breakdown.reindex(columns=weekday_order, fill_value=0)
        fmt_str = lambda x: f"(£{abs(x):,.2f})" if x < 0 else f"£{x:,.2f}"
    else:
        breakdown = daily_wl.groupby(['Outcome', 'weekday']).size().unstack(fill_value=0)
        breakdown = breakdown.reindex(columns=weekday_order, fill_value=0)
        fmt_str = lambda x: f"{int(x)}"
    breakdown['Total'] = breakdown.sum(axis=1)
    st.dataframe(breakdown.style.format(fmt_str))
    fig_hm, ax_hm = plt.subplots()
    heatmap_data = breakdown.drop(columns=['Total'])
    heatmap_data = heatmap_data.reindex(columns=weekday_order, fill_value=0)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor='gray',
        ax=ax_hm,
        cbar_kws={'label': 'metric'},
        annot_kws={"fontsize": 9},
        xticklabels=True,
        yticklabels=True,
        cbar=False
    )
    # Apply custom formatting for annotations
    for text in ax_hm.texts:
        try:
            val = float(text.get_text().replace(',', ''))
            if toggle_heatmap_metric == "Average P&L":
                if abs(val) >= 1000:
                    formatted = f"(£{abs(val)/1000:.1f}k)" if val < 0 else f"£{val/1000:.1f}k"
                else:
                    formatted = f"(£{int(round(abs(val)))})" if val < 0 else f"£{int(round(val))}"
            else:
                formatted = f"{int(val)}"
            text.set_text(formatted)
        except:
            continue
    ax_hm.set_facecolor('black')
    ax_hm.set_title("Heatmap of Day Outcomes by Weekday")
    st.pyplot(fig_hm)
    figs.append(fig_hm)

    st.subheader("Buy vs Sell Performance")
    direction_summary = df.groupby('Direction')[pnl_col].agg(['count', 'mean', 'sum']).rename(columns={'count': 'Trades', 'mean': 'Avg P&L', 'sum': 'Total P&L'})
    st.dataframe(direction_summary.style.format({
        'Avg P&L': lambda x: f"(£{abs(x):,.2f})" if x < 0 else f"£{x:,.2f}",
        'Total P&L': lambda x: f"(£{abs(x):,.2f})" if x < 0 else f"£{x:,.2f}"
    }))

    st.subheader("Manual Exit vs. Stop-Loss Performance")
    fig6, ax6 = plt.subplots()
    loss_only = df[df[pnl_col] < 0]
    sns.countplot(data=loss_only, x='TYPE', palette='pastel', ax=ax6)
    ax6.set_title("Exit Method Distribution (Losses Only)")
    ax6.set_xlabel("type")
    st.pyplot(fig6)
    figs.append(fig6)
    method_perf = df[df[pnl_col] < 0].groupby('TYPE')[pnl_col].agg(['count', 'mean', 'sum']).rename(columns={'count': 'Loss Trades', 'mean': 'Avg Loss', 'sum': 'Total Loss'})
    st.dataframe(
        method_perf.style.format({
            'Avg Loss': lambda x: f"(£{abs(x):,.2f})" if x < 0 else f"£{x:,.2f}",
            'Total Loss': lambda x: f"(£{abs(x):,.2f})" if x < 0 else f"£{x:,.2f}"
        }).applymap(lambda v: 'color: red' if isinstance(v, str) and v.startswith('(£') else '')
    )

    st.subheader("Effectiveness of Stop-Loss Follow-ups")
    # Slider to filter time window for follow-up trade
    max_minutes = 60
    followup_minutes = st.slider("Max minutes after stop-loss to consider a follow-up trade", 1, max_minutes, 10)
    successful_followups = 0
    total_qualified_followups = 0
    df_sorted = df.sort_values('DATE/TIME').reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        row = df_sorted.iloc[i]
        if row['TYPE'] == 'Stop Loss':
            next_trade = df_sorted.iloc[i + 1]
            # Check time gap
            time_diff = (next_trade['DATE/TIME'] - row['DATE/TIME']).total_seconds() / 60
            if time_diff > followup_minutes:
                continue
            # Check for opposite direction
            if row['Direction'] == 'Buy' and next_trade['Direction'] == 'Sell':
                direction_opposite = True
            elif row['Direction'] == 'Sell' and next_trade['Direction'] == 'Buy':
                direction_opposite = True
            else:
                direction_opposite = False

            if direction_opposite:
                total_qualified_followups += 1
                if next_trade[pnl_col] > 0:
                    successful_followups += 1
    # Calculate and display success rate
    if total_qualified_followups > 0:
        success_rate = (successful_followups / total_qualified_followups) * 100
    else:
        success_rate = 0
    # Build DataFrame of evaluated Stop Loss trades
    sl_records = []
    for idx in stop_loss_effective.index:
        try:
            sl_row = df.loc[idx]
            next_row = df.loc[idx + 1]
            if (next_row['Direction'] != sl_row['Direction']) and (next_row[pnl_col] > 0):
                effective = True
            else:
                effective = False
            time_diff = (next_row['DATE/TIME'] - sl_row['DATE/TIME']).total_seconds() / 60
            if time_diff <= time_limit_slider:
                sl_records.append({
                    "Stop Loss Time": sl_row["DATE/TIME"],
                    "Stop Loss Dir": sl_row["Direction"],
                    "Stop Loss P&L": sl_row[pnl_col],
                    "Next Trade Time": next_row["DATE/TIME"],
                    "Next Dir": next_row["Direction"],
                    "Next P&L": next_row[pnl_col],
                    "Effective": "Yes" if effective else "No",
                    "Minutes Later": f"{time_diff:.1f}"
                })
        except:
            continue
    sl_df = pd.DataFrame(sl_records)    
    with st.expander("📋 View Stop Loss Follow-up Trades"):
        st.dataframe(sl_df, use_container_width=True)
    # Chart to visualize
    fig_followup, ax_followup = plt.subplots()
    ax_followup.bar(['Successful', 'Unsuccessful'], [successful_followups, total_qualified_followups - successful_followups], color=["green", "red"])
    ax_followup.set_title(f"Follow-up Success Rate: {success_rate:.1f}%")
    ax_followup.set_ylabel("Number of Trades")
    st.pyplot(fig_followup)

    st.markdown("---")
    if st.button("📄 Export All Charts to PDF"):
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
        buffer.seek(0)
        st.download_button("Download PDF Report", buffer, file_name="trading_report.pdf", mime="application/pdf")

    st.caption("Upload a new file or adjust filters via the sidebar to refresh analysis.")
