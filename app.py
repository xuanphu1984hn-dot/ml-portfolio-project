
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Stock Portfolio Dashboard", layout="wide")
import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data", "processed")
MODEL_DIR = os.path.join(APP_DIR, "models")
RESULTS_DIR = os.path.join(APP_DIR, "results")

# ===== Color palette (đồng bộ toàn app) =====
COLORS = {
    "primary": "#2563EB",
    "primary_light": "#93C5FD",
    "up": "#16A34A",
    "down": "#DC2626",
    "warn_bg": "#FFFBEB", "warn_fg": "#B45309", "warn_border": "#FDE68A",
    "err_bg": "#FEF2F2", "err_fg": "#B91C1C", "err_border": "#FECACA",
    "info_bg": "#EFF6FF", "info_fg": "#1D4ED8", "info_border": "#BFDBFE",
    "buy_bg": "#DCFCE7", "buy_fg": "#166534",
    "hold_bg": "#FEF9C3", "hold_fg": "#854D0E",
    "sell_bg": "#FEE2E2", "sell_fg": "#991B1B",
    "text": "#111827", "muted": "#6B7280", "border": "#E5E7EB",
}

PLOTLY_LINE_COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#4B5563", "#65A30D", "#EA580C",
]

# ===== Lucide Icons (SVG nhúng trực tiếp) =====
ICONS = {
    "trending-up": '<path d="M23 6l-9.5 9.5-5-5L1 18"></path><polyline points="17 6 23 6 23 12"></polyline>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"></path>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>',
    "bar-chart-3": '<path d="M3 3v18h18"></path><path d="M18 17V9"></path><path d="M13 17V5"></path><path d="M8 17v-3"></path>',
    "layout-dashboard": '<rect x="3" y="3" width="7" height="9" rx="1"></rect><rect x="14" y="3" width="7" height="5" rx="1"></rect><rect x="14" y="12" width="7" height="9" rx="1"></rect><rect x="3" y="16" width="7" height="5" rx="1"></rect>',
    "alert-triangle": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line>',
    "circle-check": '<circle cx="12" cy="12" r="10"></circle><path d="m9 12 2 2 4-4"></path>',
    "circle-minus": '<circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line>',
    "circle-x": '<circle cx="12" cy="12" r="10"></circle><path d="m15 9-6 6"></path><path d="m9 9 6 6"></path>',
    "info": '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line>',
    "target": '<circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
    "trophy": '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4Z"></path><path d="M17 5h3a2 2 0 0 1 2 2 4 4 0 0 1-4 4M7 5H4a2 2 0 0 0-2 2 4 4 0 0 0 4 4"></path>',
    "trending-down": '<path d="M23 18l-9.5-9.5-5 5L1 6"></path><polyline points="17 18 23 18 23 12"></polyline>',
}

def svg_icon(name, size=20, color="currentColor", stroke_width=2):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24"
    fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"
    style="vertical-align: -4px; margin-right: 6px;">{ICONS[name]}</svg>'''

def title_with_icon(name, text, subtitle=None, size=26, color=COLORS["text"]):
    st.markdown(f'<h1 style="display:flex;align-items:center;margin-bottom:0;">{svg_icon(name, size=size, color=color)}{text}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p style="color:{COLORS["muted"]};margin-top:2px;">{subtitle}</p>', unsafe_allow_html=True)

def sidebar_item_html(name, label, active=False):
    color = COLORS["primary"] if active else "#374151"
    weight = "600" if active else "400"
    return f'<div style="display:flex;align-items:center;padding:6px 0;color:{color};font-weight:{weight};">{svg_icon(name, size=18, color=color)}{label}</div>'

def signal_badge(rank):
    if rank <= 5:
        return f'<span style="background:{COLORS["buy_bg"]};color:{COLORS["buy_fg"]};padding:3px 10px;border-radius:12px;font-size:13px;font-weight:600;display:inline-flex;align-items:center;">{svg_icon("circle-check", size=14, color=COLORS["buy_fg"])}BUY</span>'
    elif rank <= 15:
        return f'<span style="background:{COLORS["hold_bg"]};color:{COLORS["hold_fg"]};padding:3px 10px;border-radius:12px;font-size:13px;font-weight:600;display:inline-flex;align-items:center;">{svg_icon("circle-minus", size=14, color=COLORS["hold_fg"])}HOLD</span>'
    else:
        return f'<span style="background:{COLORS["sell_bg"]};color:{COLORS["sell_fg"]};padding:3px 10px;border-radius:12px;font-size:13px;font-weight:600;display:inline-flex;align-items:center;">{svg_icon("circle-x", size=14, color=COLORS["sell_fg"])}SELL</span>'

def alert_box(name, text, kind="warning"):
    kind_map = {
        "warning": (COLORS["warn_bg"], COLORS["warn_fg"], COLORS["warn_border"]),
        "error": (COLORS["err_bg"], COLORS["err_fg"], COLORS["err_border"]),
        "info": (COLORS["info_bg"], COLORS["info_fg"], COLORS["info_border"]),
    }
    bg, fg, border = kind_map[kind]
    st.markdown(f'''<div style="background:{bg};color:{fg};border:1px solid {border};border-radius:8px;
    padding:12px 16px;display:flex;align-items:flex-start;margin:10px 0;font-size:14.5px;line-height:1.5;">
    {svg_icon(name, size=18, color=fg)}<span>{text}</span></div>''', unsafe_allow_html=True)

def kpi_card(icon, label, value, delta=None, delta_positive=True):
    # Luôn render dòng delta (dù rỗng) để mọi card có cùng chiều cao khi đặt cạnh nhau
    if delta is not None:
        dcolor = COLORS["up"] if delta_positive else COLORS["down"]
        dicon = "trending-up" if delta_positive else "trending-down"
        delta_html = f'<div style="display:flex;align-items:center;font-size:13px;color:{dcolor};">{svg_icon(dicon, size=14, color=dcolor)}{delta}</div>'
    else:
        delta_html = '&nbsp;'  # giữ chỗ trống bằng 1 dòng text để chiều cao khớp với card có delta
    st.markdown(f'''
    <div style="background:white;border:1px solid {COLORS["border"]};border-radius:12px;padding:16px 18px;
    box-shadow:0 1px 2px rgba(0,0,0,0.04);min-height:118px;display:flex;flex-direction:column;justify-content:space-between;">
        <div>
            <div style="display:flex;align-items:center;color:{COLORS["muted"]};font-size:13px;font-weight:500;">
                {svg_icon(icon, size=15, color=COLORS["muted"])}{label}
            </div>
            <div style="font-size:24px;font-weight:700;color:{COLORS["text"]};margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{value}</div>
        </div>
        <div style="font-size:13px;line-height:16px;margin-top:6px;">{delta_html}</div>
    </div>''', unsafe_allow_html=True)

def plotly_layout(fig, height=440, hovermode="x unified"):
    fig.update_layout(
        template="plotly_white",
        height=height,
        hovermode=hovermode,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="-apple-system, Segoe UI, sans-serif", size=13, color=COLORS["text"]),
        colorway=PLOTLY_LINE_COLORS,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig

# ===== Custom CSS =====
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background-color: #F9FAFB; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    h1, h2, h3 { font-family: -apple-system, "Segoe UI", sans-serif; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E5E7EB; border-radius: 12px;
        padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    div[data-testid="column"] { padding: 4px; }
    div[data-testid="stHorizontalBlock"] { align-items: stretch; }
</style>
""", unsafe_allow_html=True)

# ===== Load dữ liệu =====
@st.cache_data
def load_data():
    price_history = pd.read_csv(os.path.join(DATA_DIR, "panel_clean_features.csv"), parse_dates=["Date"])
    latest_features = pd.read_csv(os.path.join(DATA_DIR, "latest_features.csv"), parse_dates=["Date"])
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    ablation_df = pd.read_csv(os.path.join(RESULTS_DIR, "phase11_ablation_study.csv"))
    return price_history, latest_features, feature_cols, ablation_df

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")), joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))

@st.cache_data
def load_returns():
    names = {
        "Buy & Hold": "baseline_buyhold_returns.csv", "Equal Weight": "baseline_equalweight_returns.csv",
        "Linear Regression": "baseline_linearreg_returns.csv", "Random Forest": "rf_returns.csv",
        "XGBoost": "xgb_returns.csv", "LSTM": "lstm_returns.csv",
        "Proposed (VolWeight)": "proposed_strategy_test_returns.csv",
    }
    return {name: pd.read_csv(os.path.join(RESULTS_DIR, fname), index_col=0, parse_dates=True).iloc[:, 0] for name, fname in names.items()}

def compute_metrics(returns, freq=252, rf_rate=0.0):
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / freq
    cagr = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = returns.std() * np.sqrt(freq)
    ann_return = returns.mean() * freq
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else np.nan
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = (returns > 0).mean()
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    return {"Total Return (%)": round(total_return * 100, 2), "CAGR (%)": round(cagr * 100, 2),
        "Volatility (%)": round(ann_vol * 100, 2), "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": round(max_drawdown * 100, 2), "Win Rate (%)": round(win_rate * 100, 2),
        "Calmar Ratio": round(calmar, 3)}

with st.spinner("Đang tải dữ liệu..."):
    price_history, latest_features, feature_cols, ablation_df = load_data()
    xgb_model, scaler = load_model()
    returns_dict = load_returns()

# ===== Sidebar =====
st.sidebar.markdown(f'<h2 style="display:flex;align-items:center;">{svg_icon("layout-dashboard", size=24, color=COLORS["primary"])}AI Portfolio</h2>', unsafe_allow_html=True)

page_options = {
    "Market Overview": "trending-up",
    "AI Stock Prediction": "cpu",
    "Recommended Portfolio": "briefcase",
    "Backtest": "bar-chart-3",
}
page = st.sidebar.radio("", list(page_options.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Đồ án: Ứng dụng ML trong dự báo lợi suất cổ phiếu & xây dựng danh mục đầu tư")
with st.sidebar:
    alert_box("alert-triangle", "Công cụ học thuật, không phải khuyến nghị đầu tư thực tế.", "warning")

# ===== DASHBOARD 1: MARKET OVERVIEW =====
if page == "Market Overview":
    title_with_icon("trending-up", "Market Overview", "Tổng quan giá 25 cổ phiếu large-cap S&P 500 (2000–2026)")

    tickers = sorted(price_history["Ticker"].unique())

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_tickers = st.multiselect("Chọn cổ phiếu", tickers, default=["AAPL", "MSFT", "XOM"])
            date_range = st.date_input("Khoảng thời gian",
                value=[pd.Timestamp("2022-01-01"), price_history["Date"].max()],
                min_value=price_history["Date"].min(), max_value=price_history["Date"].max())
        with col2:
            if selected_tickers and len(date_range) == 2:
                filtered = price_history[(price_history["Ticker"].isin(selected_tickers)) &
                    (price_history["Date"] >= pd.Timestamp(date_range[0])) &
                    (price_history["Date"] <= pd.Timestamp(date_range[1]))]
                fig = go.Figure()
                for t in selected_tickers:
                    sub = filtered[filtered["Ticker"] == t]
                    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Adj Close"], mode="lines", name=t))
                fig.update_yaxes(title_text="Adjusted Close ($)")
                fig = plotly_layout(fig)
                st.plotly_chart(fig, use_container_width=True)

    # KPI cards: biến động % của các mã đã chọn trong khoảng thời gian
    if selected_tickers and len(date_range) == 2 and not filtered.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        kpi_cols = st.columns(min(len(selected_tickers), 4))
        for i, t in enumerate(selected_tickers[:4]):
            sub = filtered[filtered["Ticker"] == t].sort_values("Date")
            if len(sub) >= 2:
                pct = (sub["Adj Close"].iloc[-1] / sub["Adj Close"].iloc[0] - 1) * 100
                with kpi_cols[i % 4]:
                    kpi_card("activity", t, f"${sub['Adj Close'].iloc[-1]:,.2f}",
                              f"{pct:+.2f}% trong kỳ", delta_positive=pct >= 0)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Bảng giá mới nhất (25 mã)")
    latest_prices = price_history.sort_values("Date").groupby("Ticker").tail(1)[["Ticker", "Date", "Adj Close", "Volume"]]
    st.dataframe(
        latest_prices.sort_values("Ticker").reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Adj Close": st.column_config.NumberColumn("Adj Close ($)", format="$%.2f"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Date": st.column_config.DateColumn("Date"),
        },
        hide_index=True,
    )

# ===== DASHBOARD 2: AI STOCK PREDICTION =====
elif page == "AI Stock Prediction":
    as_of_date = latest_features["Date"].max()
    title_with_icon("cpu", "AI Stock Prediction", f"Dự báo dựa trên dữ liệu tính đến ngày: {as_of_date.date()} (mô hình XGBoost, horizon 5 ngày)")

    with st.spinner("Đang tính toán dự báo..."):
        X = latest_features[feature_cols].values
        X_scaled = scaler.transform(X)
        preds = xgb_model.predict(X_scaled)
        pred_df = latest_features[["Ticker", "Adj Close"]].copy()
        pred_df["Predicted_Return_5D (%)"] = (preds * 100).round(2)
        pred_df["Rank"] = pred_df["Predicted_Return_5D (%)"].rank(ascending=False).astype(int)
        pred_df = pred_df.sort_values("Rank").reset_index(drop=True)

    # KPI summary row
    n_buy = (pred_df["Rank"] <= 5).sum()
    n_hold = ((pred_df["Rank"] > 5) & (pred_df["Rank"] <= 15)).sum()
    n_sell = (pred_df["Rank"] > 15).sum()
    best = pred_df.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("target", "Top pick", best["Ticker"], f"{best['Predicted_Return_5D (%)']:+.2f}% dự báo", delta_positive=best['Predicted_Return_5D (%)'] >= 0)
    with k2: kpi_card("circle-check", "BUY signals", str(n_buy))
    with k3: kpi_card("circle-minus", "HOLD signals", str(n_hold))
    with k4: kpi_card("circle-x", "SELL signals", str(n_sell))

    st.markdown("<br>", unsafe_allow_html=True)
    pred_df_display = pred_df.copy()
    pred_df_display["Signal"] = pred_df_display["Rank"].apply(signal_badge)
    st.write(pred_df_display[["Rank", "Ticker", "Adj Close", "Predicted_Return_5D (%)", "Signal"]].to_html(escape=False, index=False), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    alert_box(
        "alert-triangle",
        "Con số dự báo trên là <b>lợi suất kỳ vọng do mô hình hồi quy (XGBoost) ước tính</b>, "
        "<b>không phải</b> quyết định mua/bán trực tiếp. Nhãn BUY/HOLD/SELL bên cạnh chỉ là quy tắc "
        "hiển thị dựa trên thứ hạng (Top 5 → BUY, 6–15 → HOLD, còn lại → SELL) để dễ theo dõi, "
        "không phải output của mô hình. Độ chính xác xếp hạng của mô hình (Information Coefficient) "
        "chỉ khoảng 0,01–0,03 — mức khá yếu về mặt thống kê, do dữ liệu đầu vào chỉ là giá và khối lượng "
        "giao dịch công khai (OHLCV), chưa bao gồm tin tức, tâm lý thị trường hay yếu tố vĩ mô. "
        "Vì vậy, kết quả này chỉ mang tính minh họa học thuật, <b>không nên dùng để ra quyết định đầu tư thực tế</b>.",
        "warning",
    )

    top10 = pred_df.head(10)
    fig = go.Figure(go.Bar(
        x=top10["Predicted_Return_5D (%)"], y=top10["Ticker"], orientation="h",
        marker_color=[COLORS["up"] if v > 0 else COLORS["down"] for v in top10["Predicted_Return_5D (%)"]],
    ))
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(title_text="Predicted Return 5D (%)")
    fig = plotly_layout(fig, height=420, hovermode="y")
    st.plotly_chart(fig, use_container_width=True)

# ===== DASHBOARD 3: RECOMMENDED PORTFOLIO =====
elif page == "Recommended Portfolio":
    title_with_icon("briefcase", "Recommended Portfolio", "Danh mục Top-K dựa trên dự báo XGBoost mới nhất (Equal Weight)")

    top_k = st.slider("Chọn Top-K", min_value=3, max_value=10, value=5)
    with st.spinner("Đang tính toán danh mục..."):
        X = latest_features[feature_cols].values
        X_scaled = scaler.transform(X)
        preds = xgb_model.predict(X_scaled)
        pred_df = latest_features[["Ticker", "Adj Close"]].copy()
        pred_df["pred_return"] = preds
        top_stocks = pred_df.nlargest(top_k, "pred_return").reset_index(drop=True)
        top_stocks["Weight (%)"] = round(100 / top_k, 2)

    k1, k2, k3 = st.columns(3)
    with k1: kpi_card("briefcase", "Số mã trong danh mục", str(top_k))
    with k2: kpi_card("trending-up", "Return dự báo TB (5D)", f"{top_stocks['pred_return'].mean()*100:+.2f}%",
                       delta_positive=top_stocks['pred_return'].mean() >= 0)
    with k3: kpi_card("target", "Trọng số mỗi mã", f"{100/top_k:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"Top-{top_k} cổ phiếu được chọn")
        st.dataframe(
            top_stocks[["Ticker", "Adj Close", "pred_return", "Weight (%)"]].rename(columns={"pred_return": "Predicted Return 5D"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Adj Close": st.column_config.NumberColumn("Adj Close ($)", format="$%.2f"),
                "Predicted Return 5D": st.column_config.NumberColumn("Predicted Return 5D", format="%.4f"),
                "Weight (%)": st.column_config.ProgressColumn("Weight (%)", min_value=0, max_value=100, format="%.1f%%"),
            },
        )
    with col2:
        fig = px.pie(top_stocks, values="Weight (%)", names="Ticker", hole=0.45,
                      color_discrete_sequence=PLOTLY_LINE_COLORS)
        fig.update_traces(textinfo="label+percent")
        fig = plotly_layout(fig, height=380, hovermode="closest")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    alert_box(
        "info",
        f"Danh mục gồm {top_k} cổ phiếu có lợi suất dự báo cao nhất, được phân bổ tỷ trọng đều nhau "
        f"(Equal Weight, mỗi mã {100/top_k:.1f}%). Danh mục được <b>tái cân bằng (rebalance) mỗi 5 ngày "
        "giao dịch</b> — con số này được chọn khớp đúng với khoảng thời gian mà mô hình dự báo lợi suất "
        "(future return 5 ngày tới). Lý do: sau 5 ngày, dự báo ban đầu coi như đã \"hết hạn\" vì mô hình "
        "không được huấn luyện để dự báo xa hơn khoảng thời gian đó, nên hệ thống cần tính toán lại dự báo "
        "mới và phân bổ lại danh mục cho phù hợp, tránh việc tiếp tục nắm giữ theo một dự báo đã cũ.",
        "info",
    )

# ===== DASHBOARD 4: BACKTEST =====
elif page == "Backtest":
    title_with_icon("bar-chart-3", "Backtest Results", "So sánh hiệu suất các chiến lược trên Test set (2022-2026)")

    metrics_table = pd.DataFrame({name: compute_metrics(rets) for name, rets in returns_dict.items()}).T
    best_sharpe_strategy = metrics_table["Sharpe Ratio"].idxmax()
    best_sharpe = metrics_table["Sharpe Ratio"].max()
    best_cagr_strategy = metrics_table["CAGR (%)"].idxmax()
    worst_dd = metrics_table["Max Drawdown (%)"].min()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("trophy", "Best Sharpe", f"{best_sharpe:.3f}", best_sharpe_strategy)
    with k2: kpi_card("trending-up", "Best CAGR", best_cagr_strategy, f"{metrics_table.loc[best_cagr_strategy, 'CAGR (%)']:.2f}%")
    with k3: kpi_card("trending-down", "Worst Drawdown", f"{worst_dd:.2f}%", delta_positive=False)
    with k4: kpi_card("bar-chart-3", "Chiến lược đã test", str(len(returns_dict)))

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Bảng so sánh Metrics")
    st.dataframe(metrics_table.style.format("{:.2f}").background_gradient(subset=["Sharpe Ratio"], cmap="Blues"), use_container_width=True)

    st.subheader("Cumulative Return")
    selected_strategies = st.multiselect("Chọn chiến lược để so sánh", list(returns_dict.keys()), default=list(returns_dict.keys()))
    fig = go.Figure()
    for name in selected_strategies:
        cum = (1 + returns_dict[name]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=name,
            line=dict(width=3 if name == "Buy & Hold" else 1.8),
        ))
    fig.update_yaxes(title_text="Portfolio Value (Start = 1.0)")
    fig = plotly_layout(fig, height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ablation Study — Đóng góp của từng nhóm feature")
    st.dataframe(ablation_df, use_container_width=True, hide_index=True)

    alert_box(
        "alert-triangle",
        "<b>Kết luận: Mục tiêu Sharpe Ratio ≥ 1.8 KHÔNG đạt được</b> bằng phương pháp trung thực. "
        "Chiến lược tốt nhất trên tập Test là Buy & Hold (Sharpe 1.068) — vượt qua toàn bộ các chiến lược "
        "chủ động dựa trên Machine Learning. Bốn nguyên nhân chính đã được xác định: "
        "(1) tín hiệu dự báo từ dữ liệu giá/khối lượng công khai còn khá yếu (chỉ số Information "
        "Coefficient cao nhất chỉ đạt 0,028, trong khi mức 0,03–0,05 mới được xem là tín hiệu tốt trong "
        "ngành); (2) việc tập trung danh mục vào 5/25 cổ phiếu làm tăng độ biến động (rủi ro) nhanh hơn "
        "mức tăng lợi nhuận mà tín hiệu dự báo còn yếu này mang lại; (3) đặc điểm biến động thị trường "
        "khác nhau giữa giai đoạn huấn luyện/kiểm định và giai đoạn kiểm thử (ví dụ giai đoạn có đại dịch "
        "COVID-19 so với giai đoạn lãi suất tăng) khiến chiến lược tối ưu trên dữ liệu quá khứ khó tổng "
        "quát hóa tốt; (4) kết quả backtest chưa tính đến chi phí giao dịch và trượt giá thực tế, vốn sẽ "
        "làm giảm thêm hiệu suất nếu tính đầy đủ. Toàn bộ kết quả trên được tính trực tiếp từ dữ liệu thực "
        "nghiệm, không có bất kỳ điều chỉnh nào nhằm làm đẹp số liệu.",
        "error",
    )
