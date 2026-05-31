"""
A-Share Market Extreme RSI Monitor
大盘极值监控 Web 应用 — 基于 RSI6 历史极值分布判断当前市场位置
"""
import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="A股大盘极值监控",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
SYMBOL = "sh000001"
INDEX_NAME = "上证指数"
YEARS_BACK = 20
RSI_PERIOD = 6
DEFAULT_DISTANCE = 60
CACHE_TTL_SECONDS = 3600


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_index_data() -> pd.DataFrame:
    """Fetch Shanghai Composite Index daily data, going back ~20 years."""
    start_date = (datetime.now() - timedelta(days=YEARS_BACK * 365 + 10)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    try:
        df = ak.stock_zh_index_daily(symbol=SYMBOL)
    except Exception:
        try:
            df = ak.stock_zh_index_daily(symbol=SYMBOL, start_date=start_date, end_date=end_date)
        except Exception:
            df = ak.stock_zh_index_daily(symbol=SYMBOL, adjust="")

    if df is None or df.empty:
        raise RuntimeError("未能获取上证指数数据，请检查网络或 akshare 版本。")

    rename_map = {}
    for col in df.columns:
        low = str(col).lower().strip()
        if low in ("日期", "date"):
            rename_map[col] = "date"
        elif low in ("开盘", "open"):
            rename_map[col] = "open"
        elif low in ("最高", "high"):
            rename_map[col] = "high"
        elif low in ("最低", "low"):
            rename_map[col] = "low"
        elif low in ("收盘", "close"):
            rename_map[col] = "close"
        elif low in ("成交量", "volume"):
            rename_map[col] = "volume"
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=YEARS_BACK)
    df = df[df["date"] >= cutoff].copy()

    if df.empty:
        raise RuntimeError("过滤后数据为空，起始日期可能过近。")
    return df


# ═══════════════════════════════════════════════════════════════════
# RSI CALCULATION  (Wilder's smoothing)
# ═══════════════════════════════════════════════════════════════════
def calculate_rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= period:
        return rsi

    avg_gain = gain.iloc[1 : period + 1].mean()
    avg_loss = loss.iloc[1 : period + 1].mean()

    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

        if avg_loss == 0:
            rsi.iloc[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ═══════════════════════════════════════════════════════════════════
# PEAK / TROUGH DETECTION
# ═══════════════════════════════════════════════════════════════════
def detect_extremes(
    df: pd.DataFrame, distance: int, prominence: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect major peaks and troughs via scipy.signal.find_peaks."""
    close = df["close"].values
    dates = df["date"].values
    rsi = df["rsi6"].values

    if prominence is None:
        prominence = np.nanmedian(close) * 0.05

    peak_idx, _ = find_peaks(close, distance=distance, prominence=prominence)
    tops = pd.DataFrame({"date": dates[peak_idx], "close": close[peak_idx], "rsi6": rsi[peak_idx]})

    trough_idx, _ = find_peaks(-close, distance=distance, prominence=prominence)
    bottoms = pd.DataFrame({"date": dates[trough_idx], "close": close[trough_idx], "rsi6": rsi[trough_idx]})

    return tops, bottoms


# ═══════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════
def compute_stats(tops: pd.DataFrame, bottoms: pd.DataFrame) -> dict:
    """Summary statistics for tops and bottoms RSI6 distributions."""
    top_rsi = tops["rsi6"].dropna()
    bot_rsi = bottoms["rsi6"].dropna()

    def q(x, p):
        return x.quantile(p) if len(x) else np.nan

    return {
        "top_count": len(top_rsi),
        "top_mean": top_rsi.mean() if len(top_rsi) else np.nan,
        "top_median": top_rsi.median() if len(top_rsi) else np.nan,
        "top_p90": q(top_rsi, 0.90),
        "top_max": top_rsi.max() if len(top_rsi) else np.nan,
        "top_min": top_rsi.min() if len(top_rsi) else np.nan,
        "bot_count": len(bot_rsi),
        "bot_mean": bot_rsi.mean() if len(bot_rsi) else np.nan,
        "bot_median": bot_rsi.median() if len(bot_rsi) else np.nan,
        "bot_p10": q(bot_rsi, 0.10),
        "bot_min": bot_rsi.min() if len(bot_rsi) else np.nan,
        "bot_max": bot_rsi.max() if len(bot_rsi) else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════
# ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
def assess_current(current_rsi: float, stats: dict) -> tuple[str, str, str]:
    """Compare current RSI6 against historical extremes. Returns (label, color, detail)."""
    if np.isnan(current_rsi):
        return "数据不足", "gray", "无法计算当前 RSI6"

    bot_p10, bot_mean = stats["bot_p10"], stats["bot_mean"]
    top_mean, top_p90 = stats["top_mean"], stats["top_p90"]

    if not np.isnan(bot_p10) and current_rsi < bot_p10:
        return (
            "极度超卖 — 历史大底区间",
            "darkgreen",
            f"当前 RSI6 ({current_rsi:.1f}) 低于历史大底 10% 分位数 ({bot_p10:.1f})，属于极端超卖区域。",
        )
    if not np.isnan(bot_mean) and current_rsi < bot_mean:
        return (
            "超卖区间 — 逢低布局",
            "green",
            f"当前 RSI6 ({current_rsi:.1f}) 低于历史大底均值 ({bot_mean:.1f})，处于超卖区域。",
        )
    if not np.isnan(top_mean) and current_rsi < top_mean:
        return (
            "震荡区间",
            "orange",
            f"当前 RSI6 ({current_rsi:.1f}) 位于历史底部均值 ({bot_mean:.1f}) 与顶部均值 ({top_mean:.1f}) 之间。",
        )
    if not np.isnan(top_p90) and current_rsi < top_p90:
        return (
            "超买区间 — 注意风险",
            "red",
            f"当前 RSI6 ({current_rsi:.1f}) 高于历史大顶均值 ({top_mean:.1f})，市场处于超买区域。",
        )
    if not np.isnan(top_p90):
        return (
            "极度超买 — 历史大顶区间",
            "darkred",
            f"当前 RSI6 ({current_rsi:.1f}) 超过历史大顶 90% 分位数 ({top_p90:.1f})，属于极端超买区域。",
        )
    return "数据不足", "gray", "无法完成评估"


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════
def build_chart(
    df: pd.DataFrame, tops: pd.DataFrame, bottoms: pd.DataFrame, stats: dict
) -> go.Figure:
    """Dual-subplot Plotly figure: price + markers (top), RSI6 + reference lines (bottom)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.62, 0.38],
        subplot_titles=("上证指数 收盘价 & 历史极值标记", "RSI(6) 指标"),
    )

    # Price
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["close"], mode="lines", name="收盘价",
            line=dict(color="#1f77b4", width=1.2),
            hovertemplate="日期: %{x|%Y-%m-%d}<br>收盘: %{y:.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Tops
    if not tops.empty:
        fig.add_trace(
            go.Scatter(
                x=tops["date"], y=tops["close"], mode="markers", name="历史大顶",
                marker=dict(symbol="triangle-down", size=10, color="#dc3545", line=dict(width=1, color="#a71d2a")),
                customdata=tops["rsi6"],
                hovertemplate="<b>历史大顶</b><br>日期: %{x|%Y-%m-%d}<br>收盘: %{y:.0f}<br>RSI6: %{customdata:.1f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Bottoms
    if not bottoms.empty:
        fig.add_trace(
            go.Scatter(
                x=bottoms["date"], y=bottoms["close"], mode="markers", name="历史大底",
                marker=dict(symbol="triangle-up", size=10, color="#28a745", line=dict(width=1, color="#1c7430")),
                customdata=bottoms["rsi6"],
                hovertemplate="<b>历史大底</b><br>日期: %{x|%Y-%m-%d}<br>收盘: %{y:.0f}<br>RSI6: %{customdata:.1f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["rsi6"], mode="lines", name="RSI(6)",
            line=dict(color="#9467bd", width=1.2),
            hovertemplate="日期: %{x|%Y-%m-%d}<br>RSI6: %{y:.1f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Reference lines: 30 / 50 / 70
    for level, color, label in [
        (30, "#28a745", "超卖线 30"),
        (50, "#6c757d", "中线 50"),
        (70, "#dc3545", "超买线 70"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.4,
                      row=2, col=1, annotation_text=label, annotation_position="right")

    # Bottom RSI6 mean
    if not np.isnan(stats["bot_mean"]):
        fig.add_hline(y=stats["bot_mean"], line_dash="dot", line_color="#28a745", line_width=1.5,
                      row=2, col=1,
                      annotation_text=f"底部RSI均值 {stats['bot_mean']:.1f}",
                      annotation_position="right")

    # Top RSI6 mean
    if not np.isnan(stats["top_mean"]):
        fig.add_hline(y=stats["top_mean"], line_dash="dot", line_color="#dc3545", line_width=1.5,
                      row=2, col=1,
                      annotation_text=f"顶部RSI均值 {stats['top_mean']:.1f}",
                      annotation_position="right")

    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="收盘价", row=1, col=1)
    fig.update_yaxes(title_text="RSI(6)", range=[0, 100], row=2, col=1)
    fig.update_layout(
        height=700, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════
def status_card(close_val: float, rsi_val: float, assessment: str, color: str, detail: str):
    """Render the status card for Shanghai index."""
    bg_map = {
        "darkgreen": "#d4edda", "green": "#d4edda",
        "orange": "#fff3cd",
        "red": "#f8d7da", "darkred": "#f8d7da",
        "gray": "#e2e3e5",
    }
    border_map = {
        "darkgreen": "#28a745", "green": "#28a745",
        "orange": "#ffc107",
        "red": "#dc3545", "darkred": "#dc3545",
        "gray": "#6c757d",
    }

    level_map = {
        "darkgreen": "🟢 极度超卖 — 历史大底区间",
        "green": "🟢 超卖区间 — 逢低布局",
        "orange": "🟡 震荡区间",
        "red": "🔴 超买区间 — 注意风险",
        "darkred": "🔴 极度超买 — 历史大顶区间",
        "gray": "⚪ 数据不足",
    }

    st.markdown(
        f"""
        <div style="background-color:{bg_map.get(color, '#e2e3e5')};
                    border-left:5px solid {border_map.get(color, '#6c757d')};
                    padding:20px 24px; border-radius:8px; margin-bottom:12px;">
            <div style="font-size:1.3em;font-weight:bold;margin-bottom:8px;">上证指数 (000001)</div>
            <div style="display:flex;gap:36px;flex-wrap:wrap;">
                <div><span style="color:#555;">最新收盘价</span><br>
                     <b style="font-size:1.4em;">{close_val:.2f}</b></div>
                <div><span style="color:#555;">当前 RSI(6)</span><br>
                     <b style="font-size:1.4em;">{rsi_val:.1f}</b></div>
                <div style="flex:1;min-width:260px;">
                    <span style="color:#555;">量化评估</span><br>
                    <b style="font-size:1.2em;">{level_map.get(color, '')}</b>
                </div>
            </div>
            <div style="margin-top:10px;color:#444;font-size:0.92em;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stats_table(stats: dict) -> pd.DataFrame:
    """Build comparison dataframe for display."""
    def fmt(x):
        return f"{x:.1f}" if not np.isnan(x) else "N/A"

    rows = {
        "识别极值点数量": (stats["top_count"], stats["bot_count"]),
        "RSI6 均值": (fmt(stats["top_mean"]), fmt(stats["bot_mean"])),
        "RSI6 中位数": (fmt(stats["top_median"]), fmt(stats["bot_median"])),
        "RSI6 90%分位": (fmt(stats["top_p90"]), "—"),
        "RSI6 10%分位": ("—", fmt(stats["bot_p10"])),
        "RSI6 最大值": (fmt(stats["top_max"]), fmt(stats["bot_max"])),
        "RSI6 最小值": (fmt(stats["top_min"]), fmt(stats["bot_min"])),
    }
    data = [{"指标": k, "历史大顶 (RSI6)": v1, "历史大底 (RSI6)": v2} for k, (v1, v2) in rows.items()]
    return pd.DataFrame(data).set_index("指标")


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════
def main():
    st.title("上证指数 RSI(6) 历史极值监控")
    st.caption(f"回溯近 {YEARS_BACK} 年日线数据，识别阶段大顶/大底，量化当前位置。")

    # --- Sidebar ---
    st.sidebar.header("参数设置")
    distance = st.sidebar.slider(
        "波峰/波谷最小间距 (distance)",
        min_value=20, max_value=150, value=DEFAULT_DISTANCE, step=10,
        help="两个顶部/底部之间至少相隔多少个交易日。越大极值越少但越显著。",
    )
    prominence_pct = st.sidebar.slider(
        "显著性阈值 (prominence, 相对收盘价中位数的 %)",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="波段必须「突出」周围价格多少才算极值。百分比越高，找出的极值越少。",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "RSI(6) 使用 Wilder 平滑算法。\n"
        "阶段大顶/大底由 scipy find_peaks 识别。\n"
        "数据来源: akshare → 东方财富。"
    )

    # --- Data Loading ---
    with st.spinner(f"正在获取上证指数近 {YEARS_BACK} 年日线数据..."):
        try:
            df = fetch_index_data()
        except Exception as e:
            st.error(f"数据获取失败：{e}")
            st.info("请检查网络连接，或确认 akshare 已正确安装 (`pip install akshare`)。")
            return

    # --- Computation ---
    with st.spinner("正在计算 RSI(6) 并识别历史极值..."):
        df["rsi6"] = calculate_rsi_wilder(df["close"], RSI_PERIOD)
        prominence = np.nanmedian(df["close"].values) * (prominence_pct / 100.0)
        tops, bottoms = detect_extremes(df, distance=distance, prominence=prominence)
        stats = compute_stats(tops, bottoms)

        current_close = df["close"].iloc[-1]
        current_rsi = df["rsi6"].iloc[-1]
        assessment, color, detail = assess_current(current_rsi, stats)

    # --- Status Card ---
    st.subheader("当前状态总览")
    status_card(current_close, current_rsi, assessment, color, detail)

    # --- Stats Table ---
    st.subheader("历史极值统计对比")
    st.dataframe(stats_table(stats), use_container_width=True)

    # --- Extreme Date Tables ---
    col_t, col_b = st.columns(2)
    with col_t:
        st.markdown("**历史大顶**")
        if tops.empty:
            st.caption("未识别到符合条件的波峰，请调低 prominence 或 distance。")
        else:
            t = tops.sort_values("date", ascending=False).copy()
            t["date"] = t["date"].dt.strftime("%Y-%m-%d")
            t = t.rename(columns={"date": "日期", "close": "收盘价", "rsi6": "RSI6"}).round({"收盘价": 2, "RSI6": 1})
            st.dataframe(t.reset_index(drop=True), use_container_width=True, height=300)

    with col_b:
        st.markdown("**历史大底**")
        if bottoms.empty:
            st.caption("未识别到符合条件的波谷，请调低 prominence 或 distance。")
        else:
            b = bottoms.sort_values("date", ascending=False).copy()
            b["date"] = b["date"].dt.strftime("%Y-%m-%d")
            b = b.rename(columns={"date": "日期", "close": "收盘价", "rsi6": "RSI6"}).round({"收盘价": 2, "RSI6": 1})
            st.dataframe(b.reset_index(drop=True), use_container_width=True, height=300)

    # --- Chart ---
    st.subheader("历史走势 & 极值标记")
    fig = build_chart(df, tops, bottoms, stats)
    st.plotly_chart(fig, use_container_width=True)

    # --- Footer ---
    st.divider()
    st.caption(
        f"数据最后更新：{df['date'].iloc[-1].strftime('%Y-%m-%d')}  |  "
        f"回溯起始：{df['date'].iloc[0].strftime('%Y-%m-%d')}  |  "
        f"总交易日：{len(df)}  |  "
        f"历史大顶：{len(tops)} 个  |  历史大底：{len(bottoms)} 个"
    )


if __name__ == "__main__":
    main()
