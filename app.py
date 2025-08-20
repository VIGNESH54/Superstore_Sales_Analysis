import os
import io
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Optional imports (guarded)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False

# ------------------------- Page & Theme -------------------------
st.set_page_config(page_title="Superstore — Professional BI Dashboard", layout="wide")
st.markdown("<h1>📊 Superstore — Professional BI Dashboard</h1>", unsafe_allow_html=True)

# ------------------------- Helpers -------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Optional[str], uploaded_file) -> pd.DataFrame:
    """Load CSV (local path or uploaded file) with encoding fallbacks and tidy columns."""
    if uploaded_file is not None:
        df = None
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError("Could not read uploaded CSV. Try a different encoding.")
    else:
        if not path or not os.path.exists(path):
            raise FileNotFoundError("CSV not found. Upload a file or place it in data/sales_data.csv")
        df = None
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError("Could not read local CSV. Check file encoding.")

    # Minimal columns
    minimal = {"Order Date", "Order ID", "Sales", "Profit", "Product Name"}
    if not minimal.issubset(set(df.columns)):
        missing = minimal - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Parse/Coerce
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])
    for col in ["Sales", "Profit", "Discount", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Sales", "Profit"])
    if "Discount" not in df.columns:
        df["Discount"] = 0.0
    if "Quantity" not in df.columns:
        df["Quantity"] = 1

    # Derived
    df["YearMonth"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Profit Margin %"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, 0.0)
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def pct_delta(cur: float, prev: Optional[float]) -> Optional[float]:
    if prev in (None, 0) or prev is None or not np.isfinite(prev):
        return None
    return (cur - prev) / prev * 100.0


def month_series(fdf: pd.DataFrame) -> pd.DataFrame:
    return (
        fdf.groupby("YearMonth")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
        .sort_values("YearMonth")
    )


def previous_period(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    cur_start = pd.to_datetime(start_date)
    cur_end = pd.to_datetime(end_date)
    period_days = (cur_end - cur_start).days + 1
    prev_end = cur_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_days - 1)
    return df[(df["Order Date"] >= prev_start) & (df["Order Date"] <= prev_end)]


# ------------------------- Sidebar: Data & Filters -------------------------
st.sidebar.header("Data Upload")
uploaded = st.sidebar.file_uploader("Upload Superstore CSV", type=["csv"])
default_path = os.path.join("data", "sales_data.csv")

try:
    df = load_data(default_path, uploaded)
except Exception as e:
    st.sidebar.error(f"Data load error: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows")

st.sidebar.header("Filters")
min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
date_input = st.sidebar.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_input, tuple):
    start_date, end_date = date_input
else:
    start_date, end_date = date_input, date_input  # single date fallback

def opts(col: str) -> list:
    return sorted(df[col].dropna().unique()) if col in df.columns else []

regions   = st.sidebar.multiselect("Region",   options=opts("Region"),   default=opts("Region"))
categories= st.sidebar.multiselect("Category", options=opts("Category"), default=opts("Category"))
segments  = st.sidebar.multiselect("Segment",  options=opts("Segment"),  default=opts("Segment"))
ships     = st.sidebar.multiselect("Ship Mode",options=opts("Ship Mode"),default=opts("Ship Mode"))

mask = (df["Order Date"].dt.date >= start_date) & (df["Order Date"].dt.date <= end_date)
if regions:    mask &= df["Region"].isin(regions)
if categories: mask &= df["Category"].isin(categories)
if segments:   mask &= df["Segment"].isin(segments)
if ships:      mask &= df["Ship Mode"].isin(ships)

fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("No data for the selected filters. Adjust filters on the left.")
    st.stop()

# ------------------------- Tabs -------------------------
tabs = st.tabs(["Overview", "Forecasting", "Customer Analytics", "Market Basket", "Geo Analysis", "Reports"])

# ========================= OVERVIEW =========================
with tabs[0]:
    st.markdown("### Overview")

    # KPIs
    sales   = float(fdf["Sales"].sum())
    profit  = float(fdf["Profit"].sum())
    orders  = int(fdf["Order ID"].nunique() if "Order ID" in fdf.columns else fdf.shape[0])
    pmargin = float((profit / sales * 100) if sales > 0 else 0.0)

    prev = previous_period(df, start_date, end_date)
    sales_p   = float(prev["Sales"].sum()) if not prev.empty else None
    profit_p  = float(prev["Profit"].sum()) if not prev.empty else None
    orders_p  = int(prev["Order ID"].nunique()) if ("Order ID" in prev.columns and not prev.empty) else None
    pmargin_p = (profit_p / sales_p * 100.0) if sales_p and sales_p > 0 else None

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💰 Total Sales", f"${sales:,.0f}",   f"{pct_delta(sales, sales_p):+.1f}%" if pct_delta(sales, sales_p) is not None else None)
    c2.metric("📈 Total Profit", f"${profit:,.0f}", f"{pct_delta(profit, profit_p):+.1f}%" if pct_delta(profit, profit_p) is not None else None)
    c3.metric("🛒 Unique Orders", f"{orders:,}",   f"{pct_delta(orders, orders_p):+.1f}%" if pct_delta(orders, orders_p) is not None else None)
    c4.metric("📊 Profit Margin", f"{pmargin:.2f}%", f"{pct_delta(pmargin, pmargin_p):+.1f}%" if pct_delta(pmargin, pmargin_p) is not None else None)

    st.divider()

    # Monthly trends
    ts = month_series(fdf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["YearMonth"], y=ts["Sales"],  mode="lines+markers", name="Sales"))
    fig.add_trace(go.Scatter(x=ts["YearMonth"], y=ts["Profit"], mode="lines+markers", name="Profit"))
    ts["Sales_MA3"] = ts["Sales"].rolling(3).mean()
    fig.add_trace(go.Scatter(x=ts["YearMonth"], y=ts["Sales_MA3"], mode="lines", name="Sales MA(3)", line=dict(dash="dash")))
    fig.update_layout(height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        reg = fdf.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False) if "Region" in fdf.columns else pd.DataFrame()
        if not reg.empty:
            st.write("#### 🌍 Sales by Region")
            st.plotly_chart(px.bar(reg, x="Region", y="Sales", text_auto=".2s", height=380), use_container_width=True)
        else:
            st.info("No 'Region' column available.")

    with colB:
        if "Category" in fdf.columns and "Sub-Category" in fdf.columns:
            st.write("#### 🧭 Category → Sub-Category (Treemap)")
            tree = fdf.groupby(["Category", "Sub-Category"], as_index=False)["Sales"].sum()
            st.plotly_chart(px.treemap(tree, path=["Category","Sub-Category"], values="Sales", height=380), use_container_width=True)
        else:
            st.info("Treemap requires 'Category' and 'Sub-Category'.")

    st.divider()

    # Product leaderboard + ABC
    st.write("#### 🏆 Products Leaderboard & ABC (Pareto)")
    top_n = st.slider("Top N products", 5, 30, 10, key="topn_overview")
    prod_agg = (
        fdf.groupby("Product Name", as_index=False)
           .agg(Sales=("Sales","sum"), Profit=("Profit","sum"), Orders=("Order ID","nunique"), Qty=("Quantity","sum"))
           .sort_values("Sales", ascending=False)
    )
    st.plotly_chart(px.bar(prod_agg.head(top_n), x="Product Name", y="Sales", height=360), use_container_width=True)

    if not prod_agg.empty and prod_agg["Sales"].sum() > 0:
        abc = prod_agg.copy().reset_index(drop=True)
        abc["Sales Share %"] = abc["Sales"] / abc["Sales"].sum() * 100
        abc["Cumulative %"] = abc["Sales Share %"].cumsum()
        def _lab(x):
            if x <= 80: return "A"
            if x <= 95: return "B"
            return "C"
        abc["Class"] = abc["Cumulative %"].apply(_lab)
        colL, colR = st.columns([2,1])
        with colL:
            # Pareto
            fig2 = go.Figure()
            xrank = list(range(1, len(abc)+1))
            fig2.add_trace(go.Bar(x=xrank, y=abc["Sales"], name="Sales"))
            fig2.add_trace(go.Scatter(x=xrank, y=abc["Cumulative %"], name="Cumulative %", yaxis="y2"))
            fig2.update_layout(
                height=420,
                yaxis=dict(title="Sales"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0,100]),
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig2, use_container_width=True)
        with colR:
            counts = abc["Class"].value_counts().reindex(["A","B","C"]).fillna(0).astype(int)
            share  = (abc.groupby("Class")["Sales"].sum() / abc["Sales"].sum() * 100).reindex(["A","B","C"]).round(2)
            st.write(pd.DataFrame({"Count":counts, "Sales Share %":share}))
            st.download_button("⬇️ Download ABC CSV", data=to_csv_bytes(abc), file_name="abc_products.csv", mime="text/csv")
    else:
        st.info("ABC analysis not available (no sales).")


# ========================= FORECASTING =========================
with tabs[1]:
    st.markdown("### 🔮 Forecasting")
    st.caption("Uses Prophet if available, otherwise a simple moving-average fallback.")
    use_prophet = PROPHET_AVAILABLE and st.checkbox("Use Prophet (recommended)", value=PROPHET_AVAILABLE)
    horizon = st.slider("Forecast horizon (months)", 1, 12, 6)

    monthly = month_series(fdf)
    if st.button("Run Forecast", key="run_fc"):
        if monthly.empty:
            st.info("Not enough data to forecast.")
        else:
            ts_df = monthly.rename(columns={"YearMonth":"ds", "Sales":"y"})[["ds","y"]]
            ts_df["ds"] = pd.to_datetime(ts_df["ds"])
            if use_prophet:
                try:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(ts_df)
                    future = m.make_future_dataframe(periods=horizon, freq="M")
                    fcst = m.predict(future)[["ds","yhat"]].rename(columns={"ds":"YearMonth","yhat":"Forecast"})
                    fcst["YearMonth"] = pd.to_datetime(fcst["YearMonth"]).dt.to_period("M").dt.to_timestamp()
                except Exception as e:
                    st.warning(f"Prophet failed ({e}). Falling back to MA(3).")
                    use_prophet = False

            if not use_prophet:
                s = ts_df.set_index("ds").resample("M").sum().y
                ma = s.rolling(3, min_periods=1).mean()
                last = ma.iloc[-1] if len(ma) else s.mean()
                future_idx = pd.date_range(s.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="M")
                fcst = pd.DataFrame({"YearMonth":future_idx, "Forecast":[float(last)]*len(future_idx)})

            combined = pd.concat([monthly.set_index("YearMonth")[["Sales"]],
                                  fcst.set_index("YearMonth")[["Forecast"]]], axis=1).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined["YearMonth"], y=combined["Sales"], mode="lines+markers", name="Actual Sales"))
            fig.add_trace(go.Scatter(x=combined["YearMonth"], y=combined["Forecast"], mode="lines+markers", name="Forecast"))
            fig.update_layout(height=440, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download Forecast CSV", data=to_csv_bytes(combined), file_name="sales_forecast.csv", mime="text/csv")


# ========================= CUSTOMER ANALYTICS =========================
with tabs[2]:
    st.markdown("### 👥 Customer Analytics — RFM")

    if "Customer ID" not in fdf.columns:
        st.info("Customer-level analytics requires 'Customer ID'.")
    else:
        ref_date = fdf["Order Date"].max() + pd.Timedelta(days=1)
        cust = fdf.groupby("Customer ID").agg(
            RecencyDays=("Order Date", lambda x: (ref_date - x.max()).days),
            Frequency=("Order ID", "nunique") if "Order ID" in fdf.columns else ("Order Date", "count"),
            Monetary=("Sales", "sum"),
        ).reset_index()

        # R, F, M quartiles
        cust["R_quartile"] = pd.qcut(cust["RecencyDays"].rank(method="first"), 4, labels=[4,3,2,1]).astype(int)
        cust["F_quartile"] = pd.qcut(cust["Frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
        cust["M_quartile"] = pd.qcut(cust["Monetary"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
        cust["RFM_Score"] = cust["R_quartile"]*100 + cust["F_quartile"]*10 + cust["M_quartile"]

        def rfm_segment(score):
            if score >= 444: return "Champions"
            if score >= 344: return "Loyal"
            if score >= 244: return "At Risk"
            return "Need Attention"

        cust["Segment"] = cust["RFM_Score"].apply(rfm_segment)

        col1, col2 = st.columns([2,1])
        with col1:
            st.write("Top 10 customers by Monetary")
            st.dataframe(cust.sort_values("Monetary", ascending=False).head(10), use_container_width=True)
        with col2:
            seg_counts = cust["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "count"]
            st.plotly_chart(px.bar(seg_counts, x="Segment", y="count", height=360), use_container_width=True)

        st.download_button("⬇️ Download RFM CSV", data=to_csv_bytes(cust), file_name="rfm_customers.csv", mime="text/csv")


# ========================= MARKET BASKET =========================
with tabs[3]:
    st.markdown("### 🛒 Market Basket Analysis")

    if not MLXTEND_AVAILABLE:
        st.warning("Install `mlxtend` to enable Market Basket: `pip install mlxtend`")
    else:
        st.caption("Tips: Lower support if you see no rules. Superstore baskets are sparse.")
        min_support = st.slider("Minimum Support", 0.001, 0.05, 0.01, 0.001)
        min_conf    = st.slider("Minimum Confidence", 0.05, 1.0, 0.3, 0.05)
        top_rules   = st.slider("Show Top N Rules", 5, 50, 15, 5)

        # Build basket (OrderID × ProductName)
        if "Order ID" not in fdf.columns:
            st.info("Apriori needs 'Order ID' and 'Product Name'.")
        else:
            basket = (
                fdf.groupby(["Order ID","Product Name"])["Quantity"]
                   .sum().unstack().fillna(0)
            )
            # binarize
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)

            if basket.empty or basket.values.sum() == 0:
                st.warning("⚠️ No co-purchases found for current filters. Try expanding the date range or lowering support.")
            else:
                # Apriori
                frequent_items = apriori(basket, min_support=min_support, use_colnames=True)

                if frequent_items.empty:
                    st.info("ℹ️ No frequent itemsets at this support. Lower the threshold.")
                else:
                    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_conf)
                    if rules.empty:
                        st.info("ℹ️ No rules at this confidence. Lower the confidence threshold.")
                    else:
                        rules = rules.sort_values(["lift","confidence","support"], ascending=False)
                        st.write("**Top Rules**")
                        st.dataframe(rules.head(top_rules), use_container_width=True)
                        st.download_button("⬇️ Download Rules CSV", data=to_csv_bytes(rules), file_name="market_basket_rules.csv", mime="text/csv")

                        # Optional network graph
                        if NETWORKX_AVAILABLE and st.checkbox("Show Rules Network Graph", value=True):
                            G = nx.DiGraph()
                            # Convert frozenset to list for iteration
                            def _iter_items(s):
                                return list(s) if isinstance(s, (set, frozenset)) else [s]

                            for _, r in rules.head(25).iterrows():
                                for a in _iter_items(r["antecedents"]):
                                    for c in _iter_items(r["consequents"]):
                                        G.add_edge(str(a), str(c), weight=float(r["lift"]))

                            if len(G.nodes) > 0:
                                pos = nx.spring_layout(G, seed=42, k=0.6)
                                edge_x, edge_y = [], []
                                for u,v in G.edges():
                                    x0,y0 = pos[u]; x1,y1 = pos[v]
                                    edge_x.extend([x0,x1,None]); edge_y.extend([y0,y1,None])
                                node_x, node_y, labels = [], [], []
                                for n in G.nodes():
                                    x,y = pos[n]; node_x.append(x); node_y.append(y); labels.append(n)

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                                         line=dict(width=1, color="lightgray"), hoverinfo="none"))
                                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                                         text=labels, textposition="top center",
                                                         marker=dict(size=16)))
                                fig.update_layout(height=600, showlegend=False, title="Association Rules Network")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Not enough nodes to render a network.")
                        elif not NETWORKX_AVAILABLE:
                            st.info("Install `networkx` for the rules network: `pip install networkx`")


# ========================= GEO ANALYSIS =========================
with tabs[4]:
    st.markdown("### 🌍 Geo Analysis")

    STATE_TO_CODE = {
        "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO","connecticut":"CT",
        "delaware":"DE","district of columbia":"DC","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID","illinois":"IL",
        "indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA",
        "michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV",
        "new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND",
        "ohio":"OH","oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
        "tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY"
    }

    if "State" not in fdf.columns:
        st.info("Geo-analysis requires a 'State' column with U.S. state names.")
    else:
        state_sales = fdf.groupby("State", as_index=False)["Sales"].sum()
        state_sales["state_code"] = state_sales["State"].str.lower().str.strip().map(STATE_TO_CODE)
        state_sales = state_sales.dropna(subset=["state_code"])
        if state_sales.empty:
            st.info("Could not map any states. Ensure U.S. state names (e.g., 'California').")
        else:
            fig = px.choropleth(
                state_sales, locations="state_code", locationmode="USA-states",
                color="Sales", color_continuous_scale="Blues", scope="usa", labels={"Sales":"Sales"}
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)


# ========================= REPORTS =========================
with tabs[5]:
    st.markdown("### 🧾 Reports — Excel & PDF")

    monthly = month_series(fdf)
    prod_agg = (
        fdf.groupby("Product Name", as_index=False)
           .agg(Sales=("Sales","sum"), Profit=("Profit","sum"), Orders=("Order ID","nunique"), Qty=("Quantity","sum"))
           .sort_values("Sales", ascending=False)
    )

    def create_excel_report() -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            kpis = pd.DataFrame({
                "Metric":["Total Sales","Total Profit","Unique Orders","Profit Margin %"],
                "Value":[sales, profit, orders, pmargin],
            })
            kpis.to_excel(writer, sheet_name="KPIs", index=False)
            monthly.to_excel(writer, sheet_name="Monthly", index=False)
            prod_agg.to_excel(writer, sheet_name="Products", index=False)
            if "Customer ID" in fdf.columns:
                # regenerate RFM quickly
                ref_date = fdf["Order Date"].max() + pd.Timedelta(days=1)
                cust = fdf.groupby("Customer ID").agg(
                    RecencyDays=("Order Date", lambda x: (ref_date - x.max()).days),
                    Frequency=("Order ID", "nunique") if "Order ID" in fdf.columns else ("Order Date","count"),
                    Monetary=("Sales","sum"),
                ).reset_index()
                cust.to_excel(writer, sheet_name="RFM", index=False)
        buf.seek(0)
        return buf.read()

    def create_pdf_summary() -> bytes:
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))  # A4 portrait
        # KPIs
        txt = f"Total Sales: ${sales:,.0f}\nTotal Profit: ${profit:,.0f}\nOrders: {orders:,}\nProfit Margin: {pmargin:.2f}%"
        axes[0,0].axis("off"); axes[0,0].text(0.02, 0.5, txt, fontsize=12, va="center")

        # Monthly Sales
        axes[0,1].plot(monthly["YearMonth"], monthly["Sales"], marker="o")
        axes[0,1].set_title("Monthly Sales")

        # Top products
        top5 = prod_agg.head(5)
        axes[1,0].barh(top5["Product Name"].astype(str), top5["Sales"]); axes[1,0].invert_yaxis()
        axes[1,0].set_title("Top 5 Products by Sales")

        # Sales share pie
        if not prod_agg.empty and prod_agg["Sales"].sum() > 0:
            share = (top5["Sales"] / top5["Sales"].sum()).values
            axes[1,1].pie(share, labels=top5["Product Name"], autopct="%1.1f%%")
            axes[1,1].set_title("Top 5 Sales Share")
        else:
            axes[1,1].axis("off"); axes[1,1].text(0.5,0.5,"No data", ha="center")

        plt.tight_layout()
        out = io.BytesIO()
        plt.savefig(out, format="pdf"); plt.close(fig); out.seek(0)
        return out.read()

    colx, coly = st.columns(2)
    with colx:
        st.download_button("⬇️ Download Excel Report", data=create_excel_report(),
                           file_name="superstore_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with coly:
        st.download_button("⬇️ Download PDF Summary", data=create_pdf_summary(),
                           file_name="superstore_summary.pdf", mime="application/pdf")

    st.caption("Reports reflect the **current filters** in the left sidebar.")
