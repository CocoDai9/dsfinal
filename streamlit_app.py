import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Mushroom Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #0f0f0f !important;
    color: #e8e8e8 !important;
}
[data-testid="stHeader"] { background: #0f0f0f !important; }

[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 1px solid #232323 !important;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #888 !important;
    cursor: pointer !important;
    transition: all .15s !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #1e1e1e !important;
    color: #fff !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #2a1515 !important;
    color: #e04040 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
    display: none !important;
}

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #252525 !important;
    background: transparent !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #555 !important;
    padding: 9px 18px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e04040 !important;
    border-bottom: 2px solid #e04040 !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #ccc !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

[data-testid="stMetric"] {
    background: #191919 !important;
    border: 1px solid #252525 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetric"] label {
    color: #777 !important;
    font-size: 12px !important;
}
[data-testid="stMetricValue"] {
    color: #fff !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

[data-testid="stSelectbox"] label {
    color: #888 !important;
    font-size: 12px !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #1a1a1a !important;
    border-color: #2e2e2e !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] span { color: #ddd !important; }

[data-testid="stSlider"] label {
    color: #888 !important;
    font-size: 12px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    background: #1a1a1a !important;
}

hr { border-color: #252525 !important; }

.sec-title {
    font-size: 24px;
    font-weight: 800;
    color: #fff;
    margin-top: .8rem;
    margin-bottom: 4px;
    letter-spacing: -.02em;
}
.sec-sub {
    font-size: 13px;
    color: #666;
    margin-bottom: 1.4rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("mushrooms_encodedcopy.csv")
    df["class_label"] = df["class"].map({0: "Edible", 1: "Poisonous"})
    return df


df = load_data()
df_clean = df.drop(columns=["veil-type", "class_label"], errors="ignore")

CARD = "#ffffff"
GRID = "rgba(200,200,200,0.35)"
TICK = "#555"
GREEN = "#1D9E75"
RED = "#D85A30"
BLUE = "#378ADD"
PURPLE = "#534AB7"
GREEN_A = "rgba(29,158,117,0.65)"
RED_A = "rgba(216,90,48,0.65)"
PUR_A = "rgba(83,74,183,0.75)"
PALETTE = [BLUE, "#E87722", GREEN, RED, PURPLE, "#888880"]
CFG = {"displayModeBar": False}


def card_layout(title="", xlab="", ylab="", height=380):
    return dict(
        title=dict(text=title, font=dict(size=12, color=TICK), x=0.5),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TICK, size=11),
        xaxis=dict(title=xlab, gridcolor=GRID, tickcolor=TICK, linecolor=GRID),
        yaxis=dict(title=ylab, gridcolor=GRID, tickcolor=TICK, linecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=60, r=20, t=46, b=55),
        height=height,
    )


def pie_layout(title="", height=280):
    return dict(
        title=dict(text=title, font=dict(size=12, color=TICK), x=0.5),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TICK, size=11),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=40),
        height=height,
    )


page = "Data Overview"

with st.sidebar:
    st.markdown(
        '<div style="padding:8px 4px 4px">'
        '<span style="font-size:16px;font-weight:800;color:#fff">Mushroom</span>'
        '<div style="font-size:11px;color:#555;margin-top:2px">Dataset Dashboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "nav",
        ["Data Overview", "Visualizations"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#444;line-height:1.9;padding:0 4px">'
        'Source: UCI Mushroom Dataset'
        '<br><br>'
        'Group members<br>'
        '<span style="color:#666">Coco Dai<br>Nadalia Jin<br>Solomon Kim</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if page == "Data Overview":
    st.markdown('<div class="sec-title">Data Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Summary statistics, data sample, missing values and column profiles '
        'for the UCI Mushroom dataset.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", f"{len(df.columns) - 1}")
    m3.metric("Edible", f"{(df['class'] == 0).sum():,}")
    m4.metric("Poisonous", f"{(df['class'] == 1).sum():,}")
    m5.metric("Missing values", f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Data sample")
        n_rows = st.slider("Rows to preview", 5, 50, 10, step=5)
        st.dataframe(
            df.drop(columns=["class_label"]).head(n_rows),
            use_container_width=True,
            height=min(40 + n_rows * 35, 480),
        )

    with right:
        st.markdown("#### Descriptive statistics")
        st.markdown("")
        st.markdown("---")
        desc = df_clean.describe().round(3)
        st.dataframe(desc, use_container_width=True, height=320)

    st.markdown("---")

    st.markdown("#### Missing values per column")
    missing = df.drop(columns=["class_label"]).isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    fig_miss = go.Figure(go.Bar(
        x=missing.index.tolist(),
        y=missing_pct.values.tolist(),
        marker=dict(
            color=[RED if v > 0 else GREEN for v in missing_pct.values],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in missing_pct.values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>",
    ))
    fig_miss.update_layout(**card_layout(
        title="Missing value rate (%) by column",
        xlab="Column",
        ylab="Missing %",
        height=320,
    ))
    fig_miss.update_layout(yaxis_range=[0, max(missing_pct.max() + 5, 10)])
    st.plotly_chart(fig_miss, use_container_width=True, config=CFG)

    st.markdown("---")

    # st.markdown("#### Column profiles")
    # raw = df.drop(columns=["class_label"])
    # profile_rows = []

    # for col in raw.columns:
    #     profile_rows.append({
    #         "Column": col,
    #         "Dtype": str(raw[col].dtype),
    #         "Unique": int(raw[col].nunique()),
    #         "Missing": int(raw[col].isnull().sum()),
    #         "Min": raw[col].min(),
    #         "Max": raw[col].max(),
    #         "Mean": round(raw[col].mean(), 3),
    #         "Std": round(raw[col].std(), 3),
    #     })

    # profile_df = pd.DataFrame(profile_rows).set_index("Column")

    # def highlight_profile(row):
    #     if row["Unique"] == 1:
    #         return ["background-color:#2e1010;color:#e04040"] * len(row)
    #     elif row["Missing"] > 0:
    #         return ["background-color:#2e1a0d;color:#E87722"] * len(row)
    #     return [""] * len(row)

    # styled = profile_df.style.apply(highlight_profile, axis=1)
    # st.dataframe(styled, use_container_width=True, height=460)

    # st.markdown(
    #     '<div style="font-size:11px;color:#444;margin-top:6px">'
    #     'Red row = zero-variance column (veil-type, excluded from visualizations) · '
    #     'Orange row = columns with missing values'
    #     '</div>',
    #     unsafe_allow_html=True,
    # )

    st.markdown("---")
    
else:
    st.markdown('<div class="sec-title">Visualizations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Explore relationships between variables using five chart types. '
        'All charts show edible vs poisonous split where applicable.</div>',
        unsafe_allow_html=True,
    )

    tab_uni, tab_scatter, tab_heatmap = st.tabs([
        "Univariate  (Pie · Bar · Histogram · Boxplot)",
        "Scatterplot",
        "Correlation Heatmap",
    ])

    with tab_uni:
        chart_type = st.radio(
            "Chart type",
            ["Pie Charts", "Bar Chart", "Histogram", "Boxplot"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("---")

        if chart_type == "Pie Charts":
            st.markdown(
                '<div class="sec-sub">Proportion of each category value across the dataset.</div>',
                unsafe_allow_html=True,
            )

            pie_feats = {
                "Class (edible vs poisonous)": (
                    lambda: (["Edible (0)", "Poisonous (1)"],
                             df["class"].value_counts().sort_index().values.tolist())),
                "Gill size": (
                    lambda: ([f"Value {i}" for i in df["gill-size"].value_counts().sort_index().index],
                             df["gill-size"].value_counts().sort_index().values.tolist())),
                "Bruises": (
                    lambda: (["No bruises (0)", "Bruises (1)"],
                             df["bruises"].value_counts().sort_index().values.tolist())),
                "Ring type": (
                    lambda: ([f"Type {i}" for i in df["ring-type"].value_counts().sort_index().index],
                             df["ring-type"].value_counts().sort_index().values.tolist())),
                "Cap shape": (
                    lambda: ([f"Shape {i}" for i in df["cap-shape"].value_counts().sort_index().index],
                             df["cap-shape"].value_counts().sort_index().values.tolist())),
                "Stalk root": (
                    lambda: ([f"Root {i}" for i in df["stalk-root"].value_counts().sort_index().index],
                             df["stalk-root"].value_counts().sort_index().values.tolist())),
            }

            def make_pie(labels, values, title):
                fig = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.42,
                    marker=dict(colors=PALETTE[:len(labels)], line=dict(color="#fff", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="#333"),
                    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
                ))
                fig.update_layout(**pie_layout(title=title, height=280))
                return fig

            keys = list(pie_feats.keys())
            r1 = st.columns(3)
            for i, col in enumerate(r1):
                lbl, val = pie_feats[keys[i]]()
                col.plotly_chart(make_pie(lbl, val, keys[i]), use_container_width=True, config=CFG)

            r2 = st.columns(3)
            for i, col in enumerate(r2):
                lbl, val = pie_feats[keys[i + 3]]()
                col.plotly_chart(make_pie(lbl, val, keys[i + 3]), use_container_width=True, config=CFG)

        elif chart_type == "Bar Chart":
            st.markdown(
                '<div class="sec-sub">Grouped bar chart — count of edible vs poisonous per feature value.</div>',
                unsafe_allow_html=True,
            )

            bar_feats = [c for c in df_clean.columns if c != "class"]
            bar_feat = st.selectbox("Select feature", bar_feats, key="bar_feat")

            fig = go.Figure()
            for cls, label, color in [(0, "Edible", GREEN_A), (1, "Poisonous", RED_A)]:
                counts = df[df["class"] == cls][bar_feat].value_counts().sort_index()
                fig.add_trace(go.Bar(
                    x=[str(v) for v in counts.index],
                    y=counts.values,
                    name=label,
                    marker_color=color,
                    hovertemplate=f"<b>{label}</b><br>Value: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
                ))

            fig.update_layout(**card_layout(
                title=f"{bar_feat} — edible vs poisonous",
                xlab=bar_feat,
                ylab="Count",
                height=420,
            ))
            fig.update_layout(barmode="group", bargap=0.2)
            st.plotly_chart(fig, use_container_width=True, config=CFG)

        elif chart_type == "Histogram":
            st.markdown(
                '<div class="sec-sub">Overlapping frequency distributions — purple = edible, coral = poisonous.</div>',
                unsafe_allow_html=True,
            )

            hist_feats = [c for c in df_clean.columns if c != "class"]
            hist_feat = st.selectbox("Select feature", hist_feats, key="hist_feat")

            fig = go.Figure()
            for cls, label, color in [(0, "Edible", PUR_A), (1, "Poisonous", RED_A)]:
                counts = df[df["class"] == cls][hist_feat].value_counts().sort_index()
                fig.add_trace(go.Bar(
                    x=[str(v) for v in counts.index],
                    y=counts.values,
                    name=label,
                    marker_color=color,
                    hovertemplate=f"<b>{label}</b><br>Value: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
                ))

            fig.update_layout(**card_layout(
                title=f"{hist_feat} — frequency by class",
                xlab=hist_feat,
                ylab="Count",
                height=420,
            ))
            fig.update_layout(barmode="overlay")
            st.plotly_chart(fig, use_container_width=True, config=CFG)

        else:
            st.markdown(
                '<div class="sec-sub">IQR and whiskers for multi-value features — edible vs poisonous.</div>',
                unsafe_allow_html=True,
            )

            box_feats = [c for c in df_clean.columns if c != "class" and df_clean[c].nunique() >= 3]
            box_feat = st.selectbox("Select feature", box_feats, key="box_feat")

            fig = go.Figure()
            for cls, label, color in [(0, "Edible", GREEN), (1, "Poisonous", RED)]:
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig.add_trace(go.Box(
                    y=df[df["class"] == cls][box_feat],
                    name=label,
                    marker_color=color,
                    boxmean="sd",
                    line=dict(width=2),
                    fillcolor=f"rgba({r},{g},{b},0.25)",
                    whiskerwidth=0.6,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Q1: %{q1:.2f}<br>Median: %{median:.2f}<br>"
                        "Q3: %{q3:.2f}<br>Min: %{lowerfence:.2f}<br>"
                        "Max: %{upperfence:.2f}<extra></extra>"
                    ),
                ))

            fig.update_layout(**card_layout(
                title=f"{box_feat} — spread by class",
                ylab=box_feat,
                height=440,
            ))
            fig.update_layout(boxmode="group")
            st.plotly_chart(fig, use_container_width=True, config=CFG)

    with tab_scatter:
        st.markdown(
            '<div class="sec-sub">Select any two features to explore their bivariate relationship, '
            'colored by mushroom class.</div>',
            unsafe_allow_html=True,
        )

        feat_options = [c for c in df_clean.columns if c != "class" and df_clean[c].var() > 0]

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            x_feat = st.selectbox("X axis", feat_options, index=feat_options.index("gill-size"), key="sc_x")
        with sc2:
            y_feat = st.selectbox("Y axis", feat_options, index=feat_options.index("stalk-root"), key="sc_y")
        with sc3:
            n_pts = st.slider("Sample size", 200, min(3000, len(df)), 800, step=100, key="sc_n")

        rng = np.random.default_rng(42)
        sample = df.sample(n_pts, random_state=42).copy()
        sample["_x"] = sample[x_feat] + rng.uniform(-0.22, 0.22, len(sample))
        sample["_y"] = sample[y_feat] + rng.uniform(-0.22, 0.22, len(sample))
        corr_val = df[[x_feat, y_feat]].corr().iloc[0, 1]

        fig = go.Figure()
        for cls, label, color, symbol in [
            (0, "Edible", GREEN_A, "circle"),
            (1, "Poisonous", RED_A, "triangle-up"),
        ]:
            sub = sample[sample["class"] == cls]
            fig.add_trace(go.Scatter(
                x=sub["_x"],
                y=sub["_y"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, symbol=symbol, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"{x_feat}: %{{x:.2f}}<br>"
                    f"{y_feat}: %{{y:.2f}}<extra></extra>"
                ),
            ))

        fig.update_layout(**card_layout(
            title=f"{x_feat} vs {y_feat} · Pearson r = {corr_val:.3f}",
            xlab=x_feat,
            ylab=y_feat,
            height=480,
        ))
        st.plotly_chart(fig, use_container_width=True, config=CFG)

    with tab_heatmap:
        st.markdown(
            '<div class="sec-sub">Pearson r between all features. '
            'Red = positive correlation, blue = negative. '
            'veil-type excluded (zero variance).</div>',
            unsafe_allow_html=True,
        )

        corr = df_clean.corr().round(3)
        cols = corr.columns.tolist()

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=cols,
            y=cols,
            colorscale=[
                [0.0, "#1a5fa8"],
                [0.3, "#6aaed6"],
                [0.5, "#f5f5f5"],
                [0.7, "#f4a582"],
                [1.0, "#c0182a"],
            ],
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=9),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                title=dict(text="r", font=dict(size=12)),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
                thickness=12,
                len=0.82,
            ),
        ))

        fig.update_layout(
            title=dict(text="Correlation Matrix Heatmap", font=dict(size=12, color=TICK), x=0.5),
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            font=dict(color=TICK, size=10),
            xaxis=dict(tickangle=-40, side="bottom", tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            height=540,
            margin=dict(l=120, r=50, t=50, b=110),
        )
        st.plotly_chart(fig, use_container_width=True, config=CFG)
