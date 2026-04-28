import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mushroom Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────
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

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #252525 !important;
    gap: 16px !important;
    flex-wrap: wrap !important;
}

[data-testid="stTabs"] [role="tab"] {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #666 !important;
    padding: 10px 6px !important;
    margin-right: 6px !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e04040 !important;
}

.sec-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin-top: .5rem;
    margin-bottom: 4px;
    letter-spacing: -.02em;
}
.sec-sub {
    font-size: 14px;
    color: #8d8d8d;
    margin-bottom: 1.2rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = pd.read_csv("mushrooms_encodedcopy.csv")
    data["class_label"] = data["class"].map({0: "Edible", 1: "Poisonous"})
    return data

df = load_data()
df_clean = df.drop(columns=["veil-type", "class_label"], errors="ignore")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 4px">
            <span style="font-size:18px;font-weight:800;color:#fff">Mushroom Dashboard</span>
            <div style="font-size:11px;color:#666;margin-top:2px">Final Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    
    # --- FORM START ---
    with st.form("model_config_form"):
        st.subheader("Model Configuration")
        all_features = [col for col in df_clean.columns if col != "class"]
        
        # This will now wait for the button click
        selected_features = st.multiselect(
            "Select Features for Training",
            options=all_features,
            default=all_features
        )
        
        # The recalculate button
        submit_button = st.form_submit_button("🚀 Recalculate Model")
    # --- FORM END ---

    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature.")
        st.stop() # Stops the app here so it doesn't crash during training

    st.markdown("---")
    page = st.radio("Navigation", ["Data Overview","Visualizations","Model Predictions","Feature Importance","Hyperparameter Tuning"])
    st.markdown("---")


    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:11px;color:#555;line-height:1.9;padding:0 4px">
            Source: UCI Mushroom Dataset
            <br><br>
            Group Members<br>
            <span style="color:#777">Coco Dai<br>Nadalia Jin<br>Solomon Kim</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# CACHED MODEL TRAINING
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df_input, selected_cols):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, 
        f1_score, 
        confusion_matrix, 
        precision_score, 
        recall_score, 
    )

    # Data
    # Only use columns chosen by the user
    X = df_input[selected_cols] 
    y = df_input["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models_and_params = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=5000),
            'params': {'C': [0.1, 1.0], 'solver': ['lbfgs']},
            'scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [None, 10]},
            'scaled': False
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [5, 11]},
            'scaled': True
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [100], 'max_depth': [3]},
            'scaled': False
        },
        'MLP': {
            'model': MLPClassifier(max_iter=500, random_state=42),
            'params': {'hidden_layer_sizes': [(100,)]},
            'scaled': True
        }
    }

    trained_models = {}
    results = {}

    for name, mp in models_and_params.items():
            if mp["scaled"]:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            grid = GridSearchCV(mp["model"], mp["params"], cv=3, n_jobs=-1)
            grid.fit(X_tr, y_train)

            best_model = grid.best_estimator_
            preds = best_model.predict(X_te)

            trained_models[name] = best_model

            # --- MOVE THIS INSIDE THE LOOP (INDENT IT) ---
            results[name] = {
                "accuracy": round(accuracy_score(y_test, preds) * 100, 2),
                "f1": round(f1_score(y_test, preds, average="macro") * 100, 2),
                "recall": round(recall_score(y_test, preds) * 100, 2),
                "precision": round(precision_score(y_test, preds) * 100, 2),
                "cm": confusion_matrix(y_test, preds),
                "best_params": grid.best_params_
            }
            # ---------------------------------------------

    return trained_models, results, scaler, X


trained_models, results, scaler, X = train_models(df_clean, selected_features)



# ─────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────
label_map = {
    "class": {
        0: "e = edible",
        1: "p = poisonous",
    },
    "cap-shape": {
        0: "b = bell",
        1: "c = conical",
        2: "x = convex",
        3: "f = flat",
        4: "k = knobbed",
        5: "s = sunken",
    },
    "cap-surface": {
        0: "f = fibrous",
        1: "g = grooves",
        2: "y = scaly",
        3: "s = smooth",
    },
    "cap-color": {
        0: "n = brown",
        1: "b = buff",
        2: "c = cinnamon",
        3: "g = gray",
        4: "r = green",
        5: "p = pink",
        6: "u = purple",
        7: "e = red",
        8: "w = white",
        9: "y = yellow",
    },
    "bruises": {
        0: "f = no",
        1: "t = bruises",
    },
    "odor": {
        0: "a = almond",
        1: "l = anise",
        2: "c = creosote",
        3: "y = fishy",
        4: "f = foul",
        5: "m = musty",
        6: "n = none",
        7: "p = pungent",
        8: "s = spicy",
    },
    "gill-attachment": {
        0: "a = attached",
        1: "d = descending",
        2: "f = free",
        3: "n = notched",
    },
    "gill-spacing": {
        0: "c = close",
        1: "w = crowded",
        2: "d = distant",
    },
    "gill-size": {
        0: "b = broad",
        1: "n = narrow",
    },
    "gill-color": {
        0: "b = buff",
        1: "e = red",
        2: "g = gray",
        3: "h = chocolate",
        4: "k = black",
        5: "n = brown",
        6: "o = orange",
        7: "p = pink",
        8: "r = green",
        9: "u = purple",
        10: "w = white",
        11: "y = yellow",
    },
    "stalk-color-above-ring": {
        0: "b = buff",
        1: "c = cinnamon",
        2: "e = red",
        3: "g = gray",
        4: "n = brown",
        5: "o = orange",
        6: "p = pink",
        7: "w = white",
        8: "y = yellow",
    },
    "stalk-color-below-ring": {
        0: "b = buff",
        1: "c = cinnamon",
        2: "e = red",
        3: "g = gray",
        4: "n = brown",
        5: "o = orange",
        6: "p = pink",
        7: "w = white",
        8: "y = yellow",
    },
    "veil-type": {
        0: "p = partial",
    },
    "veil-color": {
        0: "n = brown",
        1: "o = orange",
        2: "w = white",
        3: "y = yellow",
    },
    "ring-number": {
        0: "n = none",
        1: "o = one",
        2: "t = two",
    },
    "ring-type": {
        0: "e = evanescent",
        1: "f = flaring",
        2: "l = large",
        3: "n = none",
        4: "p = pendant",
    },
    "spore-print-color": {
        0: "b = buff",
        1: "h = chocolate",
        2: "k = black",
        3: "n = brown",
        4: "o = orange",
        5: "r = green",
        6: "u = purple",
        7: "w = white",
        8: "y = yellow",
    },
}

# ─────────────────────────────────────────────────────────
# PLOTLY SETTINGS
# ─────────────────────────────────────────────────────────
CARD = "#ffffff"
GRID = "rgba(200,200,200,0.35)"
TICK = "#555"
GREEN = "#1D9E75"
RED = "#D85A30"
GREEN_A = "rgba(29,158,117,0.65)"
RED_A = "rgba(216,90,48,0.65)"
CFG = {"displayModeBar": False}

def card_layout(title="", xlab="", ylab="", height=420):
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



# THE REFRESH BUTTON
    # ─────────────────────────────────────────────────────
    if st.button("🔄 Retrain & Refresh"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:11px;color:#555;line-height:1.9;padding:0 4px">
            Source: UCI Mushroom Dataset
            <br><br>
            Group Members<br>
            <span style="color:#777">Coco Dai<br>Nadalia Jin<br>Solomon Kim</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# DATA OVERVIEW
# ─────────────────────────────────────────────────────────
if page == "Data Overview":
    st.markdown('<div class="sec-title">Data Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Summary statistics, data sample, and missing values for the UCI Mushroom dataset.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", f"{len(df.columns) - 1}")
    m3.metric("Edible", f"{(df['class'] == 0).sum():,}")
    m4.metric("Poisonous", f"{(df['class'] == 1).sum():,}")
    m5.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Data Sample")
        n_rows = st.slider("Rows to preview", 5, 50, 10, step=5)
        st.dataframe(
            df.drop(columns=["class_label"]).head(n_rows),
            use_container_width=True,
            height=min(40 + n_rows * 35, 480),
        )

    with right:
        st.subheader("Descriptive Statistics")
        st.markdown("")
        st.markdown("---")
        desc = df_clean.describe().round(3)
        st.dataframe(desc, use_container_width=True, height=320)

    st.markdown("---")

    st.subheader("Missing Values per Column")
    missing = df.drop(columns=["class_label"]).isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=missing, x="Column", y="Missing Count", hue="Column", palette="viridis", legend=False, ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.caption("The dataset contains no missing values across all recorded features.")

# ─────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────
elif page == "Visualizations":
    st.markdown('<div class="sec-title">Visualizations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Explore patterns in the mushroom dataset through five core chart types.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Label Guide: what the encoded numbers mean"):
        st.markdown("""
        The dataset was label-encoded, so values such as 0, 1, 2, and 3 represent original category labels.

        **Common variables used in this dashboard**
        - `class`: 0 = edible, 1 = poisonous
        - `bruises`: 0 = no, 1 = bruises
        - `gill-size`: 0 = broad, 1 = narrow
        """)

        for feature, mapping in label_map.items():
            st.markdown(f"**{feature}**")
            mapping_text = "  \n".join([f"- {k}: {v}" for k, v in mapping.items()])
            st.markdown(mapping_text)

    tab_pie, tab_bar, tab_hist, tab_scatter, tab_heatmap = st.tabs([
        "Pie Charts",
        "Bar Chart",
        "Histogram",
        "Scatterplot",
        "Correlation Heatmap"
    ])

    # ── PIE CHARTS ─────────────────────────────────────
    with tab_pie:
        st.subheader("Category Distribution")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        df["class"].value_counts().sort_index().plot.pie(
            autopct="%1.1f%%",
            ax=axes[0],
            colors=["#66c2a5", "#fc8d62"]
        )
        axes[0].set_title("Class Distribution")
        axes[0].set_ylabel("")

        df["bruises"].value_counts().sort_index().plot.pie(
            autopct="%1.1f%%",
            ax=axes[1],
            colors=["#8da0cb", "#e78ac3"]
        )
        axes[1].set_title("Bruises Distribution")
        axes[1].set_ylabel("")

        df["gill-size"].value_counts().sort_index().plot.pie(
            autopct="%1.1f%%",
            ax=axes[2],
            colors=["#a6d854", "#ffd92f"]
        )
        axes[2].set_title("Gill Size Distribution")
        axes[2].set_ylabel("")

        st.pyplot(fig)

        with st.expander("Label guide for pie chart variables"):
            for feature in ["class", "bruises", "gill-size"]:
                if feature in label_map:
                    st.markdown(f"**{feature}**")
                    for k, v in label_map[feature].items():
                        st.write(f"{k}: {v}")

    # ── BAR CHART ──────────────────────────────────────
    with tab_bar:
        st.subheader("Feature Count by Class")

        feature = st.selectbox(
            "Select a feature",
            [c for c in df_clean.columns if c != "class"],
            key="bar_feature"
        )

        if feature in label_map:
            with st.expander(f"Label guide for {feature}"):
                for k, v in label_map[feature].items():
                    st.write(f"{k}: {v}")

        with st.expander("Label guide for class"):
            for k, v in label_map["class"].items():
                st.write(f"{k}: {v}")

        count_df = df.groupby([feature, "class"]).size().reset_index(name="Count")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=count_df, x=feature, y="Count", hue="class", ax=ax)
        ax.set_title(f"{feature} by Class")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ── HISTOGRAM ──────────────────────────────────────
    with tab_hist:
        st.subheader("Feature Distribution")

        hist_feature = st.selectbox(
            "Select a feature",
            [c for c in df_clean.columns if c != "class"],
            key="hist_feature"
        )

        if hist_feature in label_map:
            with st.expander(f"Label guide for {hist_feature}"):
                for k, v in label_map[hist_feature].items():
                    st.write(f"{k}: {v}")

        with st.expander("Label guide for class"):
            for k, v in label_map["class"].items():
                st.write(f"{k}: {v}")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x=hist_feature, hue="class", multiple="layer", kde=False, ax=ax)
        ax.set_title(f"Distribution of {hist_feature}")
        ax.set_xlabel(hist_feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # ── SCATTERPLOT ────────────────────────────────────
    with tab_scatter:
        st.subheader("Relationship Between Two Features")

        feat_options = [c for c in df_clean.columns if c != "class" and df_clean[c].var() > 0]

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            x_feat = st.selectbox("X axis", feat_options, index=feat_options.index("gill-size"), key="sc_x")
        with sc2:
            y_feat = st.selectbox("Y axis", feat_options, index=feat_options.index("stalk-root"), key="sc_y")
        with sc3:
            n_pts = st.slider("Sample size", 200, min(3000, len(df)), 800, step=100, key="sc_n")

        with st.expander("Label guide for selected variables"):
            if x_feat in label_map:
                st.markdown(f"**{x_feat}**")
                for k, v in label_map[x_feat].items():
                    st.write(f"{k}: {v}")

            if y_feat in label_map:
                st.markdown(f"**{y_feat}**")
                for k, v in label_map[y_feat].items():
                    st.write(f"{k}: {v}")

            st.markdown("**class**")
            for k, v in label_map["class"].items():
                st.write(f"{k}: {v}")

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

    # ── CORRELATION HEATMAP ────────────────────────────
    with tab_heatmap:
        st.subheader("Correlation Matrix")

        corr = df_clean.corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        # --- PAGE: MODEL PREDICTIONS ---

# --- PAGE: MODEL PREDICTIONS ---
elif page == "Model Predictions":
    st.markdown('<div class="sec-title">Model Predictions</div>', unsafe_allow_html=True)
    
    # 1. Comparative Metrics Table
    perf_data = []
    for name, m in results.items():
        perf_data.append({
            "Model": name,
            "Accuracy (%)": m["accuracy"],
            "F1 Score (%)": m["f1"],
            "Recall (%)": m["recall"],
            "Precision (%)": m["precision"],
        })
    
    st.subheader("📊 Comparative Metrics")
    st.dataframe(pd.DataFrame(perf_data).sort_values("F1 Score (%)", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Single Model Inspection (Only ONE Selectbox here)
    st.subheader("🔍 Detailed Model Inspection")
    model_choice = st.selectbox(
        "Select Model to Inspect",
        list(results.keys()),
        key="selectbox_predictions"  
    )

    res = results[model_choice]

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['accuracy']}%")
    c2.metric("F1 Score", f"{res['f1']}%")
    c3.metric("Recall", f"{res['recall']}%")
    c4.metric("Precision", f"{res['precision']}%")

    # Confusion Matrix
    st.write(f"**Confusion Matrix for {model_choice}**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, 
                xticklabels=["Edible", "Poisonous"], 
                yticklabels=["Edible", "Poisonous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("---")

    # 3. Live Predictor
    st.markdown("### 🔮 Live Predictor")
    st.caption("Adjust the sliders to see how a specific mushroom would be classified.")

    user_input = {}
    cols = st.columns(3)

    # Use the columns from X (the trained features)
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            user_input[col] = st.slider(
                col,
                int(X[col].min()),
                int(X[col].max()),
                int(X[col].mean()),
                key=f"predict_slider_{col}"
            )

    if st.button("Predict Edibility"):
        input_df = pd.DataFrame([user_input])

        # Apply scaling ONLY if the model requires it
        # We check the same logic used in training
        models_needing_scaling = ["Logistic Regression", "KNN", "MLP"]
        
        if model_choice in models_needing_scaling:
            input_data = scaler.transform(input_df)
        else:
            input_data = input_df

        model = trained_models[model_choice]
        pred = model.predict(input_data)[0]

        if pred == 0:
            st.success(f"**Result: ✅ EDIBLE** (via {model_choice})")
        else:
            st.error(f"**Result: ⚠️ POISONOUS** (via {model_choice})")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = np.max(proba) * 100
            st.info(f"Confidence Score: {confidence:.2f}%")

# --- PAGE: FEATURE IMPORTANCE ---
elif page == "Feature Importance":
    st.markdown('<div class="sec-title">Feature Importance & Configurations</div>', unsafe_allow_html=True)
    
    # --- SECTION: BEST CONFIGURATIONS ---
    st.subheader("🏆 Best Model Configurations")
    cols = st.columns(len(results))
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.metric(name, f"{res['f1']}% F1")
            st.caption(f"Best Params: {res['best_params']}")
    
    st.markdown("---")

    # --- SECTION: FEATURE IMPORTANCE ---
    st.subheader("📊 Feature Significance")
    model_name = st.selectbox("Select Model to see Importance", [n for n in results.keys() if n != "KNN" and n != "MLP"])
    
    target_model = trained_models[model_name]
    importance_df = None

    # Check for Tree-based importance
    if hasattr(target_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Value': target_model.feature_importances_
        }).sort_values(by='Value', ascending=True)
    
    # Check for Linear-based coefficients
    elif hasattr(target_model, 'coef_'):
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Value': np.abs(target_model.coef_[0]) # Use absolute value for impact
        }).sort_values(by='Value', ascending=True)

    if importance_df is not None:
        fig = px.bar(
            importance_df, 
            x='Value', 
            y='Feature', 
            orientation='h',
            title=f"Feature Impact on {model_name}",
            color='Value',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"The {model_name} model does not provide direct feature importance scores.")

# --- PAGE: HYPERPARAMETER TUNING ---
elif page == "Hyperparameter Tuning":
    st.markdown('<div class="sec-title">Hyperparameter Tuning</div>', unsafe_allow_html=True)
    st.info("Performance comparison of model variants.")
    # Placeholder for tuning visualization
    tuning_data = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "KNN", "MLP"],
        "Best Score": [95.8, 96.5, 94.2, 95.1]
    })
    fig = px.line(tuning_data, x="Model", y="Best Score", markers=True)
    st.plotly_chart(fig, use_container_width=True)
