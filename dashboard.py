# Lean 4.0 Maturity Full Pipeline Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Lean 4.0 Maturity Explorer", layout="wide")
st.title("🌐 Lean 4.0 Maturity Dashboard")

# Load Data
@st.cache_data

def load_data():
    return pd.read_excel("processed_data.xlsx")

df = load_data()

# ======== Sidebar Filters ========
st.sidebar.title("🔍 Filtres")

sectors = st.sidebar.multiselect(
    "Secteur",
    options=df["Quelle est le secteur de votre entreprise ? "].unique(),
    default=df["Quelle est le secteur de votre entreprise ? "].unique(),
)

taille_ordered = ["Très petite", "Petite", "Moyenne", "Grande", "Très grande"]
df["Taille entreprise "] = pd.Categorical(df["Taille entreprise "], categories=taille_ordered, ordered=True)

taille_min = st.sidebar.select_slider("Taille entreprise (min)", options=taille_ordered, value="Très petite")
taille_max = st.sidebar.select_slider("Taille entreprise (max)", options=taille_ordered, value="Très grande")

maturity_range = st.sidebar.slider(
    "Niveau de Maturité Technologique",
    min_value=int(df["Digital_level_int "].min()),
    max_value=int(df["Digital_level_int "].max()),
    value=(int(df["Digital_level_int "].min()), int(df["Digital_level_int "].max()))
)

# Filter
df_filtered = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(sectors)) &
    (df["Taille entreprise "] >= taille_min) & (df["Taille entreprise "] <= taille_max) &
    (df["Digital_level_int "] >= maturity_range[0]) & (df["Digital_level_int "] <= maturity_range[1])
]

# ======== Visualisation Section ========
st.header("📊 Analyse Exploratoire")

# 1. Répartition des niveaux de maturité
col1, col2 = st.columns(2)

with col1:
    st.subheader("Répartition par Niveau de Maturité")
    fig1 = px.histogram(df_filtered, x="Digital_level_int ", nbins=6, color="Taille entreprise ", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Répartition par Secteur")
    fig2 = px.pie(df_filtered, names="Quelle est le secteur de votre entreprise ? ", title="Par secteur")
    st.plotly_chart(fig2, use_container_width=True)

# 2. Heatmap de corrélation
st.subheader("🔬 Corrélation entre Sous-Dimensions Organisationnelles et Maturité Technologique")
sub_dimensions = [
    col for col in df.columns if any(dim in col for dim in ["Leadership", "Supply", "Opérations", "Technologies", "Organisation apprenante"])
]

corr_df = df_filtered[sub_dimensions + ["Digital_level_int ", "Lean_level_int", "maturity_level_int"]].corr()
fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_df, cmap="coolwarm", annot=True, fmt=".2f", ax=ax3)
st.pyplot(fig3)

# 3. Outils Lean et Industrie 4.0
st.subheader("🛠️ Outils et Technologies Utilisées")
st.markdown("**Méthodes Lean**")
st.dataframe(df_filtered["Dans votre entreprise, quels sont les outils lean mis en place durant ces dernières années ?"].value_counts().reset_index().rename(columns={"index": "Méthode", "Dans votre entreprise, quels sont les outils lean mis en place durant ces dernières années ?": "Nombre"}))

st.markdown("**Technologies Industrie 4.0**")
st.dataframe(df_filtered["Dans votre entreprise, quelles sont les technologies mises en place durant ces dernières années ? "].value_counts().reset_index().rename(columns={"index": "Technologie", "Dans votre entreprise, quelles sont les technologies mises en place durant ces dernières années ? ": "Nombre"}))

# ======== Clustering ========
st.header("🔎 Clustering des entreprises")
cluster_data = df_filtered[sub_dimensions + ["Digital_level_int "]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

cluster_data["Cluster"] = clusters
fig4 = px.scatter_3d(
    cluster_data,
    x="Leadership - Engagement Lean ",
    y="Opérations - Juste-à-temps (JAT)",
    z="Digital_level_int ",
    color="Cluster",
    title="🧠 Visualisation des Clusters",
)
st.plotly_chart(fig4, use_container_width=True)

# ======== ML Prediction ========
st.header("🤖 Prédiction du Niveau de Maturité Technologique")

X = df_filtered[sub_dimensions].dropna()
y = df_filtered.loc[X.index, "Digital_level_int "]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Précision du modèle : {round(acc*100, 2)}%")

fig5, ax5 = plt.subplots(figsize=(20, 8))
plot_tree(clf, feature_names=X.columns, filled=True, fontsize=8)
st.pyplot(fig5)

# Footer
st.markdown("---")
st.caption("© 2025 Lean 4.0 Intelligence - Tous droits réservés")
