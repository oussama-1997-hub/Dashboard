# Lean 4.0 Maturity Dashboard - Streamlit Full App
# Author: Oussama Ben Ali

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")
st.title("📊 Tableau de bord de la maturité Lean 4.0")

# --- Load Data ---
@st.cache_data

def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filtres")

# Secteurs (catégorique)
sectors = st.sidebar.multiselect("Secteur", df["Quelle est le secteur de votre entreprise ? "].unique(),
                                  default=df["Quelle est le secteur de votre entreprise ? "].unique())

# Taille (échelle)
size_mapping = {"Petite": 1, "Moyenne": 2, "Grande": 3}
df["Taille_code"] = df["Taille entreprise "].map(size_mapping)
taille_min, taille_max = int(df["Taille_code"].min()), int(df["Taille_code"].max())
taille_slider = st.sidebar.slider("Taille d'entreprise (1: Petite → 3: Grande)", min_value=1, max_value=3, value=(taille_min, taille_max))

# Niveau de maturité (barre)
maturity_range = st.sidebar.slider("Niveau de maturité global Lean 4.0", 
                                    int(df["maturity_level_int"].min()), 
                                    int(df["maturity_level_int"].max()),
                                    (int(df["maturity_level_int"].min()), int(df["maturity_level_int"].max())))

# Filtrage des données
filtered_df = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(sectors)) &
    (df["Taille_code"] >= taille_slider[0]) & (df["Taille_code"] <= taille_slider[1]) &
    (df["maturity_level_int"] >= maturity_range[0]) & (df["maturity_level_int"] <= maturity_range[1])
]

# --- KPIs ---
st.subheader("📌 Indicateurs Clés de Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Maturité Lean", round(filtered_df["Lean_level_int"].mean(), 2))
col2.metric("Maturité Industrie 4.0", round(filtered_df["Digital_level_int"].mean(), 2))
col3.metric("Maturité Globale", round(filtered_df["maturity_level_int"].mean(), 2))

# --- Méthodes Lean & Technologies ---
st.markdown("---")
st.subheader("🛠️ Méthodes Lean et Technologies utilisées")
st.dataframe(filtered_df[["Dans votre entreprise, quels sont les outils lean mis en place durant ces dernières années ?",
                         "Dans votre entreprise, quelles sont les technologies mises en place durant ces dernières années ? "]])

# --- Heatmap de corrélation ---
st.subheader("📈 Corrélation entre les dimensions organisationnelles et la maturité technologique")
org_cols = [c for c in df.columns if any(dim in c for dim in ["Leadership", "Supply Chain", "Opérations", "Technologies", "Organisation apprenante"])]
corr = filtered_df[org_cols + ["maturity_level_int"]].corr()
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(corr[["maturity_level_int"]].sort_values(by="maturity_level_int", ascending=False), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# --- Clustering avec KMeans ---
st.subheader("🤖 Clustering des profils d'entreprise")
cluster_df = filtered_df[["Lean_level_int", "Digital_level_int", "maturity_level_int"] + org_cols].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df["Cluster"] = kmeans.fit_predict(cluster_df)
fig_cluster = px.scatter_3d(cluster_df, x="Lean_level_int", y="Digital_level_int", z="maturity_level_int",
                            color="Cluster", title="Clusters des entreprises selon leur maturité")
st.plotly_chart(fig_cluster, use_container_width=True)

# --- Arbre de Décision pour prédire la maturité ---
st.subheader("🌳 Prédiction du niveau de maturité avec un arbre de décision")
x = filtered_df[org_cols].dropna()
y = filtered_df.loc[x.index, "maturity_level_int"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
st.markdown(f"**🎯 Précision du modèle :** {accuracy_score(y_test, y_pred):.2f}")
fig_tree, ax = plt.subplots(figsize=(18, 8))
plot_tree(dt, feature_names=x.columns, class_names=True, filled=True, ax=ax)
st.pyplot(fig_tree)

# --- Données brutes ---
with st.expander("📄 Données filtrées"):
    st.dataframe(filtered_df)
