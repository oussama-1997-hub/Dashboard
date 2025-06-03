# Lean 4.0 Maturity Analysis Dashboard (Full Project)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Config
st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")
st.title("📊 Lean 4.0 Maturity Dashboard")
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        width: 300px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data --- #
@st.cache_data

def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

# Load and clean
st.sidebar.title("🔎 Filtres")
df = load_data()
df = df.dropna(subset=["maturity_level_int", "Lean_level_int", "Digital_level_int"])

# --- Filters --- #
sectors = st.sidebar.multiselect("Secteur", df["Quelle est le secteur de votre entreprise ? "].unique())
if sectors:
    df = df[df["Quelle est le secteur de votre entreprise ? "].isin(sectors)]

# Taille par ordre
taille_order = ["Très petite", "Petite", "Moyenne", "Grande", "Très grande"]
taille_presentes = [t for t in taille_order if t in df["Taille entreprise "].unique()]
taille_min, taille_max = st.sidebar.select_slider("Taille d'entreprise (échelle)",
                                                  options=taille_presentes,
                                                  value=(taille_presentes[0], taille_presentes[-1]))
def taille_idx(t): return taille_presentes.index(t)
df = df[df["Taille entreprise "].apply(lambda x: taille_idx(taille_min) <= taille_idx(x) <= taille_idx(taille_max))]

# --- KPI Display --- #
k1, k2, k3 = st.columns(3)
k1.metric("📈 Maturité Globale Moyenne", round(df["maturity_level_int"].mean(), 2))
k2.metric("📊 Lean Maturity", round(df["Lean_level_int"].mean(), 2))
k3.metric("🖥️ Digital Maturity", round(df["Digital_level_int"].mean(), 2))

# --- EDA --- #
st.subheader("📊 Analyse exploratoire")

# Distribution des scores de maturité
with st.expander("📌 Distribution des Niveaux de Maturité"):
    fig1 = px.histogram(df, x="maturity_level_int", nbins=10, title="Maturité Globale")
    fig2 = px.histogram(df, x="Lean_level_int", nbins=10, title="Lean Maturity")
    fig3 = px.histogram(df, x="Digital_level_int", nbins=10, title="Digital Maturity")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

# Corrélation heatmap
with st.expander("🔬 Corrélation entre dimensions"):
    scores = df[["maturity_level_int", "Lean_level_int", "Digital_level_int"]]
    fig, ax = plt.subplots()
    sns.heatmap(scores.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Outils utilisés --- #
with st.expander("🛠️ Outils Lean et Technologies utilisés"):
    st.write("**Méthodes Lean utilisées :**")
    st.dataframe(df["Dans votre entreprise, quels sont les outils lean mis en place durant ces dernières années ? "].value_counts())
    st.write("**Technologies Industrie 4.0 utilisées :**")
    st.dataframe(df["Dans votre entreprise, quelles sont les technologies mises en place durant ces dernières années ? "].value_counts())

# --- Clustering --- #
st.subheader("🤖 Clustering KMeans des entreprises")
features = [col for col in df.columns if col.startswith("Leadership") or col.startswith("Supply") or col.startswith("Opérations") or col.startswith("Technologies") or col.startswith("Organisation")]
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Nombre de clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
df["Cluster"] = kmeans.labels_

fig_cluster = px.scatter(df, x="Lean_level_int", y="Digital_level_int", color="Cluster",
                         title="Clustering selon les niveaux de maturité")
st.plotly_chart(fig_cluster, use_container_width=True)

# --- Modèle prédictif --- #
st.subheader("🌲 Prédiction de la Maturité Technologique (Decision Tree)")
X = df[features].dropna()
y = df.loc[X.index, "Digital_level_int"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.text("Rapport de classification :")
st.text(classification_report(y_test, y_pred))

st.text("Matrice de confusion :")
st.text(confusion_matrix(y_test, y_pred))

with st.expander("📉 Visualiser l'arbre de décision"):
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=X.columns, filled=True, fontsize=8)
    st.pyplot(fig)

# --- Données brutes --- #
with st.expander("📄 Afficher les données brutes"):
    st.dataframe(df)
