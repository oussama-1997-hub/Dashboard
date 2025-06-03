import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# --- Chargement des données ---
uploaded_file = st.sidebar.file_uploader("Importer le fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.title("Dashboard Lean 4.0 - Analyse et Prédiction de la Maturité")

    # Nettoyage des colonnes de méthodes et outils
    lean_methods = {
        "5S": 0.077, "Kaizen": 0.081, "Value Stream Mapping (VSM)": 0.157, "Kanban": 0.125,
        "Méthode TPM / TRS": 0.121, "Takt Time": 0.198, "6 sigma": 0.077, "QRQC": 0.036,
        "Heijunka": 0.048, "Poka Yoke": 0.081
    }

    i4_tools = {
        "ERP (Enterprise Resource Planning)": 0.060, "WMS (Warehouse Management System)": 0.060,
        "MES (Manufacturing Execution System)": 0.060, "RFID": 0.110, "Intelligence artificielle": 0.028,
        "Big Data et Analytics": 0.161, "Fabrication additive (Impression 3D)": 0.085,
        "Réalité augmentée": 0.057, "Maintenance prédictive": 0.047,
        "Systèmes cyber physiques": 0.136, "Simulation": 0.085, "Cloud computing": 0.110
    }

    df['lean_score'] = df.iloc[:, df.columns.get_loc("Dans votre entreprise, quels sont les outils lean mis en place durant ces dernières années ?")].apply(lambda x: sum([lean_methods[m] for m in lean_methods if m in str(x)]))
    df['i4_score'] = df.iloc[:, df.columns.get_loc("Dans votre entreprise, quelles sont les technologies mises en place durant ces dernières années ?")].apply(lambda x: sum([i4_tools[m] for m in i4_tools if m in str(x)]))

    # Mapping taille entreprise (si c'est du texte)
    if df["Taille entreprise "].dtype == 'object':
        taille_map = {"Petite": 1, "Moyenne": 2, "Grande": 3, "Très grande": 4}
        df["Taille_code"] = df["Taille entreprise "].map(taille_map)
    else:
        df["Taille_code"] = df["Taille entreprise "]

    # --- Filtres ---
    with st.sidebar:
        st.subheader("Filtres")
        taille_range = st.slider("Taille de l'entreprise", 1, 4, (1, 4), format="%d")
        lean_range = st.slider("Niveau Lean", 0, 5, (0, 5), format="%d")
        digital_range = st.slider("Niveau Digital", 0, 5, (0, 5), format="%d")

    df_filtered = df[
        (df["Taille_code"] >= taille_range[0]) & (df["Taille_code"] <= taille_range[1]) &
        (df["Lean_level_int"] >= lean_range[0]) & (df["Lean_level_int"] <= lean_range[1]) &
        (df["Digital_level_int"] >= digital_range[0]) & (df["Digital_level_int"] <= digital_range[1])
    ]

    # --- Visualisation ---
    st.subheader("Distribution des niveaux de maturité")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.histogram(df_filtered, x="maturity_level_int", nbins=6, title="Maturité Globale"))
    with col2:
        st.plotly_chart(px.histogram(df_filtered, x="Lean_level_int", nbins=6, title="Maturité Lean"))
    with col3:
        st.plotly_chart(px.histogram(df_filtered, x="Digital_level_int", nbins=6, title="Maturité Digital"))

    # --- Clustering ---
    st.subheader("Clustering (KMeans)")
    try:
        X_cluster = df_filtered[["lean_score", "i4_score"]].dropna()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X_cluster)
        df_filtered.loc[X_cluster.index, "cluster"] = kmeans.labels_
        st.plotly_chart(px.scatter(df_filtered, x="lean_score", y="i4_score", color="cluster",
                                   title="Clusters Lean / I4", hover_data=['maturity_level_int']))
    except Exception as e:
        st.warning(f"Clustering non affiché : {e}")

    # --- Modèle de Prédiction ---
    st.subheader("Prédiction de la Maturité Technologique")
    sub_dims = [c for c in df.columns if any(dim in c for dim in ["Leadership", "Supply Chain", "Opérations", "Technologies", "Organisation"])]
    X = df_filtered[sub_dims].dropna()
    y = df_filtered.loc[X.index, "maturity_level_int"]

    try:
        clf = DecisionTreeClassifier(max_depth=4, random_state=0)
        clf.fit(X, y)

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(clf, feature_names=sub_dims, class_names=True, filled=True, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Erreur dans le modèle : {e}")

else:
    st.info("Veuillez importer un fichier Excel pour commencer l'analyse.")
