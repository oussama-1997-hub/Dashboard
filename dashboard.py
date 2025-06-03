import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Dashboard Lean 4.0 Maturité", layout="wide")

# Charger les données
@st.cache_data

def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

# --- Nettoyage des données ---
def clean_data(df):
    # Colonnes sous-dimensions Lean 4.0
    features = [
        "Leadership - Engagement Lean ",
        "Leadership - Engagement DT",
        "Leadership - Stratégie ",
        "Leadership - Communication",
        "Supply Chain - Collaboration inter-organisationnelle",
        "Supply Chain - Traçabilité",
        "Supply Chain - Impact sur les employées",
        "Opérations - Standardisation des processus",
        "Opérations - Juste-à-temps (JAT)",
        "Opérations - Gestion des résistances",
        "Technologies - Connectivité et gestion des données",
        "Technologies - Automatisation",
        "Technologies - Pilotage du changement",
        "Organisation apprenante  - Formation et développement des compétences",
        "Organisation apprenante  - Collaboration et Partage des Connaissances",
        "Organisation apprenante  - Flexibilité organisationnelle"
    ]
    # Convertir en float
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Colonnes maturité et taille
    df["maturity_level_int"] = pd.to_numeric(df["maturity_level_int"], errors='coerce')
    df["Lean_level_int"] = pd.to_numeric(df["Lean_level_int"], errors='coerce')
    df["Digital_level_int"] = pd.to_numeric(df["Digital_level_int"], errors='coerce')

    # Exemple taille entreprise codée (assurez-vous que cette colonne existe dans votre fichier)
    df["Taille_code"] = pd.to_numeric(df["Taille_code"], errors='coerce')

    # Supprimer lignes avec NaN dans les colonnes essentielles
    df = df.dropna(subset=features + ["maturity_level_int", "Lean_level_int", "Digital_level_int", "Taille_code"])

    return df, features

# --- Interface utilisateur ---
st.title("Dashboard Lean 4.0 - Analyse de maturité")

# Charger dataset
uploaded_file = st.file_uploader("Charger le fichier CSV des données", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    df, features = clean_data(df)

    # Filtres
    st.sidebar.header("Filtres")

    taille_min = int(df["Taille_code"].min())
    taille_max = int(df["Taille_code"].max())
    taille_sel = st.sidebar.slider("Taille de l'entreprise (code)", taille_min, taille_max, (taille_min, taille_max))

    maturity_min = int(df["maturity_level_int"].min())
    maturity_max = int(df["maturity_level_int"].max())
    maturity_sel = st.sidebar.slider("Maturité globale", maturity_min, maturity_max, (maturity_min, maturity_max))

    lean_min = int(df["Lean_level_int"].min())
    lean_max = int(df["Lean_level_int"].max())
    lean_sel = st.sidebar.slider("Maturité Lean", lean_min, lean_max, (lean_min, lean_max))

    digital_min = int(df["Digital_level_int"].min())
    digital_max = int(df["Digital_level_int"].max())
    digital_sel = st.sidebar.slider("Maturité Digitale", digital_min, digital_max, (digital_min, digital_max))

    # Appliquer filtres
    df_filtered = df[
        (df["Taille_code"] >= taille_sel[0]) & (df["Taille_code"] <= taille_sel[1]) &
        (df["maturity_level_int"] >= maturity_sel[0]) & (df["maturity_level_int"] <= maturity_sel[1]) &
        (df["Lean_level_int"] >= lean_sel[0]) & (df["Lean_level_int"] <= lean_sel[1]) &
        (df["Digital_level_int"] >= digital_sel[0]) & (df["Digital_level_int"] <= digital_sel[1])
    ]

    st.markdown(f"### Données filtrées : {df_filtered.shape[0]} lignes")

    # --- Analyse exploratoire (EDA) ---
    st.subheader("Analyse exploratoire")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Statistiques descriptives des maturités")
        st.write(df_filtered[["maturity_level_int", "Lean_level_int", "Digital_level_int"]].describe())

    with col2:
        st.write("Histogrammes des niveaux de maturité")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df_filtered["maturity_level_int"], kde=True, color="blue", label="Globale", ax=ax)
        sns.histplot(df_filtered["Lean_level_int"], kde=True, color="green", label="Lean", ax=ax)
        sns.histplot(df_filtered["Digital_level_int"], kde=True, color="red", label="Digitale", ax=ax)
        plt.legend()
        st.pyplot(fig)

    # --- Clustering KMeans ---
    st.subheader("Clustering des sous-dimensions organisationnelles (KMeans)")

    X_cluster = df_filtered[features].copy()
    X_cluster = X_cluster.astype(float)  # assure conversion

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_cluster)

    df_filtered["Cluster"] = clusters

    st.write("Répartition des clusters")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Cluster", data=df_filtered, ax=ax2)
    st.pyplot(fig2)

    # Visualisation clusters vs maturité globale
    st.write("Maturité globale par cluster")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Cluster", y="maturity_level_int", data=df_filtered, ax=ax3)
    st.pyplot(fig3)

    # --- Modèle ML : Prédiction de la maturité digitale ---
    st.subheader("Modèle ML: Prédiction de la maturité digitale (Digital_level_int)")

    X = df_filtered[features + ["Lean_level_int"]]
    y = df_filtered["Digital_level_int"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy sur test set : {acc:.2f}")

    st.text("Rapport classification :")
    st.text(classification_report(y_test, y_pred))

    # Visualisation de l'arbre
    fig4, ax4 = plt.subplots(figsize=(15,8))
    plot_tree(clf, feature_names=X.columns, class_names=[str(x) for x in sorted(y.unique())], filled=True, ax=ax4)
    st.pyplot(fig4)

else:
    st.info("Merci de charger un fichier CSV pour commencer.")

