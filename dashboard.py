import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Dashboard Lean 4.0 Maturité", layout="wide")

# --- Chargement des données ---

@st.cache_data
def load_data():
    df = pd.read_excel("processed_data.xlsx")  # Assure-toi que le fichier est dans le même dossier que ton script
    return df

df = load_data()

# --- Nettoyage et préparation des données ---

def clean_data(df):
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
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["maturity_level_int"] = pd.to_numeric(df["maturity_level_int"], errors='coerce')
    df["Lean_level_int"] = pd.to_numeric(df["Lean_level_int"], errors='coerce')
    df["Digital_level_int"] = pd.to_numeric(df["Digital_level_int"], errors='coerce')
    df["Taille entreprise"] = pd.to_numeric(df["Taille entreprise "], errors='coerce')

    df = df.dropna(subset=features + ["maturity_level_int", "Lean_level_int", "Digital_level_int", "Taille entreprise "])

    return df, features

df, features = clean_data(df)

# --- Interface utilisateur ---

st.sidebar.header("Filtres")

# Dans la partie filtres
taille_min = int(df["Taille entreprise"].min())
taille_max = int(df["Taille entreprise"].max())
taille_sel = st.sidebar.slider("Taille de l'entreprise", taille_min, taille_max, (taille_min, taille_max))


maturity_min = int(df["maturity_level_int"].min())
maturity_max = int(df["maturity_level_int"].max())
maturity_sel = st.sidebar.slider("Maturité globale", maturity_min, maturity_max, (maturity_min, maturity_max))

lean_min = int(df["Lean_level_int"].min())
lean_max = int(df["Lean_level_int"].max())
lean_sel = st.sidebar.slider("Maturité Lean", lean_min, lean_max, (lean_min, lean_max))

digital_min = int(df["Digital_level_int"].min())
digital_max = int(df["Digital_level_int"].max())
digital_sel = st.sidebar.slider("Maturité Digitale", digital_min, digital_max, (digital_min, digital_max))

# Dans le filtre du dataframe
df_filtered = df[
    (df["Taille entreprise"] >= taille_sel[0]) & (df["Taille entreprise"] <= taille_sel[1]) &
    (df["maturity_level_int"] >= maturity_sel[0]) & (df["maturity_level_int"] <= maturity_sel[1]) &
    (df["Lean_level_int"] >= lean_sel[0]) & (df["Lean_level_int"] <= lean_sel[1]) &
    (df["Digital_level_int"] >= digital_sel[0]) & (df["Digital_level_int"] <= digital_sel[1])
]
st.markdown(f"### Données filtrées : {df_filtered.shape[0]} lignes")

def plot_progress_bar(value, max_value, label):
    pct = int((value / max_value) * 100)
    st.progress(pct)
    st.write(f"{label}: {value:.2f} / {max_value}")

st.subheader("Maturité moyenne des entreprises filtrées")
avg_maturity = df_filtered["maturity_level_int"].mean()
avg_lean = df_filtered["Lean_level_int"].mean()
avg_digital = df_filtered["Digital_level_int"].mean()

max_maturity = max(df["maturity_level_int"])
max_lean = max(df["Lean_level_int"])
max_digital = max(df["Digital_level_int"])

plot_progress_bar(avg_maturity, max_maturity, "Maturité globale moyenne")
plot_progress_bar(avg_lean, max_lean, "Maturité Lean moyenne")
plot_progress_bar(avg_digital, max_digital, "Maturité Digitale moyenne")

# Analyse exploratoire simple

st.subheader("Distribution des maturités")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_filtered["maturity_level_int"], bins=10, ax=ax[0], color="skyblue")
ax[0].set_title("Maturité globale")

sns.histplot(df_filtered["Lean_level_int"], bins=10, ax=ax[1], color="lightgreen")
ax[1].set_title("Maturité Lean")

sns.histplot(df_filtered["Digital_level_int"], bins=10, ax=ax[2], color="salmon")
ax[2].set_title("Maturité Digitale")

st.pyplot(fig)

st.subheader("Carte de corrélation")

corr_df = df_filtered[features + ["maturity_level_int", "Lean_level_int", "Digital_level_int"]].corr()
fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax_corr, fmt=".2f")
st.pyplot(fig_corr)

# Clustering KMeans

st.subheader("Clustering KMeans sur sous-dimensions")

X_cluster = df_filtered[features].copy()

X_norm = (X_cluster - X_cluster.min()) / (X_cluster.max() - X_cluster.min())

n_clusters = st.slider("Nombre de clusters KMeans", 2, 10, 3)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_norm)

df_filtered["Cluster"] = clusters

st.write("Répartition des clusters:")
st.bar_chart(df_filtered["Cluster"].value_counts().sort_index())

st.write("Maturité moyenne par cluster:")
st.write(df_filtered.groupby("Cluster")[["maturity_level_int", "Lean_level_int", "Digital_level_int"]].mean())

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_norm)
df_filtered["pca1"] = pca_result[:, 0]
df_filtered["pca2"] = pca_result[:, 1]

fig_pca, ax_pca = plt.subplots()
sns.scatterplot(data=df_filtered, x="pca1", y="pca2", hue="Cluster", palette="Set2", ax=ax_pca)
ax_pca.set_title("Clusters visualisés en 2D avec PCA")
st.pyplot(fig_pca)

# Modèle Decision Tree

st.subheader("Modèle de prédiction de la maturité digitale (Digital_level_int)")

df_filtered["Digital_level_cat"] = pd.cut(df_filtered["Digital_level_int"], bins=3, labels=["Bas", "Moyen", "Haut"])

X = df_filtered[features + ["maturity_level_int", "Lean_level_int"]]
y = df_filtered["Digital_level_cat"]

y_num = y.cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.3, random_state=42)

dtc = DecisionTreeClassifier(max_depth=4, random_state=42)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Accuracy du modèle Decision Tree : **{acc:.2f}**")

st.text("Classification report :")
st.text(classification_report(y_test, y_pred, target_names=["Bas", "Moyen", "Haut"]))

fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
plot_tree(dtc, feature_names=X.columns, class_names=["Bas", "Moyen", "Haut"], filled=True, ax=ax_tree, rounded=True)
st.pyplot(fig_tree)

st.markdown("---")
st.markdown("© 2025 - Dashboard Lean 4.0 Maturité")
