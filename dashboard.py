import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Dashboard Lean 4.0", layout="wide")

# 📥 Load data
@st.cache_data
def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

st.title("Dashboard d'analyse Lean 4.0 - Maturité organisationnelle et technologique")

# --- Sidebar : filtres ---
st.sidebar.header("Filtres")

# Filtre Taille entreprise (catégories string)
taille_categories = df["Taille entreprise "].unique().tolist()
taille_sel = st.sidebar.multiselect(
    "Taille de l'entreprise",
    options=taille_categories,
    default=taille_categories  # par défaut, tout sélectionné
)

# Si aucune taille sélectionnée, on prend tout
if len(taille_sel) == 0:
    taille_sel = taille_categories

# Filtre maturité globale
maturity_min = int(df["maturity_level_int"].min())
maturity_max = int(df["maturity_level_int"].max())
maturity_sel = st.sidebar.slider("Maturité globale (Lean 4.0)", maturity_min, maturity_max, (maturity_min, maturity_max))

# Filtre maturité Lean
lean_min = int(df["Lean_level_int"].min())
lean_max = int(df["Lean_level_int"].max())
lean_sel = st.sidebar.slider("Maturité Lean", lean_min, lean_max, (lean_min, lean_max))

# Filtre maturité Digitale
digital_min = int(df["Digital_level_int"].min())
digital_max = int(df["Digital_level_int"].max())
digital_sel = st.sidebar.slider("Maturité Digitale", digital_min, digital_max, (digital_min, digital_max))

# Appliquer filtres
df_filtered = df[
    (df["Taille entreprise "].isin(taille_sel)) &
    (df["maturity_level_int"] >= maturity_sel[0]) & (df["maturity_level_int"] <= maturity_sel[1]) &
    (df["Lean_level_int"] >= lean_sel[0]) & (df["Lean_level_int"] <= lean_sel[1]) &
    (df["Digital_level_int"] >= digital_sel[0]) & (df["Digital_level_int"] <= digital_sel[1])
]

st.markdown(f"### Données filtrées : {df_filtered.shape[0]} lignes")

# --- Analyse exploratoire (EDA) ---

st.header("Analyse exploratoire")

# Histogramme maturité globale
fig, ax = plt.subplots()
sns.histplot(df_filtered["maturity_level_int"], bins=10, kde=True, ax=ax)
ax.set_title("Distribution de la maturité globale")
st.pyplot(fig)

# Scatter plot maturité Lean vs Digitale
fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=df_filtered,
    x="Lean_level_int",
    y="Digital_level_int",
    hue="maturity_level_int",
    palette="viridis",
    ax=ax2
)
ax2.set_title("Maturité Lean vs Maturité Digitale")
st.pyplot(fig2)

# --- Clustering KMeans ---

st.header("Clustering KMeans sur les maturités")

cluster_features = df_filtered[["maturity_level_int", "Lean_level_int", "Digital_level_int"]].dropna()

if cluster_features.shape[0] > 0:
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(cluster_features)
    df_filtered = df_filtered.loc[cluster_features.index]
    df_filtered["Cluster"] = clusters

    fig3, ax3 = plt.subplots()
    sns.scatterplot(
        data=df_filtered,
        x="Lean_level_int",
        y="Digital_level_int",
        hue="Cluster",
        palette="Set2",
        ax=ax3,
        legend="full"
    )
    ax3.set_title("Clusters de maturité")
    st.pyplot(fig3)

    st.write("Taille de chaque cluster:")
    st.write(df_filtered["Cluster"].value_counts())
else:
    st.warning("Pas assez de données pour effectuer le clustering.")

# --- Modèle de prédiction (Decision Tree) ---

st.header("Prédiction du niveau de maturité digitale")

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

df_ml = df_filtered.dropna(subset=features + ["Digital_level_int"])
X = df_ml[features]
y = df_ml["Digital_level_int"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.subheader("Performance du modèle")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

fig4, ax4 = plt.subplots(figsize=(20,10))
plot_tree(clf, feature_names=features, class_names=[str(i) for i in sorted(y.unique())], filled=True, ax=ax4)
st.pyplot(fig4)
