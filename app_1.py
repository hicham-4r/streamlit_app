import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# CONFIGURATION GÉNÉRALE
# =====================================================
st.set_page_config(
    page_title="DataVision - Analyse interactive",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.25rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# JEU DE DONNÉES D'EXEMPLE : 100 LIGNES × 20 COLONNES
# =====================================================
@st.cache_data
def generate_demo_data() -> pd.DataFrame:
    """Crée un jeu de données commercial fictif de 100 lignes et 20 colonnes."""
    rng = np.random.default_rng(2026)
    n = 100

    return pd.DataFrame(
        {
            # 7 variables quantitatives
            "Age": rng.integers(18, 66, n),
            "Salaire_Mensuel": rng.integers(3000, 18001, n),
            "Depenses": np.round(rng.uniform(100, 5000, n), 2),
            "Note_Service": np.round(rng.uniform(1, 10, n), 1),
            "Nombre_Commandes": rng.integers(1, 31, n),
            "Duree_Visite": np.round(rng.uniform(2, 150, n), 2),
            "Remise": np.round(rng.uniform(0, 30, n), 1),

            # 9 variables qualitatives
            "Ville": rng.choice(
                ["Casablanca", "Rabat", "Meknès", "Agadir", "Tanger"], n
            ),
            "Sexe": rng.choice(["Femme", "Homme"], n),
            "Categorie": rng.choice(
                ["Informatique", "Maison", "Mode", "Sport", "Beauté"], n
            ),
            "Profil_Client": rng.choice(["Nouveau", "Régulier", "VIP"], n),
            "Etat_Commande": rng.choice(
                ["Livrée", "En préparation", "Annulée"], n
            ),
            "Zone": rng.choice(["Nord", "Sud", "Est", "Ouest", "Centre"], n),
            "Canal_Vente": rng.choice(["Site web", "Application", "Magasin"], n),
            "Mode_Paiement": rng.choice(["Carte", "Espèces", "Virement"], n),
            "Abonnement": rng.choice(["Oui", "Non"], n),

            # 4 variables temporelles
            "Date_Achat": pd.date_range("2026-01-05", periods=n, freq="D"),
            "Date_Inscription": pd.date_range("2025-08-01", periods=n, freq="2D"),
            "Date_Expedition": pd.date_range("2026-01-07", periods=n, freq="D"),
            "Date_Contact": pd.date_range("2026-02-01", periods=n, freq="3D"),
        }
    )


# =====================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Charge un fichier CSV ou Excel avec des messages d'erreur compréhensibles."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin-1")

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Format non pris en charge. Utilisez un fichier CSV ou Excel.")


def try_convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit automatiquement les colonnes texte contenant majoritairement des dates."""
    converted_df = df.copy()

    for column in converted_df.select_dtypes(include=["object", "string"]).columns:
        non_null_count = converted_df[column].notna().sum()
        if non_null_count == 0:
            continue

        parsed = pd.to_datetime(converted_df[column], errors="coerce")
        conversion_rate = parsed.notna().sum() / non_null_count

        if conversion_rate >= 0.80:
            converted_df[column] = parsed

    return converted_df


def detect_variable_types(df: pd.DataFrame):
    quantitative = df.select_dtypes(include=np.number).columns.tolist()
    temporal = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    qualitative = [
        column
        for column in df.columns
        if column not in quantitative and column not in temporal
    ]
    return quantitative, qualitative, temporal


def variable_type(column, quantitative, qualitative, temporal) -> str:
    if column == "Aucune":
        return "Aucune"
    if column in quantitative:
        return "Quantitative"
    if column in qualitative:
        return "Qualitative"
    if column in temporal:
        return "Date"
    return "Inconnue"


# =====================================================
# SÉLECTION AUTOMATIQUE DU GRAPHIQUE
# =====================================================
def choose_auto_chart(x, y, quantitative, qualitative, temporal) -> str:
    x_type = variable_type(x, quantitative, qualitative, temporal)
    y_type = variable_type(y, quantitative, qualitative, temporal)

    if x_type == "Date" or y_type == "Date":
        return "Courbe temporelle"
    if x_type == "Qualitative" and y == "Aucune":
        return "Diagramme en barres"
    if x_type == "Quantitative" and y == "Aucune":
        return "Histogramme"
    if x_type == "Quantitative" and y_type == "Quantitative":
        return "Nuage de points"
    if {x_type, y_type} == {"Qualitative", "Quantitative"}:
        return "Boîte à moustaches"
    return "Diagramme en barres"


def create_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    chart_type: str,
    quantitative,
    qualitative,
    temporal,
):
    """Crée un graphique Plotly adapté aux variables choisies."""
    if df.empty:
        raise ValueError("Aucune ligne ne correspond au filtre sélectionné.")

    x_type = variable_type(x, quantitative, qualitative, temporal)
    y_type = variable_type(y, quantitative, qualitative, temporal)
    dynamic_title = f"{chart_type} — {x}" if y == "Aucune" else f"{chart_type} — {x} et {y}"

    if chart_type == "Diagramme en barres":
        counts = df[x].fillna("Valeur manquante").value_counts().reset_index()
        counts.columns = [x, "Effectif"]
        fig = px.bar(counts, x=x, y="Effectif", text="Effectif", title=dynamic_title)

    elif chart_type == "Histogramme":
        if x_type != "Quantitative":
            raise ValueError("Un histogramme nécessite une variable quantitative en X.")
        fig = px.histogram(df, x=x, nbins=18, marginal="box", title=dynamic_title)

    elif chart_type == "Nuage de points":
        if x_type != "Quantitative" or y_type != "Quantitative":
            raise ValueError("Le nuage de points nécessite deux variables quantitatives.")
        fig = px.scatter(df, x=x, y=y, title=dynamic_title, hover_data=df.columns)

    elif chart_type == "Boîte à moustaches":
        if {x_type, y_type} != {"Qualitative", "Quantitative"}:
            raise ValueError(
                "La boîte à moustaches nécessite une variable qualitative et une variable quantitative."
            )

        category_column = x if x_type == "Qualitative" else y
        numeric_column = y if y_type == "Quantitative" else x
        fig = px.box(
            df,
            x=category_column,
            y=numeric_column,
            points="outliers",
            title=dynamic_title,
        )

    elif chart_type == "Courbe temporelle":
        date_column = x if x_type == "Date" else y if y_type == "Date" else None
        other_column = y if date_column == x else x

        if date_column is None:
            raise ValueError("La courbe temporelle nécessite au moins une variable de type date.")

        temp_df = df.dropna(subset=[date_column]).copy()

        if y == "Aucune" or other_column == "Aucune":
            grouped = temp_df.groupby(date_column).size().reset_index(name="Effectif")
            fig = px.line(
                grouped,
                x=date_column,
                y="Effectif",
                markers=True,
                title=dynamic_title,
            )
        elif other_column in quantitative:
            grouped = (
                temp_df.groupby(date_column, as_index=False)[other_column]
                .mean()
                .sort_values(date_column)
            )
            fig = px.line(
                grouped,
                x=date_column,
                y=other_column,
                markers=True,
                title=dynamic_title,
            )
        else:
            grouped = (
                temp_df.groupby([date_column, other_column])
                .size()
                .reset_index(name="Effectif")
                .sort_values(date_column)
            )
            fig = px.line(
                grouped,
                x=date_column,
                y="Effectif",
                color=other_column,
                markers=True,
                title=dynamic_title,
            )

    else:
        raise ValueError("Type de graphique inconnu.")

    fig.update_layout(
        height=540,
        title_x=0.02,
        hovermode="closest",
        margin=dict(l=25, r=25, t=70, b=25),
    )
    return fig


# =====================================================
# INTERFACE STREAMLIT
# =====================================================
st.markdown('<div class="main-title">📈 DataVision Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Exploration automatique des variables et création de graphiques interactifs.</div>',
    unsafe_allow_html=True,
)

st.sidebar.title("⚙️ Paramètres")
st.sidebar.subheader("1. Source des données")
uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier",
    type=["csv", "xlsx", "xls"],
    help="Formats acceptés : CSV et Excel.",
)

try:
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        st.sidebar.success(f"Fichier chargé : {uploaded_file.name}")
    else:
        df = generate_demo_data()
        st.sidebar.info("Jeu de démonstration utilisé : 100 lignes × 20 colonnes.")
except Exception as error:
    st.error(f"Erreur pendant le chargement du fichier : {error}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("Le fichier ne contient aucune donnée exploitable.")
    st.stop()

df = try_convert_dates(df)
quantitative, qualitative, temporal = detect_variable_types(df)

# =====================================================
# FILTRES
# =====================================================
st.sidebar.subheader("2. Filtre des données")
filtered_df = df.copy()

filter_options = ["Aucun filtre"] + qualitative
filter_column = st.sidebar.selectbox("Colonne à filtrer", filter_options)

if filter_column != "Aucun filtre":
    values = sorted(filtered_df[filter_column].dropna().astype(str).unique().tolist())
    selected_values = st.sidebar.multiselect(
        f"Valeurs de {filter_column}",
        options=values,
        default=values,
    )
    filtered_df = filtered_df[
        filtered_df[filter_column].astype(str).isin(selected_values)
    ]

st.sidebar.caption(f"{len(filtered_df)} ligne(s) affichée(s) sur {len(df)}")

# =====================================================
# INDICATEURS PRINCIPAUX
# =====================================================
st.subheader("Vue d'ensemble")
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Lignes", len(filtered_df))
metric_2.metric("Colonnes", len(df.columns))
metric_3.metric("Valeurs manquantes", int(filtered_df.isna().sum().sum()))
metric_4.metric("Taux de complétude", f"{(1 - filtered_df.isna().mean().mean()) * 100:.1f}%")

with st.expander("Voir la détection automatique des types", expanded=True):
    type_col_1, type_col_2, type_col_3 = st.columns(3)
    with type_col_1:
        st.metric("Quantitatives", len(quantitative))
        st.write(quantitative or "Aucune")
    with type_col_2:
        st.metric("Qualitatives", len(qualitative))
        st.write(qualitative or "Aucune")
    with type_col_3:
        st.metric("Dates", len(temporal))
        st.write(temporal or "Aucune")

# =====================================================
# VISUALISATION INTERACTIVE
# =====================================================
st.subheader("Visualisation interactive")
all_columns = df.columns.tolist()

selection_col_1, selection_col_2, selection_col_3 = st.columns(3)
with selection_col_1:
    x_variable = st.selectbox("Variable X", all_columns)
with selection_col_2:
    y_variable = st.selectbox("Variable Y", ["Aucune"] + all_columns)
with selection_col_3:
    chart_choice = st.selectbox(
        "Type de graphique",
        [
            "Automatique",
            "Diagramme en barres",
            "Histogramme",
            "Nuage de points",
            "Boîte à moustaches",
            "Courbe temporelle",
        ],
    )

final_chart = (
    choose_auto_chart(
        x_variable,
        y_variable,
        quantitative,
        qualitative,
        temporal,
    )
    if chart_choice == "Automatique"
    else chart_choice
)

st.info(
    f"X : **{variable_type(x_variable, quantitative, qualitative, temporal)}** · "
    f"Y : **{variable_type(y_variable, quantitative, qualitative, temporal)}** · "
    f"Graphique : **{final_chart}**"
)

try:
    figure = create_chart(
        filtered_df,
        x_variable,
        y_variable,
        final_chart,
        quantitative,
        qualitative,
        temporal,
    )
    st.plotly_chart(figure, use_container_width=True)
except Exception as error:
    st.warning(f"Ce graphique ne peut pas être généré avec cette sélection : {error}")

# =====================================================
# TABLEAU ET EXPORT
# =====================================================
st.subheader("Aperçu des données filtrées")
st.dataframe(filtered_df, use_container_width=True, height=360)

csv_buffer = io.StringIO()
filtered_df.to_csv(csv_buffer, index=False)
st.download_button(
    "⬇️ Télécharger les données filtrées",
    data=csv_buffer.getvalue().encode("utf-8-sig"),
    file_name="donnees_filtrees.csv",
    mime="text/csv",
)

st.caption(
    "Mini-projet Streamlit — détection des variables, filtre simple, titre dynamique et visualisations Plotly."
)
