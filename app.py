import streamlit as st
import pandas as pd
import joblib

# --- Informations d'identification (à personnaliser) ---
VALID_USERNAME = "admin"
VALID_PASSWORD = "aziz123"

# --- Gestion de la session ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- Formulaire de connexion ---
if not st.session_state.authenticated:
    st.title("🔐 Connexion requise")
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")

    if submit:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.success("Connexion réussie !")
            st.rerun()
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")
    st.stop()
# ------------------------------------------------------------


st.set_page_config(
    page_title="Prédiction Dégradation Réseau",
    layout="centered",
    initial_sidebar_state="auto"
)

@st.cache_data
def load_data_and_models():
    df_all = pd.read_pickle("./saved_data/full_df_before_FE.pkl")

    le_Zone         = joblib.load("./saved_encoders/le_Zone.pkl")
    le_Constructeur = joblib.load("./saved_encoders/le_Constructeur.pkl")
    le_Région       = joblib.load("./saved_encoders/le_Région.pkl")
    le_Site_Critique= joblib.load("./saved_encoders/le_Site_Critique.pkl")

    best_model = joblib.load("./saved_models/best_xgboost.pkl")
    return df_all, le_Zone, le_Constructeur, le_Région, le_Site_Critique, best_model

df_all, le_Zone, le_Constructeur, le_Région, le_Site_Critique, best_model = load_data_and_models()
site_ids = df_all['SiteID'].unique().tolist()

st.title("🔮 Application de prédiction de Dégradation Réseau (Label T+1)")

st.markdown("""
- Sélectionnez un **SiteID**.
- Sélectionnez la **semaine suivante disponible** (T+1) pour ce site.
- Modifiez les moyennes 2G, 3G et 4G.
- Le champ **Site Critique** est dynamique selon la semaine.
""")

site_id = st.selectbox("SiteID", options=site_ids)
site_data = df_all[df_all['SiteID'] == site_id].sort_values(by=['Année', 'Semaine'])

# Dernière semaine connue
dernier_année = site_data['Année'].iloc[-1]
derniere_semaine = site_data['Semaine'].iloc[-1]

def semaine_suivante(année, semaine):
    if semaine >= 52:
        return année + 1, 1
    else:
        return année, semaine + 1

année_suiv, semaine_suiv = semaine_suivante(dernier_année, derniere_semaine)

# Vérifier si la semaine T+1 existe
semaine_dispo = ((site_data['Année'] == année_suiv) & (site_data['Semaine'] == semaine_suiv)).any()
année_choisie = année_suiv if semaine_dispo else dernier_année
semaine_choisie = semaine_suiv if semaine_dispo else derniere_semaine

st.selectbox("Semaine (T+1)", options=[semaine_choisie], disabled=True)

# Données à prédire
ligne_match = df_all[
    (df_all['SiteID'] == site_id) &
    (df_all['Année'] == année_choisie) &
    (df_all['Semaine'] == semaine_choisie)
]

if not ligne_match.empty:
    ligne = ligne_match.iloc[0]
else:
    ligne = pd.Series({
        'Constructeur': '',
        'Latitude': 0.0,
        'Longitude': 0.0,
        'Zone': '',
        'Région': '',
        'Site_Critique': '',
        'Moyenne_2G': 0.0,
        'Moyenne_3G': 0.0,
        'Moyenne_4G': 0.0
    })

# Champs non modifiables
st.text_input("Constructeur", value=ligne['Constructeur'], disabled=True)
st.text_input("Zone", value=ligne['Zone'], disabled=True)
st.text_input("Région", value=ligne['Région'], disabled=True)

# Site Critique dynamique
site_critique_options = df_all['Site_Critique'].unique().tolist()
try:
    default_index = site_critique_options.index(ligne['Site_Critique'])
except ValueError:
    default_index = 0
site_critique = st.selectbox("Site Critique", options=site_critique_options, index=default_index)

# Latitude/Longitude
st.number_input("Latitude", value=float(ligne['Latitude']), format="%.6f", disabled=True)
st.number_input("Longitude", value=float(ligne['Longitude']), format="%.6f", disabled=True)
st.number_input("Année", value=année_choisie, disabled=True)

# Moyennes modifiables
col1, col2, col3 = st.columns(3)
moy_2G = col1.number_input("Moyenne_2G", min_value=0.0, value=float(ligne.get('Moyenne_2G', 0.0)))
moy_3G = col2.number_input("Moyenne_3G", min_value=0.0, value=float(ligne.get('Moyenne_3G', 0.0)))
moy_4G = col3.number_input("Moyenne_4G", min_value=0.0, value=float(ligne.get('Moyenne_4G', 0.0)))

# Carte
st.map(pd.DataFrame({'lat': [ligne['Latitude']], 'lon': [ligne['Longitude']]}))

def predict_new_record(raw_dict):
    new_row = pd.DataFrame([raw_dict])

    temp = pd.concat([df_all.copy(), new_row], ignore_index=True)
    temp['Année'] = temp['Année'].astype(int)
    temp['Semaine'] = temp['Semaine'].astype(int)

    for col in ['Moyenne_2G','Moyenne_3G','Moyenne_4G','Latitude','Longitude']:
        temp[col] = temp[col].astype(float)

    for cat in ['Zone','Constructeur','Région','Site_Critique']:
        temp[cat] = temp[cat].astype(str)

    temp = temp.sort_values(by=['SiteID','Année','Semaine']).reset_index(drop=True)

    for gen in ['2G','3G','4G']:
        temp[f'moy_{gen}_3sem'] = (
            temp.groupby('SiteID')[f'Moyenne_{gen}']
                .rolling(window=3, min_periods=1)
                .mean().reset_index(level=0, drop=True)
        )
        temp[f'std_{gen}_3sem'] = (
            temp.groupby('SiteID')[f'Moyenne_{gen}']
                .rolling(window=3, min_periods=1)
                .std().reset_index(level=0, drop=True)
        ).fillna(0)

    # ⚠️ ALERT: Données groupées après ajout de la nouvelle ligne
    # st.warning("📈 Données après ajout et groupement par SiteID (aperçu des moyennes) :")
    grouped_data = temp.groupby("SiteID")[['Moyenne_2G', 'Moyenne_3G', 'Moyenne_4G']].mean().round(2)
    # st.dataframe(grouped_data.loc[[raw_dict["SiteID"]]])

    temp['Zone']          = le_Zone.transform(temp['Zone'])
    temp['Constructeur']  = le_Constructeur.transform(temp['Constructeur'])
    temp['Région']        = le_Région.transform(temp['Région'])
    temp['Site_Critique'] = le_Site_Critique.transform(temp['Site_Critique'])

    # ✅ Afficher le DataFrame après encodage
    # st.success("✅ DataFrame complet après encodage des variables catégorielles :")
    # st.dataframe(temp)


    # Retrouver précisément la ligne ajoutée
    new_record = temp[temp["row_id"] == raw_dict["row_id"]].copy()
    if new_record.empty:
        st.error("❌ Erreur : ligne ajoutée introuvable pour la prédiction.")
        return {"label_prédit": -1, "probabilité": 0.0}



    feature_cols = [
        'moy_2G_3sem','std_2G_3sem',
        'moy_3G_3sem','std_3G_3sem',
        'moy_4G_3sem','std_4G_3sem',
        'Site_Critique','Zone','Constructeur','Région','Semaine'
    ]

    X_new = new_record[feature_cols]
    st.write("📊 **Features envoyées au modèle** :", X_new)

    proba = best_model.predict_proba(X_new)[:, 1][0]
    pred = best_model.predict(X_new)[0]
    return {"label_prédit": int(pred), "probabilité": float(proba)}


if st.button("🔍 Prédire le Label T+1"):
    import uuid
    row_id = str(uuid.uuid4())[:8]  # identifiant unique pour tracer la ligne

    raw_dict = {
        'row_id': row_id,  # <-- Ajout ici
        'SiteID': site_id,
        'Année': année_choisie,
        'Semaine': semaine_choisie,
        'Moyenne_2G': moy_2G,
        'Moyenne_3G': moy_3G,
        'Moyenne_4G': moy_4G,
        'Latitude': float(ligne['Latitude']),
        'Longitude': float(ligne['Longitude']),
        'Zone': ligne['Zone'],
        'Constructeur': ligne['Constructeur'],
        'Région': ligne['Région'],
        'Site_Critique': site_critique
    }


    # ➕ Nouvelle ligne pour afficher les données brutes
    # st.warning("🔍 Données avant transformation par les encoders :")
    # st.json(raw_dict)

    try:
        résultat = predict_new_record(raw_dict)

        label = résultat['label_prédit']
        proba = résultat['probabilité']

        if label == 1:
            st.error(f"⚠️ **Risque de dégradation détecté**")
            st.markdown(f"""
            - 🗓️ **Semaine T+1** : Le modèle prédit que le site **sera dégradé**.
            - 🔢 **Probabilité de dégradation** : **{proba:.2%}**
            """)
        else:
            st.success("✅ **Aucune dégradation prévue**")
            st.markdown(f"""
            - 🗓️ **Semaine T+1** : Le modèle prédit que le site **ne sera pas dégradé**.
            - 🔢 **Probabilité de dégradation** : **{proba:.2%}**
            """)

    except Exception as e:
        st.error(f"❌ Une erreur est survenue lors de la prédiction : {e}")


st.markdown("---")
st.caption("© 2025 – Votre Prédicteur Réseau en Streamlit")
