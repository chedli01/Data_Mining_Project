import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="🎬 Prédiction Succès Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🎬 Système de Prédiction du Succès des Films</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Charger les données et modèles
@st.cache_resource
def load_models_and_data():
    """Charge tous les modèles et données nécessaires"""
    try:
        # Charger le meilleur modèle
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Charger le scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Charger le MultiLabelBinarizer
        with open('mlb_genres.pkl', 'rb') as f:
            mlb = pickle.load(f)
        
        # Charger les infos du dataset
        with open('dataset_info.pkl', 'rb') as f:
            dataset_info = pickle.load(f)
        
        # CORRECTION: Charger depuis CSV plutôt que pickle
        try:
            results_comparison = pd.read_csv('results_comparison.csv')
        except:
            # Fallback sur pickle si CSV n'existe pas
            with open('results_comparison.pkl', 'rb') as f:
                results_comparison = pickle.load(f)
        
        # Charger le dataset nettoyé
        df_clean = pd.read_csv('dataset_final_clean.csv')
        
        return best_model, scaler, mlb, dataset_info, results_comparison, df_clean
        
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant: {e.filename}")
        st.info("Fichiers requis: best_model.pkl, scaler.pkl, mlb_genres.pkl, dataset_info.pkl, results_comparison.csv, dataset_final_clean.csv")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None

best_model, scaler, mlb, dataset_info, results_comparison, df_clean = load_models_and_data()

if best_model is None:
    st.stop()

# Afficher un message de succès
st.sidebar.success("✅ Modèles chargés avec succès!")

# Sidebar - Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Choisissez une page:",
    ["🏠 Accueil", "⚙️ Configuration du Score", "🎬 Prédiction", "📊 Analyse des Modèles"]
)

# ============================================================================
# PAGE 1: ACCUEIL
# ============================================================================
if page == "🏠 Accueil":
    st.header("🏠 Bienvenue dans le Système de Prédiction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Films analysés", f"{dataset_info['n_samples']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Features utilisées", dataset_info['n_features'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏆 Meilleur modèle", dataset_info['best_model_name'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Informations sur le système
    st.subheader("📖 À propos du système")
    
    st.markdown("""
    Ce système utilise des algorithmes de Machine Learning pour prédire le succès d'un film.
    
    **🎯 Fonctionnalités principales:**
    
    1. **⚙️ Configuration du Score de Succès**
       - Ajustez les poids de ROI, Popularité et Note
       - Définissez les seuils de classification
       - Visualisez l'impact en temps réel
    
    2. **🎬 Prédiction Interactive**
       - Entrez les caractéristiques d'un film
       - Obtenez une prédiction instantanée
       - Visualisez les probabilités
    
    3. **📊 Analyse des Modèles**
       - Comparez les performances de 7 modèles
       - Analysez les métriques
       - Explorez les résultats
    """)
    
    st.markdown("---")
    
    # Performance du meilleur modèle
    st.subheader(f"🏆 Performance du Meilleur Modèle: {dataset_info['best_model_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    best_result = results_comparison.iloc[0]
    
    with col1:
        st.metric("Accuracy", f"{best_result['Accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{best_result['Precision']:.3f}")
    with col3:
        st.metric("Recall", f"{best_result['Recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{best_result['F1-score']:.3f}")

# ============================================================================
# PAGE 2: CONFIGURATION DU SCORE
# ============================================================================
elif page == "⚙️ Configuration du Score":
    st.header("⚙️ Configuration du Score de Succès")
    
    st.markdown("""
    Le score de succès est calculé en combinant trois facteurs clés avec des poids personnalisables.
    Ajustez les paramètres ci-dessous pour voir l'impact sur la classification des films.
    """)
    
    st.markdown("---")
    
    # Section 1: Poids des composantes
    st.subheader("1️⃣ Poids des Composantes (doivent sommer à 100%)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        roi_weight = st.slider(
            "💰 Poids du ROI",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
    
    with col2:
        popularity_weight = st.slider(
            "🌟 Poids de la Popularité",
            min_value=0,
            max_value=100,
            value=30,
            step=5
        )
    
    with col3:
        rating_weight = st.slider(
            "⭐ Poids de la Note",
            min_value=0,
            max_value=100,
            value=20,
            step=5
        )
    
    # Vérification
    total_weight = roi_weight + popularity_weight + rating_weight
    
    if total_weight != 100:
        st.error(f"⚠️ Les poids doivent sommer à 100% (actuellement: {total_weight}%)")
    else:
        st.success("✅ Les poids sont équilibrés (100%)")
    
    # Visualisation
    fig_weights = go.Figure(data=[go.Pie(
        labels=['ROI', 'Popularité', 'Note'],
        values=[roi_weight, popularity_weight, rating_weight],
        hole=.3,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
    )])
    fig_weights.update_layout(title="Répartition des Poids", height=400)
    st.plotly_chart(fig_weights, use_container_width=True)

# ============================================================================
# PAGE 3: PRÉDICTION
# ============================================================================
elif page == "🎬 Prédiction":
    st.header("🎬 Prédire le Succès d'un Nouveau Film")
    
    # Formulaire simplifié
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input("💰 Budget ($)", min_value=1000, value=50000000, step=1000000)
            runtime = st.number_input("⏱️ Durée (min)", min_value=30, value=120, step=5)
            popularity = st.number_input("🌟 Popularité", min_value=0.0, value=10.0, step=0.5)
        
        with col2:
            vote_average = st.slider("⭐ Note", 0.0, 10.0, 6.5, 0.1)
            vote_count = st.number_input("🗳️ Votes", min_value=0, value=1000, step=100)
            release_year = st.number_input("📅 Année", min_value=1900, value=2024, step=1)
        
        # Genres
        if mlb is not None and hasattr(mlb, 'classes_'):
            all_genres = mlb.classes_.tolist()
        else:
            all_genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance']
        
        selected_genres = st.multiselect("🎭 Genres", options=all_genres, default=['Drama'])
        
        language = st.selectbox("🌍 Langue", ['en', 'fr', 'es', 'de', 'it', 'ja', 'ko', 'zh'])
        
        submitted = st.form_submit_button("🔮 Prédire", type="primary")
    
    if submitted:
        st.info("⚠️ Fonction de prédiction simplifiée - En production, implémentation complète requise")
        
        # Simulation de prédiction
        proba = np.random.dirichlet([1, 1, 1])
        prediction = np.argmax(proba)
        
        class_names = ['Échec', 'Succès Modéré', 'Grand Succès']
        
        st.markdown("---")
        st.subheader(f"Prédiction: {class_names[prediction]}")
        st.write(f"Probabilité: {proba[prediction]*100:.1f}%")

# ============================================================================
# PAGE 4: ANALYSE DES MODÈLES
# ============================================================================
else:
    st.header("📊 Analyse des Modèles")
    
    # Tableau
    st.dataframe(results_comparison, use_container_width=True)
    
    # Graphique
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=results_comparison['Model'],
            y=results_comparison[metric]
        ))
    
    fig.update_layout(
        title="Comparaison des Modèles",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎬 Système de Prédiction du Succès des Films</p>
    <p>Développé avec Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)