import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# 1. Chargement du dataset
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("user_ratings_genres_mov.csv")
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return pd.DataFrame()

df = load_dataset()

st.title("Système de recommandation de films")
st.header("Création de votre profil utilisateur")
st.write("Veuillez sélectionner 3 films et leur attribuer une note.")

if df.empty:
    st.warning("Le dataset doit être chargé avant de choisir les films.")
    st.stop()

# Extraction et tri de la liste des films (unique par titre)
film_options = df[["title", "genres"]].drop_duplicates().sort_values("title")

def film_selector(label, key_prefix):
    selected_title = st.selectbox(f"{label} - Choisissez un film", film_options["title"], key=f"{key_prefix}_title")
    selected_genres = film_options[film_options["title"] == selected_title]["genres"].values[0]
    st.markdown(f"**Genres :** {selected_genres}")
    selected_rating = st.slider(f"{label} - Note", min_value=0.0, max_value=5.0, step=0.5, key=f"{key_prefix}_rating")
    return {"titre": selected_title, "genres": selected_genres, "note": selected_rating}

# Sélection des 3 films par l'utilisateur
film1 = film_selector("Film 1", "film1")
film2 = film_selector("Film 2", "film2")
film3 = film_selector("Film 3", "film3")

# Validation du profil utilisateur
if st.button("Valider mon profil"):
    titres = [film1["titre"], film2["titre"], film3["titre"]]
    if len(set(titres)) < 3:
        st.error("Veuillez choisir trois films différents.")
    else:
        profil = {"Film 1": film1, "Film 2": film2, "Film 3": film3}
        st.session_state["profil"] = profil
        st.success("Profil créé avec succès !")
        st.write("Votre profil :", profil)

# On arrête l'exécution si le profil n'est pas encore créé ou si le dataset est vide
if "profil" not in st.session_state or st.session_state["profil"] is None:
    st.warning("Veuillez créer votre profil utilisateur ci-dessus.")
    st.stop()

if df.empty:
    st.error("Le dataset est vide ou introuvable.")
    st.stop()

# 2. Préparation du dataset pour l'intégration du nouvel utilisateur
profil = st.session_state["profil"]
df_unique = df.drop_duplicates(subset="title")[["title", "genres"]]

new_user_id = "user_new"
nouveau_user = []
for film in profil.values():
    if film["titre"] and film["genres"]:
        nouveau_user.append({
            "userId": new_user_id,
            "title": film["titre"],
            "rating": film["note"],
            "genres": film["genres"]
        })

nouveau_df = pd.DataFrame(nouveau_user)
df_updated = pd.concat([df, nouveau_df], ignore_index=True)

# Création de la matrice de notes
rating_matrix = df_updated.pivot_table(index="userId", columns="title", values="rating")
rating_matrix_filled = rating_matrix.fillna(0)

# 3. Recommandation basée sur le contenu (Jaccard)
st.header("Recommandation basée sur le contenu")

def jaccard_similarity(g1, g2):
    s1 = set(g1.split("|"))
    s2 = set(g2.split("|"))
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0

# Sélection du film préféré (celui avec la note la plus élevée)
best_film = max(profil.values(), key=lambda f: f["note"], default=None)
if best_film:
    st.write("Votre film préféré :", best_film["titre"])
    df_unique["similarity"] = df_unique["genres"].apply(lambda g: jaccard_similarity(g, best_film["genres"]))
    # On exclut le film préféré et on sélectionne les 5 films les plus similaires
    recommendations = df_unique[df_unique["title"] != best_film["titre"]].nlargest(5, "similarity")
    st.write("Films recommandés (contenu) :", recommendations)
else:
    st.error("Aucun film préféré détecté.")

# 4. Recommandation collaborative – Approche mémoire (Cosine Similarity)
st.header("Recommandation basée sur la mémoire")

# Calcul de la similarité cosinus entre tous les utilisateurs
user_sim = cosine_similarity(rating_matrix_filled)
user_sim_df = pd.DataFrame(user_sim, index=rating_matrix_filled.index, columns=rating_matrix_filled.index)

# Identification des films non notés par le nouvel utilisateur (à partir de la matrice non remplie)
movies_to_predict = rating_matrix.loc[new_user_id][rating_matrix.loc[new_user_id].isna()].index
sim_new_user = user_sim_df.loc[new_user_id]

predictions_memory = {}
for movie in movies_to_predict:
    ratings = rating_matrix[movie]
    if ratings.notna().sum() == 0:
        continue
    # Moyenne pondérée des notes des autres utilisateurs
    predictions_memory[movie] = np.dot(ratings[ratings.notna()], sim_new_user[ratings.notna()]) / sim_new_user[ratings.notna()].sum()

if predictions_memory:
    reco_memory = pd.DataFrame(list(predictions_memory.items()), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    st.write("Films recommandés (mémoire) :", reco_memory)
else:
    st.error("Aucune recommandation mémoire disponible.")

# 5. Recommandation collaborative – Approche NMF
st.header("Recommandation basée sur NMF")

nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=300)
W = nmf_model.fit_transform(rating_matrix_filled)
H = nmf_model.components_
pred_nmf_df = pd.DataFrame(np.dot(W, H), index=rating_matrix.index, columns=rating_matrix.columns)

predictions_nmf = {movie: pred_nmf_df.loc[new_user_id, movie] for movie in movies_to_predict}
if predictions_nmf:
    reco_nmf = pd.DataFrame(predictions_nmf.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    st.write("Films recommandés (NMF) :", reco_nmf)
else:
    st.error("Aucune recommandation NMF disponible.")

# 6. Recommandation collaborative – Approche SVD
st.header("Recommandation basée sur SVD")

svd_model = TruncatedSVD(n_components=20, random_state=42)
U = svd_model.fit_transform(rating_matrix_filled)
VT = svd_model.components_
pred_svd_df = pd.DataFrame(np.dot(U, VT), index=rating_matrix_filled.index, columns=rating_matrix_filled.columns)

predictions_svd = {movie: pred_svd_df.loc[new_user_id, movie] for movie in movies_to_predict}
if predictions_svd:
    reco_svd = pd.DataFrame(predictions_svd.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    st.write("Films recommandés (SVD) :", reco_svd)
else:
    st.error("Aucune recommandation SVD disponible.")

# 7. Recommandation collaborative – Approche KNN
st.header("Recommandation basée sur KNN")

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(rating_matrix_filled)
distances, indices = knn_model.kneighbors(rating_matrix_filled.loc[[new_user_id]], n_neighbors=10)

neighbors = rating_matrix_filled.iloc[indices[0]]
similarities = 1 - distances[0]

predictions_knn = {}
for movie in movies_to_predict:
    neighbor_ratings = neighbors[movie]
    mask = neighbor_ratings != 0
    if not mask.any():
        continue
    weighted_sum = np.dot(neighbor_ratings[mask], similarities[mask])
    total_similarity = similarities[mask].sum()
    predictions_knn[movie] = weighted_sum / total_similarity if total_similarity else 0

if predictions_knn:
    reco_knn = pd.DataFrame(predictions_knn.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    st.write("Films recommandés (KNN) :", reco_knn)
else:
    st.error("Aucune recommandation KNN disponible.")
