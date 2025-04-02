import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors


# 1. Création du Profil Utilisateur

st.title("Système de recommandation de films")
st.header("Création de votre profil utilisateur")
st.write("Veuillez saisir vos préférences pour 3 films.")

# Saisie pour Film 1
st.subheader("Film 1")
film1_title = st.text_input("Titre du film 1", key="film1_title")
film1_genres = st.text_input("Genres du film 1 (séparés par '|')", key="film1_genres")
film1_rating = st.slider("Note pour le film 1", min_value=0.0, max_value=5.0, step=0.5, key="film1_rating")

# Saisie pour Film 2
st.subheader("Film 2")
film2_title = st.text_input("Titre du film 2", key="film2_title")
film2_genres = st.text_input("Genres du film 2 (séparés par '|')", key="film2_genres")
film2_rating = st.slider("Note pour le film 2", min_value=0.0, max_value=5.0, step=0.5, key="film2_rating")

# Saisie pour Film 3
st.subheader("Film 3")
film3_title = st.text_input("Titre du film 3", key="film3_title")
film3_genres = st.text_input("Genres du film 3 (séparés par '|')", key="film3_genres")
film3_rating = st.slider("Note pour le film 3", min_value=0.0, max_value=5.0, step=0.5, key="film3_rating")

if st.button("Valider mon profil"):
    erreurs = []
    if film1_title.strip() == "" or film1_genres.strip() == "":
        erreurs.append("Film 1 : Titre et genres requis.")
    if film2_title.strip() == "" or film2_genres.strip() == "":
        erreurs.append("Film 2 : Titre et genres requis.")
    if film3_title.strip() == "" or film3_genres.strip() == "":
        erreurs.append("Film 3 : Titre et genres requis.")
    
    if erreurs:
        for err in erreurs:
            st.error(err)
    else:
        profil = {
            "Film 1": {"titre": film1_title.strip(), "note": film1_rating, "genres": film1_genres.strip()},
            "Film 2": {"titre": film2_title.strip(), "note": film2_rating, "genres": film2_genres.strip()},
            "Film 3": {"titre": film3_title.strip(), "note": film3_rating, "genres": film3_genres.strip()},
        }
        st.session_state["profil"] = profil
        st.success("Profil créé avec succès !")
        st.write("Votre profil :", profil)

 
# 2. CHARGEMENT DU DATASET

@st.cache_data
# Le décorateur @st.cache_data permet de mettre en cache le résultat de la fonction load_dataset.
# Ainsi, lors des interactions ultérieures, le dataset ne sera pas rechargé depuis le disque, ce qui accélère l'application.
def load_dataset():
    try:
        # Tente de lire le fichier CSV "user_ratings_genres_mov.csv".
        # Ce fichier doit contenir les colonnes avec les notes attribuées par les utilisateurs, les genres et les titres des films.
        return pd.read_csv("user_ratings_genres_mov.csv")
    except Exception as e:
        # Si une exception survient (exemple: le fichier n'existe pas, erreurs de format),
        # affiche un message d'erreur dans l'application pour informer l'utilisateur.
        st.error(f"Erreur de chargement : {e}")
        # Retourne un DataFrame vide pour permettre à l'application de continuer à fonctionner sans planter.
        return pd.DataFrame()

# Appelle la fonction load_dataset pour charger les données et les stocker dans la variable 'df'.
df = load_dataset()

# Vérifie si le profil utilisateur a été créé et stocké dans st.session_state.
if "profil" not in st.session_state or st.session_state["profil"] is None:
    # Si le profil n'est pas présent, affiche un avertissement pour inciter l'utilisateur à créer son profil.
    st.warning("Veuillez créer votre profil utilisateur ci-dessus.")
    # Arrête l'exécution du script afin d'éviter les erreurs liées à l'absence de profil.
    st.stop()

# Vérifie également que le DataFrame df contient des données.
if df.empty:
    # Si le DataFrame est vide, affiche une erreur indiquant que le dataset n'est pas disponible.
    st.error("Le dataset est vide ou introuvable.")
    # Arrête l'exécution du script car il n'est pas possible de continuer sans données.
    st.stop()

# Récupère le profil utilisateur à partir de l'état de session.
# Le profil doit contenir les informations sur les films saisis par l'utilisateur (titre, note, genres).
profil = st.session_state["profil"]


# 3. PRÉPARATION DU DATASET

# Supprime les doublons du DataFrame 'df' en se basant sur la colonne "title".
# Seules les colonnes "title" et "genres" sont conservées pour créer un DataFrame de films uniques.
df_unique = df.drop_duplicates(subset="title")[["title", "genres"]]

# Définit un identifiant unique pour le nouvel utilisateur, ici "user_new".
# Cet identifiant sera utilisé pour l'intégrer dans la matrice utilisateur-film.
new_user_id = "user_new"

# Crée une liste vide qui va contenir les dictionnaires représentant les films notés par le nouvel utilisateur.
nouveau_user = []
# Parcourt toutes les entrées (films) dans le profil utilisateur.
for film in profil.values():
    # Vérifie que le film possède un titre et que les genres sont renseignés.
    if film["titre"] and film["genres"]:
        # Ajoute un dictionnaire dans la liste 'nouveau_user' contenant :
        # - "userId" : l'identifiant du nouvel utilisateur.
        # - "title"  : le titre du film.
        # - "rating" : la note donnée par l'utilisateur.
        # - "genres" : les genres du film sous forme d'une chaîne de caractères séparée par '|'.
        nouveau_user.append({
            "userId": new_user_id,
            "title": film["titre"],
            "rating": film["note"],
            "genres": film["genres"]
        })

# Convertit la liste 'nouveau_user' en DataFrame pour faciliter son intégration avec le dataset existant.
nouveau_df = pd.DataFrame(nouveau_user)

# Concatène le DataFrame original 'df' et le DataFrame 'nouveau_df' contenant les films notés par le nouvel utilisateur.
# L'argument ignore_index=True permet de réindexer le DataFrame résultant afin d'avoir un index continu.
df_updated = pd.concat([df, nouveau_df], ignore_index=True)

# Crée une matrice utilisateur-film en utilisant la fonction pivot_table de pandas.
# La matrice aura pour lignes les identifiants des utilisateurs ("userId"),
# pour colonnes les titres des films ("title") et pour valeurs les notes attribuées ("rating").
rating_matrix = df_updated.pivot_table(index="userId", columns="title", values="rating")

# Remplace toutes les valeurs manquantes (NaN) dans la matrice par 0.
# Une valeur 0 signifie que l'utilisateur n'a pas noté ce film.
rating_matrix_filled = rating_matrix.fillna(0)


# 4. RECOMMANDATION BASÉE SUR LE CONTENU (Jaccard)

# Affiche un titre dans l'interface pour indiquer la section de recommandation basée sur le contenu.
st.header("Recommandation basée sur le contenu")

# Définition d'une fonction qui calcule la similarité de Jaccard entre deux chaînes de genres.
def jaccard_similarity(g1, g2):
    # Transforme la chaîne g1 en une liste de genres en se basant sur le séparateur '|', puis convertit en ensemble.
    s1 = set(g1.split("|"))
    # Transforme la chaîne g2 en une liste de genres et la convertit en ensemble.
    s2 = set(g2.split("|"))
    # Vérifie que les deux ensembles ne sont pas vides pour éviter une division par zéro.
    # Calcule la similarité comme le ratio de la taille de l'intersection sur la taille de l'union.
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0

# Sélectionne le film préféré de l'utilisateur parmi ceux présents dans son profil.
# La fonction max() est utilisée avec une fonction lambda qui extrait la note ("note") de chaque film.
best_film = max(profil.values(), key=lambda f: f["note"], default=None)

# Vérifie si un film préféré a bien été trouvé.
if best_film:
    # Affiche le titre du film préféré dans l'interface pour que l'utilisateur le voie.
    st.write("Votre film préféré :", best_film["titre"])
    
    # Pour chaque film unique dans df_unique, calcule la similarité entre ses genres et ceux du film préféré.
    # La méthode apply() passe chaque valeur de la colonne "genres" à la fonction jaccard_similarity.
    df_unique["similarity"] = df_unique["genres"].apply(lambda g: jaccard_similarity(g, best_film["genres"]))
    
    # Exclut le film préféré de la liste et utilise la méthode nlargest() pour sélectionner les 5 films
    # ayant la valeur la plus élevée dans la colonne "similarity". Cela signifie que ce sont les films
    # les plus similaires au film préféré en termes de genres.
    recommendations = df_unique[df_unique["title"] != best_film["titre"]].nlargest(5, "similarity")
    
    # Affiche le DataFrame des films recommandés dans l'application.
    st.write("Films recommandés (contenu) :", recommendations)
else:
    # Si aucun film préféré n'est trouvé dans le profil, affiche un message d'erreur.
    st.error("Aucun film préféré détecté.")




# 5. RECOMMANDATION COLLABORATIVE – APPROCHE MÉMOIRE (Cosine Similarity)

# Affiche un titre pour indiquer que l'on passe à la recommandation collaborative basée sur la mémoire.
st.header("Recommandation basée sur la mémoire")

# Calcule la similarité cosinus entre tous les utilisateurs à partir de la matrice de notes remplie.
# La similarité cosinus mesure l'angle entre deux vecteurs, ici les vecteurs de notes des utilisateurs.
user_sim = cosine_similarity(rating_matrix_filled)

# Convertit le tableau numpy de similarités en DataFrame pour faciliter l'accès aux valeurs par identifiant.
# Les index et colonnes du DataFrame correspondent aux identifiants des utilisateurs présents dans rating_matrix_filled.
user_sim_df = pd.DataFrame(user_sim, index=rating_matrix_filled.index, columns=rating_matrix_filled.index)

# Identifie les films que le nouvel utilisateur n'a pas noté.
# rating_matrix.loc[new_user_id] récupère toutes les notes (ou absences de note) du nouvel utilisateur.
# La méthode isna() retourne True pour les films non notés, et index permet d'obtenir les titres de ces films.
movies_to_predict = rating_matrix.loc[new_user_id][rating_matrix.loc[new_user_id].isna()].index

# Récupère la ligne de similarité correspondant au nouvel utilisateur depuis user_sim_df.
# Cela donne un vecteur de similarités entre le nouvel utilisateur et chaque autre utilisateur.
sim_new_user = user_sim_df.loc[new_user_id]

# Initialise un dictionnaire vide pour stocker les prédictions des notes pour chaque film non noté.
predictions_memory = {}

# Pour chaque film dans la liste des films non notés par le nouvel utilisateur :
for movie in movies_to_predict:
    # Récupère la colonne de la matrice rating_matrix qui correspond au film courant.
    # Cette colonne contient les notes données par tous les utilisateurs pour ce film.
    ratings = rating_matrix[movie]
    
    # Vérifie s'il existe au moins une note valide (non NaN) pour ce film.
    # ratings.notna() retourne une série booléenne, et .sum() compte le nombre de valeurs True.
    if ratings.notna().sum() == 0:
        # Si aucune note n'est disponible pour ce film (aucun utilisateur n'a noté), passe au suivant.
        continue
    
    # Calcule la note prédite pour ce film en utilisant une moyenne pondérée.
    # La pondération est basée sur la similarité du nouvel utilisateur avec ceux ayant noté le film.
    # np.dot() effectue le produit scalaire entre les notes existantes et les similarités correspondantes.
    # La division par la somme des similarités normalise le résultat.
    predictions_memory[movie] = np.dot(ratings[ratings.notna()], sim_new_user[ratings.notna()]) / sim_new_user[ratings.notna()].sum()

# Si le dictionnaire predictions_memory n'est pas vide, cela signifie que des prédictions ont été réalisées.
if predictions_memory:
    # Convertit le dictionnaire en DataFrame où chaque ligne représente un film et sa note prédite.
    # La méthode nlargest(5, "predicted_rating") sélectionne les 5 films avec la plus haute note prédite.
    reco_memory = pd.DataFrame(list(predictions_memory.items()), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    
    # Affiche les films recommandés basés sur la mémoire dans l'application.
    st.write("Films recommandés (mémoire) :", reco_memory)
else:
    # Si aucune prédiction n'est disponible pour aucun film, affiche un message d'erreur.
    st.error("Aucune recommandation mémoire disponible.")





# 6. RECOMMANDATION COLLABORATIVE – APPROCHE NMF (Non-negative Matrix Factorization)

# Affiche un titre pour indiquer que l'on passe à la recommandation basée sur NMF.
st.header("Recommandation basée sur NMF")

# Initialise le modèle NMF avec les paramètres suivants :
# - n_components=20 : le nombre de facteurs latents à extraire.
# - init='random' : initialisation aléatoire des matrices W et H.
# - random_state=42 : fixe la graine aléatoire pour obtenir des résultats reproductibles.
# - max_iter=300 : nombre maximum d'itérations pour la convergence du modèle.
nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=300)

# Applique le modèle NMF à la matrice de notes remplie pour obtenir la matrice W.
# W représente l'importance ou la contribution de chaque utilisateur aux facteurs latents.
W = nmf_model.fit_transform(rating_matrix_filled)

# Récupère la matrice H à partir du modèle NMF.
# H représente l'importance de chaque film dans chacun des facteurs latents.
H = nmf_model.components_

# Reconstruit la matrice des notes prédites en multipliant les matrices W et H.
# np.dot(W, H) réalise le produit matriciel, qui est ensuite converti en DataFrame pour conserver la structure originale.
pred_nmf_df = pd.DataFrame(np.dot(W, H), index=rating_matrix.index, columns=rating_matrix.columns)

# Pour chaque film que le nouvel utilisateur n'a pas noté, extrait la note prédite depuis la matrice reconstruite.
# Utilise une compréhension de dictionnaire pour construire un dictionnaire où :
# - La clé est le titre du film.
# - La valeur est la note prédite extraite de pred_nmf_df pour le nouvel utilisateur.
predictions_nmf = {movie: pred_nmf_df.loc[new_user_id, movie] for movie in movies_to_predict}

# Si des prédictions de notes ont été générées :
if predictions_nmf:
    # Convertit le dictionnaire predictions_nmf en DataFrame et sélectionne les 5 films avec la note prédite la plus élevée.
    reco_nmf = pd.DataFrame(predictions_nmf.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    # Affiche les films recommandés basés sur NMF dans l'interface Streamlit.
    st.write("Films recommandés (NMF) :", reco_nmf)
else:
    # Si aucune prédiction n'est disponible, affiche un message d'erreur.
    st.error("Aucune recommandation NMF disponible.")




# 7. RECOMMANDATION COLLABORATIVE – APPROCHE SVD (Singular Value Decomposition)

# Affiche un titre pour la section de recommandation basée sur SVD.
st.header("Recommandation basée sur SVD")

# Initialise le modèle TruncatedSVD avec les paramètres suivants :
# - n_components=20 : nombre de composantes principales à extraire.
# - random_state=42 : fixe la graine aléatoire pour garantir la reproductibilité.
svd_model = TruncatedSVD(n_components=20, random_state=42)

# Applique le modèle SVD à la matrice de notes remplie pour obtenir la matrice U.
# U est la matrice qui représente les utilisateurs dans l'espace réduit des composantes principales.
U = svd_model.fit_transform(rating_matrix_filled)

# Récupère la matrice des composantes (VT) du modèle SVD.
# VT représente les films dans l'espace des composantes principales.
VT = svd_model.components_

# Reconstruit la matrice des notes prédites en effectuant le produit matriciel de U et VT.
# Le résultat est converti en DataFrame pour conserver les mêmes index (identifiants d'utilisateurs)
# et les mêmes colonnes (titres de films) que la matrice originale.
pred_svd_df = pd.DataFrame(np.dot(U, VT), index=rating_matrix_filled.index, columns=rating_matrix_filled.columns)

# Pour chaque film que le nouvel utilisateur n'a pas noté, récupère la note prédite depuis pred_svd_df.
predictions_svd = {movie: pred_svd_df.loc[new_user_id, movie] for movie in movies_to_predict}

# Si le dictionnaire predictions_svd n'est pas vide :
if predictions_svd:
    # Convertit predictions_svd en DataFrame et sélectionne les 5 films avec la note prédite la plus élevée.
    reco_svd = pd.DataFrame(predictions_svd.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    # Affiche les films recommandés basés sur SVD.
    st.write("Films recommandés (SVD) :", reco_svd)
else:
    # Si aucune prédiction n'est disponible, affiche un message d'erreur.
    st.error("Aucune recommandation SVD disponible.")



# 8. RECOMMANDATION COLLABORATIVE – APPROCHE KNN (K-Nearest Neighbors)

# Affiche un titre pour la section de recommandation basée sur KNN.
st.header("Recommandation basée sur KNN")

# Crée le modèle KNN avec les paramètres suivants :
# - metric='cosine' : utilise la distance cosinus pour mesurer la similitude entre les utilisateurs.
# - algorithm='brute' : utilise une approche brute force pour calculer les distances,
#   ce qui est acceptable pour un dataset de petite à moyenne taille.
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
# Entraîne le modèle KNN en utilisant la matrice de notes remplie.
knn_model.fit(rating_matrix_filled)

# Pour le nouvel utilisateur, trouve les 10 voisins les plus proches.
# rating_matrix_filled.loc[[new_user_id]] sélectionne les données du nouvel utilisateur sous forme de DataFrame.
# La méthode kneighbors retourne deux tableaux :
#   - distances : un tableau de distances entre le nouvel utilisateur et chacun des voisins trouvés.
#   - indices : un tableau des indices (position dans la matrice) des voisins.
distances, indices = knn_model.kneighbors(rating_matrix_filled.loc[[new_user_id]], n_neighbors=10)

# Utilise les indices pour récupérer les lignes correspondantes dans rating_matrix_filled, qui représentent les voisins.
neighbors = rating_matrix_filled.iloc[indices[0]]
# Convertit les distances en similarités. La similarité cosinus est définie comme 1 moins la distance cosinus.
similarities = 1 - distances[0]

# Initialise un dictionnaire vide pour stocker les prédictions de notes basées sur KNN.
predictions_knn = {}

# Parcourt chaque film que le nouvel utilisateur n'a pas noté.
for movie in movies_to_predict:
    # Récupère la colonne correspondant aux notes des voisins pour le film courant.
    neighbor_ratings = neighbors[movie]
    # Crée un masque booléen qui indique pour chaque voisin si celui-ci a noté le film (valeur différente de 0).
    mask = neighbor_ratings != 0
    # Si aucun voisin n'a noté le film (aucune valeur True dans le masque), passe au film suivant.
    if not mask.any():
        continue
    # Calcule le produit scalaire (somme pondérée) entre les notes des voisins (filtrées par le masque)
    # et les similarités correspondantes. Cette somme représente l'influence pondérée des voisins.
    weighted_sum = np.dot(neighbor_ratings[mask], similarities[mask])
    # Calcule la somme des similarités pour les voisins ayant noté le film.
    # Cette somme est utilisée pour normaliser la somme pondérée et obtenir une moyenne.
    total_similarity = similarities[mask].sum()
    # Calcule la note prédite pour le film comme étant la moyenne pondérée des notes des voisins.
    # La division est réalisée uniquement si total_similarity n'est pas zéro pour éviter une division par zéro.
    predictions_knn[movie] = weighted_sum / total_similarity if total_similarity else 0

# Si le dictionnaire predictions_knn contient des prédictions :
if predictions_knn:
    # Convertit le dictionnaire en DataFrame et sélectionne les 5 films avec la note prédite la plus élevée.
    reco_knn = pd.DataFrame(predictions_knn.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
    # Affiche les films recommandés basés sur KNN dans l'interface.
    st.write("Films recommandés (KNN) :", reco_knn)
else:
    # Si aucune prédiction n'est disponible, affiche un message d'erreur.
    st.error("Aucune recommandation KNN disponible.")
