import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from surprise import KNNBasic, Reader, Dataset
from surprise.model_selection import cross_validate

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC MUSIC DATASET
# ─────────────────────────────────────────────

def generate_music_data(n_users=500, n_artists=200, interactions_per_user=30, random_state=42):
    """
    Generates a synthetic Last.fm-style dataset.
    Each user has a genre preference, and artists belong to genres,
    so recommendations are meaningful rather than purely random.
    """
    np.random.seed(random_state)

    artist_names = [
        "Taylor Swift", "Drake", "The Weeknd", "Billie Eilish", "Kendrick Lamar",
        "Ariana Grande", "Post Malone", "Dua Lipa", "Ed Sheeran", "Bad Bunny",
        "Harry Styles", "Olivia Rodrigo", "J. Cole", "SZA", "Travis Scott",
        "Doja Cat", "The Beatles", "Radiohead", "Kanye West", "Frank Ocean",
        "Tyler the Creator", "Mac Miller", "Arctic Monkeys", "Tame Impala", "Fleetwood Mac",
        "Led Zeppelin", "Pink Floyd", "David Bowie", "Nirvana", "The Strokes",
        "LCD Soundsystem", "Vampire Weekend", "Bon Iver", "Sufjan Stevens", "Phoebe Bridgers",
        "Mitski", "Japanese Breakfast", "Big Thief", "Soccer Mommy", "Snail Mail",
        "Metro Boomin", "21 Savage", "Lil Baby", "Future", "Gunna",
        "Roddy Ricch", "NBA YoungBoy", "Polo G", "Juice WRLD", "Lil Uzi Vert",
        "Beyonce", "Rihanna", "Nicki Minaj", "Cardi B", "Megan Thee Stallion",
        "Bruno Mars", "The Chainsmokers", "Calvin Harris", "Marshmello", "Zedd",
        "Childish Gambino", "Anderson Paak", "Thundercat", "Flying Lotus", "Kamasi Washington",
        "Miles Davis", "John Coltrane", "Bill Evans", "Herbie Hancock", "Charles Mingus",
        "Bach", "Mozart", "Beethoven", "Chopin", "Debussy",
        "Johnny Cash", "Willie Nelson", "Dolly Parton", "Hank Williams", "Merle Haggard",
        "Luke Combs", "Morgan Wallen", "Blake Shelton", "Garth Brooks", "Kenny Rogers",
        "Eminem", "Jay-Z", "Nas", "Biggie", "Tupac",
        "Wu-Tang Clan", "A Tribe Called Quest", "De La Soul", "Public Enemy", "Ice Cube",
        "Metallica", "Black Sabbath", "Iron Maiden", "Slayer", "Megadeth",
        "Red Hot Chili Peppers", "Foo Fighters", "Pearl Jam", "Soundgarden", "Alice in Chains",
        "Daft Punk", "Justice", "Aphex Twin", "Chemical Brothers", "Massive Attack",
        "Bob Marley", "Damian Marley", "Toots and the Maytals", "Steel Pulse", "Burning Spear",
    ] + [f"Artist_{i}" for i in range(100, n_artists)]  # pad to n_artists

    artist_names = artist_names[:n_artists]

    # assign each artist to a genre cluster (0-9)
    n_genres = 10
    artist_genres = np.random.randint(0, n_genres, size=n_artists)

    # assign each user a primary and secondary genre preference
    user_primary_genre   = np.random.randint(0, n_genres, size=n_users)
    user_secondary_genre = np.random.randint(0, n_genres, size=n_users)

    rows, cols, play_counts = [], [], []

    for user in range(n_users):
        pg = user_primary_genre[user]
        sg = user_secondary_genre[user]

        # pool of preferred artists (primary genre 60%, secondary 25%, random 15%)
        primary_artists   = np.where(artist_genres == pg)[0]
        secondary_artists = np.where(artist_genres == sg)[0]
        random_artists    = np.arange(n_artists)

        pool = np.concatenate([
            np.random.choice(primary_artists,   size=int(interactions_per_user * 0.60), replace=True),
            np.random.choice(secondary_artists, size=int(interactions_per_user * 0.25), replace=True),
            np.random.choice(random_artists,    size=int(interactions_per_user * 0.15), replace=True),
        ])

        # deduplicate
        pool = np.unique(pool)

        for artist in pool:
            play_count = np.random.randint(1, 500)
            rows.append(user)
            cols.append(artist)
            play_counts.append(play_count)

    df = pd.DataFrame({'user': rows, 'item': cols, 'play_count': play_counts})

    # normalize play counts to 1-5 rating scale
    df['rating'] = pd.cut(df['play_count'], bins=5, labels=[1, 2, 3, 4, 5]).astype(float)
    df = df.dropna(subset=['rating'])

    print(f"Generated dataset — Users: {n_users}, Artists: {n_artists}, Interactions: {len(df)}")

    return df, n_users, n_artists, {i: name for i, name in enumerate(artist_names)}

# ─────────────────────────────────────────────
# 2. BUILD INTERACTION MATRIX FOR LIGHTFM
# ─────────────────────────────────────────────

def build_interaction_matrix(df, n_users, n_items, test_ratio=0.2):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    def to_coo(data):
        return sp.coo_matrix(
            (np.ones(len(data)), (data['user'].values, data['item'].values)),
            shape=(n_users, n_items),
            dtype=np.int32
        )

    return to_coo(train_df), to_coo(test_df)

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE LIGHTFM MODELS
# ─────────────────────────────────────────────

def train_lightfm(train, test, loss_type, epochs=20):
    print(f"\nTraining LightFM ({loss_type.upper()})...")
    model = LightFM(loss=loss_type, random_state=42)
    model.fit(train, epochs=epochs, num_threads=2, verbose=False)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision  = precision_at_k(model, test, k=10, train_interactions=train).mean()
    train_auc       = auc_score(model, train).mean()
    test_auc        = auc_score(model, test, train_interactions=train).mean()

    print(f"  Train Precision@10: {train_precision:.4f} | Test Precision@10: {test_precision:.4f}")
    print(f"  Train AUC:          {train_auc:.4f}       | Test AUC:          {test_auc:.4f}")

    return model, test_precision, test_auc

# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE KNN (SURPRISE)
# ─────────────────────────────────────────────

def train_knn(df):
    print("\nTraining KNN (Surprise)...")
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

    knn = KNNBasic(sim_options={'user_based': True})
    results = cross_validate(knn, surprise_data, measures=['RMSE', 'MAE'], cv=3, verbose=False)

    rmse = results['test_rmse'].mean()
    mae  = results['test_mae'].mean()
    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    return knn, rmse, mae

# ─────────────────────────────────────────────
# 5. COMPARE MODELS & PRINT WINNER
# ─────────────────────────────────────────────

def compare_models(warp_auc, logistic_auc, knn_rmse):
    print("\n" + "="*55)
    print("           MODEL COMPARISON RESULTS")
    print("="*55)
    print(f"  LightFM WARP     — Test AUC:  {warp_auc:.4f}  (higher = better)")
    print(f"  LightFM Logistic — Test AUC:  {logistic_auc:.4f}  (higher = better)")
    print(f"  KNN (Surprise)   — RMSE:      {knn_rmse:.4f}  (lower  = better)")
    print("="*55)

    # normalize KNN RMSE to a 0-1 score so all 3 are comparable
    knn_score = 1 - (knn_rmse / 4.0)

    scores = {
        "LightFM WARP":     warp_auc,
        "LightFM Logistic": logistic_auc,
        "KNN (Surprise)":   knn_score,
    }

    winner = max(scores, key=scores.get)
    print(f"\n  BEST MODEL: {winner}")
    print(f"  Normalized scores — WARP: {warp_auc:.4f}, "
          f"Logistic: {logistic_auc:.4f}, KNN: {knn_score:.4f}")
    print("="*55)
    return winner

# ─────────────────────────────────────────────
# 6. SAMPLE RECOMMENDATIONS FROM BEST MODEL
# ─────────────────────────────────────────────

def sample_recommendation(model, train, item_labels, user_ids, model_name):
    print(f"\n--- Sample Recommendations ({model_name}) ---")
    n_users, n_items = train.shape

    for user_id in user_ids:
        known_positives_idx = train.tocsr()[user_id].indices
        known_positives = [item_labels.get(i, f"Artist {i}") for i in known_positives_idx[:3]]

        scores = model.predict(user_id, np.arange(n_items))
        top_idx = np.argsort(-scores)[:3]
        recommendations = [item_labels.get(i, f"Artist {i}") for i in top_idx]

        print(f"\nUser {user_id}")
        print("     Known positives:")
        for name in known_positives:
            print(f"           {name}")
        print("     Recommended:")
        for name in recommendations:
            print(f"           {name}")

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # generate synthetic music data
    df, n_users, n_items, artist_labels = generate_music_data(
        n_users=500,
        n_artists=200,
        interactions_per_user=30
    )

    # build matrices for LightFM
    train_matrix, test_matrix = build_interaction_matrix(df, n_users, n_items)

    # train all 3 models
    warp_model,     warp_precision,     warp_auc     = train_lightfm(train_matrix, test_matrix, loss_type='warp')
    logistic_model, logistic_precision, logistic_auc = train_lightfm(train_matrix, test_matrix, loss_type='logistic')
    knn_model,      knn_rmse,           knn_mae      = train_knn(df)

    # compare and print winner
    winner = compare_models(warp_auc, logistic_auc, knn_rmse)

    # show recommendations from best LightFM model
    best_lightfm = warp_model if "WARP" in winner else logistic_model
    best_name    = "LightFM WARP" if "WARP" in winner else "LightFM Logistic"

    sample_recommendation(best_lightfm, train_matrix, artist_labels, [0, 1, 2], best_name)