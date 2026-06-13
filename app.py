from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from surprise import KNNBasic, Reader, Dataset
from surprise.model_selection import cross_validate

app = Flask(__name__)

# ─────────────────────────────────────────────
# DATA & MODEL (loaded once at startup)
# ─────────────────────────────────────────────

ARTIST_NAMES = [
    "Taylor Swift","Drake","The Weeknd","Billie Eilish","Kendrick Lamar",
    "Ariana Grande","Post Malone","Dua Lipa","Ed Sheeran","Bad Bunny",
    "Harry Styles","Olivia Rodrigo","J. Cole","SZA","Travis Scott",
    "Doja Cat","The Beatles","Radiohead","Kanye West","Frank Ocean",
    "Tyler the Creator","Mac Miller","Arctic Monkeys","Tame Impala","Fleetwood Mac",
    "Led Zeppelin","Pink Floyd","David Bowie","Nirvana","The Strokes",
    "LCD Soundsystem","Vampire Weekend","Bon Iver","Sufjan Stevens","Phoebe Bridgers",
    "Mitski","Japanese Breakfast","Big Thief","Soccer Mommy","Snail Mail",
    "Metro Boomin","21 Savage","Lil Baby","Future","Gunna",
    "Roddy Ricch","Juice WRLD","Polo G","Lil Uzi Vert","Beyonce",
    "Rihanna","Nicki Minaj","Cardi B","Megan Thee Stallion","Bruno Mars",
    "The Chainsmokers","Calvin Harris","Marshmello","Zedd","Childish Gambino",
    "Anderson Paak","Thundercat","Flying Lotus","Kamasi Washington","Miles Davis",
    "John Coltrane","Bill Evans","Herbie Hancock","Johnny Cash","Willie Nelson",
    "Dolly Parton","Luke Combs","Morgan Wallen","Eminem","Jay-Z",
    "Nas","Biggie","Tupac","Wu-Tang Clan","A Tribe Called Quest",
    "Metallica","Black Sabbath","Iron Maiden","Red Hot Chili Peppers","Foo Fighters",
    "Pearl Jam","Soundgarden","Daft Punk","Aphex Twin","Chemical Brothers",
    "Massive Attack","Bob Marley","Damian Marley","Toots and the Maytals","Burning Spear",
]

N_USERS   = 500
N_ARTISTS = len(ARTIST_NAMES)

def generate_music_data(n_users=N_USERS, n_artists=N_ARTISTS, interactions_per_user=30, random_state=42):
    np.random.seed(random_state)
    n_genres = 10
    artist_genres        = np.random.randint(0, n_genres, size=n_artists)
    user_primary_genre   = np.random.randint(0, n_genres, size=n_users)
    user_secondary_genre = np.random.randint(0, n_genres, size=n_users)

    rows, cols, play_counts = [], [], []
    for user in range(n_users):
        pg = user_primary_genre[user]
        sg = user_secondary_genre[user]
        primary_artists   = np.where(artist_genres == pg)[0]
        secondary_artists = np.where(artist_genres == sg)[0]
        random_artists    = np.arange(n_artists)
        pool = np.concatenate([
            np.random.choice(primary_artists,   size=int(interactions_per_user * 0.60), replace=True),
            np.random.choice(secondary_artists, size=int(interactions_per_user * 0.25), replace=True),
            np.random.choice(random_artists,    size=int(interactions_per_user * 0.15), replace=True),
        ])
        pool = np.unique(pool)
        for artist in pool:
            rows.append(user)
            cols.append(artist)
            play_counts.append(np.random.randint(1, 500))

    df = pd.DataFrame({'user': rows, 'item': cols, 'play_count': play_counts})
    df['rating'] = pd.cut(df['play_count'], bins=5, labels=[1,2,3,4,5]).astype(float)
    df = df.dropna(subset=['rating'])
    return df, artist_genres, user_primary_genre

def build_matrices(df, n_users, n_artists, test_ratio=0.2):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    def to_coo(d):
        return sp.coo_matrix(
            (np.ones(len(d)), (d['user'].values, d['item'].values)),
            shape=(n_users, n_artists), dtype=np.int32
        )
    return to_coo(train_df), to_coo(test_df)

def train_lightfm_model(train, test, loss_type, epochs=20):
    model = LightFM(loss=loss_type, random_state=42)
    model.fit(train, epochs=epochs, num_threads=2, verbose=False)
    test_prec = float(precision_at_k(model, test, k=10, train_interactions=train).mean())
    test_auc  = float(auc_score(model, test, train_interactions=train).mean())
    return model, round(test_prec, 4), round(test_auc, 4)

def train_knn_model(df):
    reader       = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(df[['user','item','rating']], reader)
    knn          = KNNBasic(sim_options={'user_based': True})
    results      = cross_validate(knn, surprise_data, measures=['RMSE','MAE'], cv=3, verbose=False)
    return knn, round(float(results['test_rmse'].mean()), 4)

print("Generating data and training models...")
df, artist_genres, user_primary_genre = generate_music_data()
train_matrix, test_matrix             = build_matrices(df, N_USERS, N_ARTISTS)

warp_model,     warp_prec,     warp_auc     = train_lightfm_model(train_matrix, test_matrix, 'warp')
logistic_model, logistic_prec, logistic_auc = train_lightfm_model(train_matrix, test_matrix, 'logistic')
knn_model,      knn_rmse                    = train_knn_model(df)

knn_score   = round(1 - (knn_rmse / 4.0), 4)
train_csr   = train_matrix.tocsr()

MODEL_META = {
    'warp':     {'label': 'LightFM WARP',     'auc': warp_auc,     'precision': warp_prec,     'score': warp_auc},
    'logistic': {'label': 'LightFM Logistic', 'auc': logistic_auc, 'precision': logistic_prec, 'score': logistic_auc},
    'knn':      {'label': 'KNN (Surprise)',   'auc': None,         'precision': None,           'score': knn_score, 'rmse': knn_rmse},
}

scores  = {'warp': warp_auc, 'logistic': logistic_auc, 'knn': knn_score}
WINNER  = max(scores, key=scores.get)
print(f"Training complete. Best model: {WINNER}")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
        n_users=N_USERS,
        n_artists=N_ARTISTS,
        model_meta=MODEL_META,
        winner=WINNER,
        scores=scores
    )

@app.route('/api/recommend')
def recommend():
    user_id = int(request.args.get('user_id', 0))
    model   = request.args.get('model', 'warp')

    known_idx      = train_csr[user_id].indices.tolist()
    known_artists  = [ARTIST_NAMES[i] for i in known_idx[:5]]

    if model == 'warp':
        lfm = warp_model
    else:
        lfm = logistic_model

    scores_arr = lfm.predict(user_id, np.arange(N_ARTISTS))
    # exclude already known
    scores_arr[known_idx] = -np.inf
    top_idx    = np.argsort(-scores_arr)[:5]
    max_score  = scores_arr[top_idx[0]]
    min_score  = scores_arr[top_idx[-1]]
    rng        = max_score - min_score if max_score != min_score else 1

    recommendations = [
        {
            'artist': ARTIST_NAMES[i],
            'score':  round(float((scores_arr[i] - min_score) / rng), 3)
        }
        for i in top_idx
    ]

    return jsonify({
        'user_id':        user_id,
        'model':          model,
        'known_artists':  known_artists,
        'recommendations': recommendations,
    })

@app.route('/api/models')
def models():
    return jsonify({'models': MODEL_META, 'winner': WINNER, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
