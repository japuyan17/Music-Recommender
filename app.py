from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from surprise import KNNBasic, SVD, NMF, Reader, Dataset
from surprise.model_selection import cross_validate
import scipy.sparse as sp

app = Flask(__name__)

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

N_USERS = 500
N_ARTISTS = len(ARTIST_NAMES)


def generate_data(n_users=N_USERS, n_artists=N_ARTISTS, interactions_per_user=30, seed=42):
    np.random.seed(seed)
    n_genres = 10
    artist_genres = np.random.randint(0, n_genres, size=n_artists)
    user_primary_genre = np.random.randint(0, n_genres, size=n_users)
    user_secondary_genre = np.random.randint(0, n_genres, size=n_users)
    rows, cols, ratings = [], [], []
    for user in range(n_users):
        pg = user_primary_genre[user]
        sg = user_secondary_genre[user]
        primary = np.where(artist_genres == pg)[0]
        secondary = np.where(artist_genres == sg)[0]
        pool = np.unique(np.concatenate([
            np.random.choice(primary, size=int(interactions_per_user * 0.60), replace=True),
            np.random.choice(secondary, size=int(interactions_per_user * 0.25), replace=True),
            np.random.choice(np.arange(n_artists), size=int(interactions_per_user * 0.15), replace=True),
        ]))
        for artist in pool:
            rows.append(user)
            cols.append(int(artist))
            ratings.append(np.random.randint(1, 6))
    df = pd.DataFrame({"user": rows, "item": cols, "rating": ratings})
    return df


def train_models(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

    print("Training KNN...")
    knn = KNNBasic(sim_options={"user_based": True})
    knn_res = cross_validate(knn, data, measures=["RMSE"], cv=3, verbose=False)
    knn_rmse = round(float(knn_res["test_rmse"].mean()), 4)

    print("Training SVD...")
    svd = SVD(random_state=42)
    svd_res = cross_validate(svd, data, measures=["RMSE"], cv=3, verbose=False)
    svd_rmse = round(float(svd_res["test_rmse"].mean()), 4)

    print("Training NMF...")
    nmf = NMF(random_state=42)
    nmf_res = cross_validate(nmf, data, measures=["RMSE"], cv=3, verbose=False)
    nmf_rmse = round(float(nmf_res["test_rmse"].mean()), 4)

    trainset = data.build_full_trainset()
    knn.fit(trainset)
    svd.fit(trainset)
    nmf.fit(trainset)

    return (knn, knn_rmse), (svd, svd_rmse), (nmf, nmf_rmse)


print("Generating data...")
df = generate_data()

(knn_model, knn_rmse), (svd_model, svd_rmse), (nmf_model, nmf_rmse) = train_models(df)


def rmse_to_score(r):
    return round(1 - (r / 4.0), 4)


MODEL_META = {
    "knn": {"label": "KNN", "rmse": knn_rmse, "score": rmse_to_score(knn_rmse)},
    "svd": {"label": "SVD", "rmse": svd_rmse, "score": rmse_to_score(svd_rmse)},
    "nmf": {"label": "NMF", "rmse": nmf_rmse, "score": rmse_to_score(nmf_rmse)},
}
WINNER = min(MODEL_META, key=lambda k: MODEL_META[k]["rmse"])
print(f"Best model: {WINNER} (RMSE {MODEL_META[WINNER]['rmse']})")

user_item = sp.coo_matrix(
    (df["rating"].values, (df["user"].values, df["item"].values)),
    shape=(N_USERS, N_ARTISTS),
).tocsr()

MODELS = {"knn": knn_model, "svd": svd_model, "nmf": nmf_model}


@app.route("/")
def index():
    return render_template(
        "index.html",
        n_users=N_USERS,
        n_artists=N_ARTISTS,
        model_meta=MODEL_META,
        winner=WINNER,
    )


@app.route("/api/recommend")
def recommend():
    user_id = int(request.args.get("user_id", 0))
    model = request.args.get("model", WINNER)
    m = MODELS[model]

    known_idx = user_item[user_id].indices.tolist()
    known_artists = [ARTIST_NAMES[i] for i in known_idx[:5]]
    known_set = set(known_idx)

    candidates = [i for i in range(N_ARTISTS) if i not in known_set]
    preds = [(i, m.predict(user_id, i).est) for i in candidates]
    preds.sort(key=lambda x: -x[1])
    top5 = preds[:5]

    max_s = top5[0][1]
    min_s = top5[-1][1]
    rng = max_s - min_s if max_s != min_s else 1

    recommendations = [
        {"artist": ARTIST_NAMES[i], "score": round((s - min_s) / rng, 3)}
        for i, s in top5
    ]
    return jsonify(
        {
            "user_id": user_id,
            "model": model,
            "known_artists": known_artists,
            "recommendations": recommendations,
        }
    )


@app.route("/api/models")
def models():
    return jsonify({"models": MODEL_META, "winner": WINNER})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
