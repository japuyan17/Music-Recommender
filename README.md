# Music Recommender — Web App

Collaborative filtering recommendation system with a Flask frontend.
Trains LightFM WARP, LightFM Logistic, and KNN on startup and serves
personalized artist recommendations via a clean web UI.

## Run locally

```bash
pip3 install -r requirements.txt
python3 app.py
```

Then open http://localhost:5000

## Deploy to Render (free)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command:  `gunicorn app:app`
5. Click Deploy

## Deploy to Railway (free)

1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Select your repo — Railway auto-detects the Procfile
4. Done

## Project structure

```
music_recommender_web/
├── app.py              # Flask backend + model training
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt
├── Procfile            # For Render / Railway
└── README.md
```
