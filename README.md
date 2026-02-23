# Product Review Sentiment Analyzer

An end-to-end web application that analyzes **product review sentiments** using classic NLP techniques.

**Upload reviews →** get sentiment predictions (Positive / Neutral / Negative) + (optional) natural language explanations.

Perfect beginner-to-intermediate full-stack ML project to learn:

- Data processing with **Pandas + DuckDB**
- Classic ML & text vectorization with **scikit-learn**
- **FastAPI** backend
- **Streamlit** frontend
- **Docker** containerization
- Deployment to **Render**

## ✨ Features

- Upload CSV, Excel or JSON files containing product reviews
- Fast sentiment classification (Positive / Neutral / Negative)
- Optional: Generate human-readable explanations via OpenAI (GPT-3.5-turbo or GPT-4o-mini)
- Interactive dashboard with:
  - Overall sentiment distribution (pie/bar charts)
  - Word clouds
  - Top positive & negative phrases
  - Example reviews per sentiment class
- Lightning-fast analytics powered by **DuckDB**
- Production-ready **Docker** setup
- Easy one-click deployment to **Render** (free tier works)

## 🛠 Tech Stack

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| Backend API        | FastAPI                                 |
| Frontend           | Streamlit                               |
| Data Processing    | Pandas + DuckDB                         |
| ML / NLP           | scikit-learn (TF-IDF + LogisticRegression / LinearSVC) |
| Optional LLM       | OpenAI (GPT-3.5-turbo / GPT-4o-mini)    |
| Containerization   | Docker                                  |
| Deployment         | Render.com                              |
| Python version     | 3.10 – 3.12                             |

## Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/product-review-sentiment.git
cd product-review-sentiment

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# or
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add OpenAI API key
# Create a file called .env in the root folder:
# OPENAI_API_KEY=sk-...

# 5. Launch the app
streamlit run app.py
# or
python -m streamlit run app.py
```

→ Open http://localhost:8501 in your browser

## Docker (Recommended for Production / Deployment)

```bash
# Build the image
docker build -t review-sentiment:latest .

# Run locally
docker run -p 8501:8501 --env-file .env review-sentiment:latest
```

For Render: just push to GitHub → connect via Render dashboard (see below).

## Project Structure

```
├── app.py                    # Streamlit frontend + main app logic
├── api/
│   ├── main.py               # FastAPI application
│   ├── models.py
│   ├── schemas.py
│   └── sentiment.py          # Model loading & prediction logic
├── notebooks/
│   └── 01-train-model.ipynb  # Model training & experiments
├── models/
│   └── sentiment_model.pkl   # Trained scikit-learn pipeline
├── data/
│   ├── raw/                  # ← Put your review CSV/Excel files here
│   └── processed/
├── utils/
│   ├── data.py               # Data loading & cleaning helpers
│   ├── viz.py                # Visualization functions
│   └── llm.py                # Optional OpenAI explanation logic
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Deployment to Render (Free Tier)

1. Push your code to a public GitHub repository
2. Go to https://render.com → New → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Runtime** → Docker
   - **Branch** → `main` (or your branch)
   - **Root Directory** → leave empty (or set if repo is in subfolder)
   - **Instance Type** → Free
5. Add environment variable (optional):
   - Key: `OPENAI_API_KEY`  
     Value: `sk-...`
6. Click **Deploy** → wait ~3–5 minutes

→ App will be live at: `https://your-app-name.onrender.com`

**Note:** Free instances sleep after ~15 minutes of inactivity. First load after sleep may take 30–60 seconds.

## Sample Datasets to Get Started Quickly

Place any of these CSV files in the `data/raw/` folder and upload via the app:

- [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) (~3–4 million reviews, polarity labels)
- [Product Reviews Dataset for Sentiment Analysis](https://www.kaggle.com/datasets/muhammadzamin1/product-reviews-dataset-for-sentiment-analysis) (small, 1,000 synthetic reviews — great for testing)
- [171k Product Reviews with Sentiment](https://www.kaggle.com/datasets/mansithummar67/171k-product-review-with-sentiment-dataset) (larger, real-world style)
- [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset) (~1.6k reviews)

## Training / Improving Your Own Model

```bash
cd notebooks
jupyter notebook
# or
jupyter lab
```

Open `01-train-model.ipynb` and follow these steps:

1. Load & clean review data
2. Train TF-IDF + Logistic Regression or LinearSVC
3. Evaluate (accuracy, precision, recall, confusion matrix)
4. Save the trained pipeline → `../models/sentiment_model.pkl`

## Possible Next Steps / Improvements

- [ ] Model versioning with MLflow or DVC
- [ ] Try transformer models (DistilBERT, RoBERTa via Hugging Face)
- [ ] Add user authentication
- [ ] Support very large files (>100k rows) with batch processing
- [ ] Export analysis report as PDF
- [ ] Dark/light theme toggle
- [ ] Show prediction confidence scores

## Contributing

Pull requests welcome!

Especially appreciated:

- Streamlit UI/UX improvements
- Faster or more accurate ML pipelines
- Better documentation & type hints

## License

MIT

---

Made with ❤️ as a beginner-friendly full-stack ML learning project  
Happy coding & deploying! 🚀
