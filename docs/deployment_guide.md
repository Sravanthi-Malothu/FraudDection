# Free Cloud Deployment Guide: Render & Hugging Face Spaces

This guide provides step-by-step instructions on deploying the **Sentra Fraud Detection & Alert Investigation Platform** to free cloud hosting platforms (**Render** or **Hugging Face Spaces**).

---

## ⚡ Fast Startup Architecture

Cloud free tiers impose strict resource limits (Render provides 512 MB RAM; Hugging Face provides 16 GB RAM). To ensure cold starts take **less than 2 seconds** with zero runtime latency:

1. **Lightweight Model Footprint:** Serialized model artifacts (`xgboost.joblib`, `random_forest.joblib`, `ensemble.joblib`, etc.) total only **4.8 MB** and are bundled directly in the repository.
2. **Pre-Scored Alert Queue (`data/alerts_cache.json`):** 53 high-risk real test cases pre-scored with SHAP attributions (321 KB) load in sub-10 milliseconds without requiring large CSV file parsing.
3. **Curated Transaction Sample (`data/sample_test_transactions.json`):** 97 real transactions (35 KB) provide instant population of the Transactions tab without downloading or copying the 143 MB `fraudTest.csv`.
4. **Universal WSGI Binding:** Gunicorn dynamically binds to `${PORT:-7860}`, automatically accommodating Hugging Face Spaces (port `7860`), Render (dynamic `$PORT`), and local Docker (`5001`).

---

## 🌐 Option 1: Deploy to Hugging Face Spaces (Recommended)

Hugging Face Spaces offers **16 GB RAM and 2 vCPUs** on its free tier, making it the most responsive hosting option.

### Exactly What to Click:

1. **Sign Up / Log In:** Go to [huggingface.co](https://huggingface.co) and sign in.
2. **Create New Space:**
   - Click on your profile picture in the top-right corner.
   - Click **"New Space"** (or visit [huggingface.co/new-space](https://huggingface.co/new-space)).
3. **Configure Space Settings:**
   - **Space name:** Enter `sentra-fraud-detection`
   - **License:** Select `MIT`
   - **Select the Space SDK:** Click on **`Docker`** *(Blank)*
   - **Space Hardware:** Leave at default **`CPU basic • 2 vCPU • 16 GB RAM • Free`**
   - **Privacy:** Select **`Public`**
4. **Click "Create Space"** (black button at the bottom).
5. **Push Code to Your Space:**
   In your local terminal, add the Hugging Face Space git remote and push:
   ```bash
   # Replace <YOUR_HF_USERNAME> with your actual Hugging Face username
   git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/sentra-fraud-detection

   # Push your current branch (ml-upgrade) to the Space's main branch
   git push space ml-upgrade:main
   ```
   *(If prompted for credentials, use your Hugging Face username and an Access Token from **Settings → Access Tokens** as your password).*
6. **Watch the Build:**
   - Hugging Face automatically detects the `Dockerfile`, installs dependencies, and starts Gunicorn on port `7860`.
   - In ~90 seconds, the status pill changes from **"Building"** to **"Running"**.
   - Your interactive dashboard is live at:
     ```
     https://huggingface.co/spaces/<YOUR_HF_USERNAME>/sentra-fraud-detection
     ```

---

## 🚀 Option 2: Deploy to Render (Free Web Service)

Render offers free web service hosting directly from your GitHub repository.

### Method A: 1-Click Blueprint Deploy (`render.yaml`)

1. **Push to GitHub:** Ensure your code is pushed to your personal GitHub repository.
2. **Log in to Render:** Go to [dashboard.render.com](https://dashboard.render.com) (sign up with GitHub).
3. **Click "New +" Button:** In the top navigation bar, click the blue **"New +"** button.
4. **Select "Blueprint":** Click **"Blueprint"** from the dropdown menu.
5. **Connect Repository:**
   - Select your `FraudDetection` repository.
   - Render automatically reads `render.yaml` from the root directory.
6. **Click "Apply":** Review the service name (`sentra-fraud-detection`) and click **"Apply"**.
7. Render will build the environment and deploy your app. When the status turns green (**"Live"**), click the generated URL (e.g. `https://sentra-fraud-detection.onrender.com`).

---

### Method B: Manual Web Service Setup on Render

If you prefer to configure the service manually through Render's dashboard:

1. In the Render Dashboard, click **"New +"** → **"Web Service"**.
2. Select **"Build and deploy from a Git repository"** and click **Next**.
3. Choose your repository and click **Connect**.
4. Configure the settings:
   | Setting | Value |
   |---|---|
   | **Name** | `sentra-fraud-detection` |
   | **Region** | Oregon (US West) or Frankfurt (EU Central) |
   | **Branch** | `main` (or `ml-upgrade`) |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 60 app.app:app` |
   | **Instance Type** | **Free** ($0 / month) |
5. Click **"Advanced"** → **"Add Environment Variable"**:
   - Key: `PYTHON_VERSION`
   - Value: `3.11.9`
6. Click **"Create Web Service"** at the bottom of the page.
7. Wait 2–3 minutes for the build to finish. Once live, open your public `.onrender.com` URL.

> [!NOTE]
> **Render Free Tier Sleep Policy:** Render spins down free web services after 15 minutes of inactivity. When a new request arrives, Render spins the container back up in 30–45 seconds. Hugging Face Spaces does not spin down while marked active.

---

## ✅ Deployment Verification Checklist

Once deployed, verify your service is operating properly:

### 1. Health Endpoint Check
Visit `https://<YOUR_APP_URL>/health` in your browser. Expected response:
```json
{
  "active_models": [
    "xgboost",
    "random_forest",
    "ensemble",
    "logistic_regression"
  ],
  "database": "connected (sqlite)",
  "model_version": "xgboost-v1.0",
  "models_loaded": true,
  "project": "Sentra Fraud & AML Intelligence Platform",
  "status": "healthy",
  "total_alerts_queued": 53
}
```

### 2. Alert Queue & Investigation Tab
- Open the root URL `https://<YOUR_APP_URL>/`.
- Confirm the priority queue displays alerts sorted descending by risk score (e.g. `ALT-10482`, `ALT-10483`).
- Verify currency displays in USD (`$`).
- Verify the **AI Insight** card and **Risk Triggers** display real SHAP attributions.

### 3. Real-Time SQLite Decision Capture
- Select an alert (e.g. `ALT-10482`).
- Click **"Review decision →"** and select **Approve**, **Block / Freeze Card**, or **Escalate**.
- Refresh the browser page: the status badge and timeline update persist permanently from SQLite.

### 4. Direct Scoring via REST API
Test model inference via `curl`:
```bash
curl -X POST "https://<YOUR_APP_URL>/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amt": 985.50,
       "category": "shopping_net",
       "city_pop": 45000,
       "lat": 36.16,
       "long": -86.78,
       "hour": 23,
       "age": 31,
       "gender": "F"
     }'
```
Expected response:
```json
{
  "confidence": 98.2,
  "is_fraud": 1,
  "model_version": "xgboost-v1.0",
  "risk_level": "High risk",
  "risk_score": 98,
  "triggers": [...]
}
```

---

## 🛠️ Local Docker Testing Before Deployment

You can test the exact container image locally using Docker:

```bash
# 1. Build the production container
docker build -t sentra-fraud-api .

# 2. Run container on port 7860
docker run -p 7860:7860 -e PORT=7860 sentra-fraud-api

# 3. Test health check
curl http://localhost:7860/health
```
Navigate to `http://localhost:7860` to preview the production build.
