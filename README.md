# Jamboree Admission Streamlit App

This repository contains a Streamlit dashboard for Jamboree Admission analysis and a prediction UI that uses pre-trained sklearn pipelines saved as pickle files.

Files of interest:
- `index.py` - Streamlit app
- `Jamboree_Admission.csv` - dataset used for analysis
- `model_pipeline.pkl`, `Polynomial_pipeline.pkl`, `Ridge_pipeline.pkl` - saved sklearn pipelines used for prediction

Requirements
- Python 3.9+
- See `requirements.txt` for exact dependencies

Deployment options

1) Streamlit Community Cloud (fastest)
- Create a GitHub repo (if not already), push your project.
- On https://streamlit.io/cloud, connect your GitHub repo and deploy the app.
- Make sure `requirements.txt` is present and the three `.pkl` files are committed or accessible in the repo.

Notes: Streamlit Cloud has repository size limits. If your pickles are large, consider storing them in cloud storage (S3/GCS) and loading them at runtime.

2) Render (simple Docker-free)
- Create a new Web Service on Render, connect GitHub repo.
- Set the start command to: `streamlit run index.py --server.port $PORT --server.headless true`
- Set `PORT` environment variable if necessary.

3) Docker (most portable)
- Build and run locally:
  docker build -t jamboree-app .
  docker run -p 8501:8501 jamboree-app
- Push to a container registry and deploy on Cloud Run, ECS, or DigitalOcean App Platform.

Model files and large assets
- If your `.pkl` files are >100MB, do NOT commit them to GitHub. Instead:
  - Upload to cloud storage (S3 / GCS / Azure Blob) and load them by URL inside `index.py` using requests and pickle.loads.
  - Or use Git LFS to store large files.

Security
- Do not expose private credentials in the repo.
- If loading models from cloud storage, use short-lived or restricted credentials.

Troubleshooting
- If the app crashes on startup, check logs for FileNotFoundError for model files or missing dependencies.
- Ensure the working directory when deploying contains `index.py` and the `requirements.txt`.

If you tell me which provider you prefer (Streamlit Cloud, Render, Docker->Cloud Run, or something else), I can walk through exact steps and create any small scripts (Procfile, app.yaml) needed for that provider.
