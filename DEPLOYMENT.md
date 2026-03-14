# GridSense — Deployment Guide

Complete walkthrough for going from local Docker stack to fully public project.

---

## Step 1 — Push to GitHub

### 1.1 Create the repository

1. Go to https://github.com/new
2. Repository name: `gridsense`
3. Visibility: **Public**
4. Do NOT initialize with README, .gitignore, or license (we have all of these)
5. Click **Create repository**

### 1.2 Initialize git and push

```bat
cd C:\Users\luizg\Downloads\ProjetoGithub\gridsense

git init
git add .
git commit -m "feat: initial GridSense release v0.1.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/gridsense.git
git push -u origin main
```

### 1.3 Set up Codecov (coverage badge)

1. Go to https://codecov.io and sign in with GitHub
2. Click **Add repository** → select `gridsense`
3. Copy your **CODECOV_TOKEN**
4. In GitHub: go to your repo → **Settings** → **Secrets and variables** → **Actions**
5. Click **New repository secret**:
   - Name: `CODECOV_TOKEN`
   - Value: paste the token

### 1.4 Verify CI runs

After pushing, go to your repo on GitHub:
- Click the **Actions** tab
- You should see the CI workflow running automatically
- Wait for the green checkmark ✓

### 1.5 Update badge URLs in README.md

Replace `YOUR_USERNAME` in README.md with your actual GitHub username:

```bash
# On Windows — find and replace in your editor (VS Code: Ctrl+H)
# Replace: YOUR_USERNAME
# With:    your-actual-github-username
```

Then push the update:
```bat
git add README.md
git commit -m "docs: update badge URLs with real GitHub username"
git push
```

Once CI passes, your README will show live green badges. ✓

---

## Step 2 — Publish to PyPI

### 2.1 Create PyPI account

1. Go to https://pypi.org/account/register/
2. Create an account and verify your email

### 2.2 Create an API token

1. Go to https://pypi.org/manage/account/token/
2. Token name: `gridsense-release`
3. Scope: **Entire account** (for first upload; restrict to project after)
4. Copy the token — it starts with `pypi-`

### 2.3 Add PyPI token to GitHub Secrets

In GitHub: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
- Name: `PYPI_TOKEN`
- Value: paste the `pypi-` token

### 2.4 Build and check the package locally first

```bat
pip install build twine

python -m build

twine check dist/*
```

Expected output: `PASSED gridsense-0.1.0-py3-none-any.whl` — no warnings.

### 2.5 Do a test upload to TestPyPI (optional but recommended)

```bat
twine upload --repository testpypi dist/*
```

Visit https://test.pypi.org/project/gridsense/ to confirm it looks correct.

### 2.6 Publish the real release via git tag

```bat
git tag v0.1.0
git push origin v0.1.0
```

This triggers `.github/workflows/release.yml` automatically.
GitHub Actions will build and upload to PyPI within ~2 minutes.

### 2.7 Verify

```bat
pip install gridsense
python -c "import gridsense; print(gridsense.__version__)"
```

Expected: `0.1.0`

Your package is now at: https://pypi.org/project/gridsense/

---

## Step 3 — Deploy Dashboard to Streamlit Cloud

### 3.1 Sign up for Streamlit Cloud

Go to https://share.streamlit.io and sign in with GitHub.

### 3.2 Create a requirements file for Streamlit Cloud

Streamlit Cloud doesn't use pyproject.toml — it needs a plain `requirements.txt`.
We already have this handled via `packages.txt` and `requirements.txt` (see files below).

### 3.3 Deploy

1. Click **New app**
2. Repository: `YOUR_USERNAME/gridsense`
3. Branch: `main`
4. Main file path: `dashboard/app.py`
5. Click **Advanced settings**:
   - Add secret: `API_URL` = your public API URL (from Step 4), or leave blank to use demo data
6. Click **Deploy**

Streamlit Cloud will install dependencies and launch your app.
Your public URL will be: `https://YOUR_USERNAME-gridsense-dashboard-app-XXXXX.streamlit.app`

### 3.4 Update README with the live URL

```markdown
**Live dashboard →** [your-app.streamlit.app](https://your-app.streamlit.app)
```

---

## Step 4 — Deploy API to Fly.io

### 4.1 Install the Fly CLI

```bat
# Windows (PowerShell as Administrator)
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

Or download from: https://fly.io/docs/flyctl/install/

### 4.2 Create a Fly.io account and log in

```bat
fly auth signup   # or: fly auth login
```

### 4.3 Launch the app

From inside the `gridsense/` folder:

```bat
fly launch \
  --name gridsense-api \
  --dockerfile docker/Dockerfile.api \
  --region gru \
  --no-deploy
```

- `gru` = São Paulo region (closest to Florianópolis)
- `--no-deploy` lets you review the config first

This creates a `fly.toml` file (already provided below).

### 4.4 Set environment variables

```bat
fly secrets set DATABASE_URL="postgresql://gridsense:gridsense@your-db-host:5432/gridsense"
```

For a fully public demo without a database, the API falls back gracefully to synthetic data.

### 4.5 Deploy

```bat
fly deploy --dockerfile docker/Dockerfile.api
```

Your API will be live at: `https://gridsense-api.fly.dev`
Swagger docs: `https://gridsense-api.fly.dev/docs`

### 4.6 Verify

```bat
curl https://gridsense-api.fly.dev/healthz
```

Expected: `{"status":"ok","version":"0.1.0"}`

### 4.7 Update Streamlit Cloud secret

Go back to your Streamlit app settings and update:
- `API_URL` = `https://gridsense-api.fly.dev`

The dashboard will now show live API data instead of demo data.

---

## Final checklist

After completing all four steps:

```
✓ git push → CI runs → green badge in README
✓ git tag v0.1.0 → PyPI release → pip install gridsense works
✓ dashboard.streamlit.app → public live dashboard
✓ gridsense-api.fly.dev/docs → public live API
```

Update your README.md with the live URLs, commit, and push.
Then open a GitHub Issue with label `good first issue` to start attracting contributors.
