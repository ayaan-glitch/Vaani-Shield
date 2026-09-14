# Deploying Flutter Web to Vercel

Yes! You can deploy the Vaani-Shield Flutter Web frontend to Vercel smoothly. Flutter Web compiles into static HTML, JavaScript, WebAssembly, and assets (`frontend/build/web`), which Vercel serves with ultra-low latency worldwide.

---

## Method 1: Deploy with Vercel CLI (Fastest & Simplest)

1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```

2. Build the Flutter web application:
   ```bash
   flutter build web --release
   ```

3. Deploy the compiled static output directory `build/web`:
   ```bash
   # If you have vercel installed:
   vercel deploy build/web --prod
   ```

---

## Method 2: Git-Integrated Deployment via GitHub Actions (Recommended for CI/CD)

Because standard Vercel serverless build images do not come with the Flutter SDK pre-installed in Linux containers, the industry standard is to use a simple GitHub Action to build and deploy to Vercel automatically on push:

Create `.github/workflows/deploy-frontend.yml`:

```yaml
name: Deploy Flutter Web to Vercel

on:
  push:
    branches: [ main ]
    paths:
      - 'frontend/**'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Java
        uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '17'

      - name: Set up Flutter
        uses: subosito/flutter-action@v2
        with:
          channel: 'stable'
          cache: true

      - name: Install dependencies
        run: |
          cd frontend
          flutter pub get

      - name: Build Flutter Web
        run: |
          cd frontend
          flutter build web --release

      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: frontend/build/web
          vercel-args: '--prod'
```

---

## Method 3: Direct Git Repo Import in Vercel with Custom Install Script

If you connect the repo directly on [vercel.com](https://vercel.com):

1. **Root Directory**: Set to `frontend`
2. **Framework Preset**: Select `Other`
3. **Build Command**:
   ```bash
   if [ ! -d "$HOME/flutter" ]; then git clone https://github.com/flutter/flutter.git -b stable $HOME/flutter; fi && export PATH="$PATH:$HOME/flutter/bin" && flutter build web --release
   ```
4. **Output Directory**: `build/web`

---

## SPA Routing & Headers Configuration

A [`vercel.json`](file:///c:/Users/nahar/Desktop/Vaani-Shield/frontend/vercel.json) has already been placed in `frontend/`:

- Directs all routes (`/(.*)`) to `/index.html` so Flutter navigation works on page refresh.
- Includes `Cross-Origin-Embedder-Policy` and `Cross-Origin-Opener-Policy` headers to support WebAssembly (WASM) and multi-threaded audio features.
