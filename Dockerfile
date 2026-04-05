# ──────────────────────────────────────────────────────────────
# Dockerfile — Smart Energy RL v2.0  (HF Spaces / local)
# Exposes port 7860. Run: docker build -t smart-energy . && docker run -p 7860:7860 smart-energy
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN chown -R user:user /home/user/app

# Install Python deps first (layer cache)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=user:user . .

# Remove git history and pycache from image
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    rm -rf .git

USER user
ENV PYTHONPATH=/home/user/app
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Default: run the web API. Override with: docker run ... python inference.py
CMD ["python", "app.py"]
