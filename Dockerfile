
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application entry point
COPY agent.py ./

# Copy packages
COPY core/ ./core/
COPY bot/ ./bot/
COPY tools/ ./tools/
COPY configs/ ./configs/

# Create directory for vector database persistence
RUN mkdir -p /app/code_vector_db

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Healthcheck: verify Python + critical imports load
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from agent import run_agent_with_session; print('ok')"

# Entry point: Slack bot (socket mode)
CMD ["python", "bot/slack_bot.py"]
