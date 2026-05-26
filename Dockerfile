FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/faq_seed.csv data/faq_seed.csv
COPY scripts/ scripts/
COPY wiki/ wiki/

RUN python scripts/init_all.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.bot.main"]
