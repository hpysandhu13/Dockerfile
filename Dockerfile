FROM python:3.10-slim

# Step 1: Install system-level dependencies for Pandas and Postgres
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 2: Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Copy your Pro Bot code
COPY . .

# Step 4: Run the bot
CMD ["python", "main.py"]
