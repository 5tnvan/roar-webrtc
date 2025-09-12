# Base Pipecat image
FROM dailyco/pipecat-base:latest

# Copy Python dependencies
COPY ./requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your bot code
COPY ./bot-gemini.py bot-gemini.py
