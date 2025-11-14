# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip and install dependencies in one layer
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port (Vercel sets $PORT dynamically)
EXPOSE 8501

# Use environment variable PORT if available (Vercel sets $PORT)
ENV PORT 8501

# Run Streamlit in headless mode
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.enableCORS=false --server.headless=true"]
