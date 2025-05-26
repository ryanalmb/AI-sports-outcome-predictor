#!/bin/bash
# Quick deployment commands for Sports Prediction Bot

echo "🚀 Quick Deploy - Sports Prediction Bot"

# Stop existing containers
docker-compose down -v

# Rebuild with optimized config
docker-compose up --build

# Alternative: Single container deployment
# docker build -t sports-bot .
# docker run -p 8080:8080 -e TELEGRAM_BOT_TOKEN=your_token sports-bot

echo "✅ Sports Prediction Bot deployed successfully!"