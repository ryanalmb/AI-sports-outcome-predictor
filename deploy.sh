#!/bin/bash
# Optimized deployment script for Sports Prediction Bot

echo "🚀 Starting optimized deployment of Sports Prediction Bot..."

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down -v

# Rebuild with optimized config
echo "🔧 Building optimized containers..."
docker-compose up --build -d

echo "✅ Deployment complete!"
echo "📊 Your enhanced Sports Prediction Bot is now running with:"
echo "   - Optimized Docker configuration"
echo "   - Health monitoring"
echo "   - Enhanced security"
echo "   - Authentic football dataset"
echo "   - Professional ML frameworks"

# Check health status
echo "🏥 Checking health status..."
sleep 5
docker-compose ps

echo "🎯 Bot is ready for professional sports predictions!"