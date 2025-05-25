# Deploy Sports Prediction Bot to Render.com

## Quick Deploy Steps

1. **Connect GitHub Repository**
   - Go to [Render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository: `ryanalmb/AI-sports-outcome-predictor`

2. **Configure Service Settings**
   ```
   Name: sports-prediction-bot
   Environment: Python 3
   Build Command: pip install uv && uv sync --frozen
   Start Command: python fixed_bot.py
   ```

3. **Set Environment Variables**
   Add these required variables in Render dashboard:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   FOOTBALL_API_KEY=your_football_api_key
   ODDS_API_KEY=your_odds_api_key
   ```

4. **Database Setup (Optional)**
   - Create PostgreSQL database in Render
   - Add DATABASE_URL environment variable
   - Your bot will automatically create tables

## Features Ready for Deployment

✅ **Enhanced 27-Feature Prediction System**
- Professional-grade sports analysis
- Multiple ML frameworks (XGBoost, LightGBM, TensorFlow, PyTorch)
- Authentic match data from comprehensive dataset

✅ **Production Ready**
- Flexible PORT binding for Render.com
- Health check endpoints
- Error handling and logging
- Docker deployment support

✅ **Complete Bot Functionality**
- Real-time match predictions
- Live betting odds integration
- Community features and leaderboards
- Professional prediction accuracy

## Your Bot Capabilities

🎯 **Prediction Tiers Available:**
- Standard predictions (`/predict`)
- Enhanced analysis (`/analysis`) 
- Professional predictions (`/advanced`)
- Neural network predictions (`/deepml`)
- Ultimate God Ensemble predictions

🏆 **League Coverage:**
Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League, MLS, Liga MX, Eredivisie

Your sophisticated prediction platform is ready to deliver intelligent sports analysis using authentic data!