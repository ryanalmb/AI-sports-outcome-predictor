const express = require('express');
const path = require('path');
const app = express();

// Enable CORS for Telegram
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

app.use(express.json());
app.use(express.static('client'));

// API endpoints
app.get('/api/stats', (req, res) => {
  res.json({
    totalUsers: 156,
    activePredictors: 89,
    todaysPredictions: 23,
    communityAccuracy: 67.8
  });
});

app.get('/api/leaderboard', (req, res) => {
  res.json([
    { rank: 1, name: "PredictionMaster", accuracy: 78.5, predictions: 45 },
    { rank: 2, name: "SoccerGuru", accuracy: 76.2, predictions: 52 },
    { rank: 3, name: "OddsWhiz", accuracy: 74.8, predictions: 38 },
    { rank: 4, name: "MatchPredictor", accuracy: 73.1, predictions: 41 },
    { rank: 5, name: "SportsSage", accuracy: 71.9, predictions: 33 }
  ]);
});

// Serve the Mini App
app.get('/', (req, res) => {
  res.send(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sports Prediction Community</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--tg-theme-bg-color, #ffffff);
            color: var(--tg-theme-text-color, #000000);
            padding: 20px;
            min-height: 100vh;
        }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #2481cc; font-size: 24px; margin-bottom: 10px; }
        .header p { color: #666; font-size: 14px; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 30px; }
        .stat-card { 
            background: var(--tg-theme-button-color, #2481cc); 
            color: white; 
            padding: 20px; 
            border-radius: 12px; 
            text-align: center;
        }
        .stat-number { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 12px; opacity: 0.8; }
        .section { margin-bottom: 30px; }
        .section h2 { font-size: 18px; margin-bottom: 15px; color: #2481cc; }
        .leaderboard-item { 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            padding: 15px; 
            background: var(--tg-theme-secondary-bg-color, #f8f9fa); 
            border-radius: 8px; 
            margin-bottom: 10px;
        }
        .rank { 
            background: #2481cc; 
            color: white; 
            width: 30px; 
            height: 30px; 
            border-radius: 50%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-weight: bold; 
            font-size: 14px;
        }
        .player-info { flex: 1; margin-left: 15px; }
        .player-name { font-weight: bold; font-size: 16px; }
        .player-stats { font-size: 12px; color: #666; margin-top: 2px; }
        .accuracy { font-weight: bold; color: #28a745; font-size: 16px; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .refresh-btn {
            background: #2481cc;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 Sports Prediction Community</h1>
        <p>Connect with fellow predictors and track your success!</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number" id="totalUsers">---</div>
            <div class="stat-label">Total Users</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="activePredictors">---</div>
            <div class="stat-label">Active Predictors</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="todaysPredictions">---</div>
            <div class="stat-label">Today's Predictions</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="communityAccuracy">---%</div>
            <div class="stat-label">Community Accuracy</div>
        </div>
    </div>

    <div class="section">
        <h2>🥇 Top Predictors</h2>
        <div id="leaderboard">
            <div class="loading">Loading leaderboard...</div>
        </div>
    </div>

    <button class="refresh-btn" onclick="loadData()">🔄 Refresh Data</button>

    <script>
        // Initialize Telegram WebApp
        let tg = window.Telegram.WebApp;
        tg.ready();
        tg.expand();
        
        // Set theme colors
        document.body.style.backgroundColor = tg.themeParams.bg_color || '#ffffff';
        document.body.style.color = tg.themeParams.text_color || '#000000';

        // Load community data
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('totalUsers').textContent = stats.totalUsers;
                document.getElementById('activePredictors').textContent = stats.activePredictors;
                document.getElementById('todaysPredictions').textContent = stats.todaysPredictions;
                document.getElementById('communityAccuracy').textContent = stats.communityAccuracy + '%';
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        async function loadLeaderboard() {
            try {
                const response = await fetch('/api/leaderboard');
                const leaderboard = await response.json();
                
                const leaderboardHTML = leaderboard.map(player => \`
                    <div class="leaderboard-item">
                        <div class="rank">\${player.rank}</div>
                        <div class="player-info">
                            <div class="player-name">\${player.name}</div>
                            <div class="player-stats">\${player.predictions} predictions</div>
                        </div>
                        <div class="accuracy">\${player.accuracy}%</div>
                    </div>
                \`).join('');
                
                document.getElementById('leaderboard').innerHTML = leaderboardHTML;
            } catch (error) {
                console.error('Error loading leaderboard:', error);
                document.getElementById('leaderboard').innerHTML = '<div class="loading">Error loading leaderboard</div>';
            }
        }

        function loadData() {
            loadStats();
            loadLeaderboard();
        }

        // Load data when page loads
        loadData();
        
        // Optional: Haptic feedback for interactions
        document.querySelector('.refresh-btn').addEventListener('click', () => {
            if (tg.HapticFeedback) {
                tg.HapticFeedback.impactOccurred('medium');
            }
        });
    </script>
</body>
</html>
  `);
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => {
  console.log('🚀 Simple Mini App running on port ' + PORT);
});