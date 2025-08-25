const express = require('express');
const path = require('path');
const { createServer } = require('http');

const app = express();
const httpServer = createServer(app);

// Middleware
app.use(express.json());
app.use(express.static('client'));

// Simple API routes for the Mini App
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Mini App server running!' });
});

// Mock data for development
app.get('/api/user/:telegramId', (req, res) => {
  const { telegramId } = req.params;
  
  res.json({
    id: 1,
    telegramId: telegramId,
    username: 'TestUser',
    firstName: 'Predictor',
    totalPredictions: 25,
    correctPredictions: 18,
    currentStreak: 3,
    bestStreak: 7,
    confidencePoints: 1250,
    rank: 'Skilled Predictor',
    accuracy: 72.0,
    badges: [
      { badgeName: '🎯 First Prediction', earnedAt: '2024-01-01' },
      { badgeName: '🔥 Hot Streak', earnedAt: '2024-01-15' }
    ],
    predictions: [
      {
        homeTeam: 'Arsenal',
        awayTeam: 'Chelsea',
        prediction: 'Arsenal Win',
        confidence: 75.2,
        marketBacked: true,
        createdAt: '2024-01-20T10:00:00Z'
      }
    ]
  });
});

app.get('/api/leaderboard', (req, res) => {
  res.json([
    {
      position: 1,
      telegramId: '123456789',
      displayName: 'Champion Predictor',
      totalPredictions: 45,
      correctPredictions: 36,
      accuracy: 80.0,
      currentStreak: 8,
      confidencePoints: 2100,
      rank: 'Expert Predictor'
    },
    {
      position: 2,
      telegramId: '987654321',
      displayName: 'Pro Analyst',
      totalPredictions: 38,
      correctPredictions: 28,
      accuracy: 73.7,
      currentStreak: 4,
      confidencePoints: 1850,
      rank: 'Advanced Analyst'
    }
  ]);
});

app.get('/api/community', (req, res) => {
  res.json({
    totalUsers: 156,
    activeUsers: 89,
    totalPredictions: 2340,
    communityAccuracy: 67.8
  });
});

app.get('/api/feed', (req, res) => {
  res.json([
    {
      id: 1,
      homeTeam: 'Manchester City',
      awayTeam: 'Liverpool',
      league: 'Premier League',
      prediction: 'Manchester City Win',
      confidence: 68.5,
      marketBacked: true,
      createdAt: '2024-01-20T14:30:00Z',
      user: {
        displayName: 'Football Expert',
        rank: 'Expert Predictor'
      }
    }
  ]);
});

// Serve the Mini App
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/index.html'));
});

const PORT = process.env.PORT || 5000;

httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Mini App server running on port ${PORT}`);
  console.log(`📱 Community Hub accessible at http://localhost:${PORT}`);
});

module.exports = app;