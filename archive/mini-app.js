const express = require('express');
const path = require('path');

const app = express();

// Middleware
app.use(express.json());
app.use(express.static('client'));

// API endpoints for the Mini App
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Community Hub is running!' });
});

// User profile data
app.get('/api/user/:telegramId', (req, res) => {
  const { telegramId } = req.params;
  
  res.json({
    id: 1,
    telegramId: telegramId,
    username: 'CommunityMember',
    firstName: 'Predictor',
    totalPredictions: 15,
    correctPredictions: 11,
    currentStreak: 2,
    bestStreak: 5,
    confidencePoints: 1150,
    rank: 'Skilled Predictor',
    accuracy: 73.3,
    badges: [
      { badgeName: '🎯 First Prediction', earnedAt: '2024-01-01' },
      { badgeName: '🔥 Hot Streak', earnedAt: '2024-01-10' }
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

// Leaderboard data
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
    },
    {
      position: 3,
      telegramId: '456789123',
      displayName: 'Market Expert',
      totalPredictions: 32,
      correctPredictions: 22,
      accuracy: 68.8,
      currentStreak: 3,
      confidencePoints: 1420,
      rank: 'Skilled Predictor'
    }
  ]);
});

// Community stats
app.get('/api/community', (req, res) => {
  res.json({
    totalUsers: 156,
    activeUsers: 89,
    totalPredictions: 2340,
    communityAccuracy: 67.8
  });
});

// Community feed
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
    },
    {
      id: 2,
      homeTeam: 'Barcelona',
      awayTeam: 'Real Madrid',
      league: 'La Liga',
      prediction: 'Draw',
      confidence: 45.2,
      marketBacked: true,
      createdAt: '2024-01-20T13:15:00Z',
      user: {
        displayName: 'La Liga Specialist',
        rank: 'Advanced Analyst'
      }
    }
  ]);
});

// Serve the Mini App
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'client/index.html'));
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Community Hub running on port ${PORT}`);
  console.log(`📱 Access your Mini App at: http://localhost:${PORT}`);
});

module.exports = app;