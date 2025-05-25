import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import WebSocket, { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { db } from './db';
import { users, predictions, badges } from '../shared/schema';
import { eq, desc, sql } from 'drizzle-orm';

const app = express();
const httpServer = createServer(app);

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors());
app.use(express.json());

// WebSocket server for real-time updates
const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

wss.on('connection', (ws: WebSocket) => {
  console.log('Client connected to Mini App');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message.toString());
      console.log('Received:', data);
      
      // Handle real-time requests
      if (data.type === 'subscribe_leaderboard') {
        // Send initial leaderboard data
        broadcastLeaderboard();
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });
  
  ws.on('close', () => {
    console.log('Client disconnected from Mini App');
  });
});

// Broadcast leaderboard updates to all connected clients
async function broadcastLeaderboard() {
  try {
    const leaderboard = await getLeaderboardData();
    const message = JSON.stringify({
      type: 'leaderboard_update',
      data: leaderboard
    });
    
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  } catch (error) {
    console.error('Error broadcasting leaderboard:', error);
  }
}

// API Routes for Mini App

// Get user profile and stats
app.get('/api/user/:telegramId', async (req, res) => {
  try {
    const { telegramId } = req.params;
    
    const user = await db.query.users.findFirst({
      where: eq(users.telegramId, telegramId),
      with: {
        badges: true,
        predictions: {
          orderBy: desc(predictions.createdAt),
          limit: 10
        }
      }
    });
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    const accuracy = user.totalPredictions > 0 
      ? (user.correctPredictions / user.totalPredictions) * 100 
      : 0;
    
    res.json({
      ...user,
      accuracy: Math.round(accuracy * 10) / 10
    });
  } catch (error) {
    console.error('Error fetching user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get leaderboard
app.get('/api/leaderboard', async (req, res) => {
  try {
    const leaderboard = await getLeaderboardData();
    res.json(leaderboard);
  } catch (error) {
    console.error('Error fetching leaderboard:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

async function getLeaderboardData() {
  const topUsers = await db
    .select({
      id: users.id,
      telegramId: users.telegramId,
      username: users.username,
      firstName: users.firstName,
      totalPredictions: users.totalPredictions,
      correctPredictions: users.correctPredictions,
      currentStreak: users.currentStreak,
      bestStreak: users.bestStreak,
      confidencePoints: users.confidencePoints,
      rank: users.rank,
      accuracy: sql<number>`CASE WHEN ${users.totalPredictions} > 0 THEN (${users.correctPredictions}::float / ${users.totalPredictions}::float) * 100 ELSE 0 END`
    })
    .from(users)
    .where(sql`${users.totalPredictions} >= 3`)
    .orderBy(desc(sql`CASE WHEN ${users.totalPredictions} > 0 THEN (${users.correctPredictions}::float / ${users.totalPredictions}::float) * 100 ELSE 0 END`), desc(users.totalPredictions))
    .limit(50);
  
  return topUsers.map((user, index) => ({
    position: index + 1,
    ...user,
    displayName: user.username || user.firstName || 'Unknown User'
  }));
}

// Get community insights
app.get('/api/community', async (req, res) => {
  try {
    const stats = await db
      .select({
        totalUsers: sql<number>`count(*)`,
        activeUsers: sql<number>`count(*) filter (where ${users.totalPredictions} > 0)`,
        totalPredictions: sql<number>`sum(${users.totalPredictions})`,
        totalCorrect: sql<number>`sum(${users.correctPredictions})`
      })
      .from(users);
    
    const communityStats = stats[0];
    const communityAccuracy = communityStats.totalPredictions > 0 
      ? (communityStats.totalCorrect / communityStats.totalPredictions) * 100 
      : 0;
    
    res.json({
      totalUsers: communityStats.totalUsers,
      activeUsers: communityStats.activeUsers,
      totalPredictions: communityStats.totalPredictions,
      communityAccuracy: Math.round(communityAccuracy * 10) / 10
    });
  } catch (error) {
    console.error('Error fetching community stats:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get recent predictions for community feed
app.get('/api/feed', async (req, res) => {
  try {
    const recentPredictions = await db.query.predictions.findMany({
      with: {
        user: true
      },
      orderBy: desc(predictions.createdAt),
      limit: 20
    });
    
    const feed = recentPredictions.map(prediction => ({
      id: prediction.id,
      homeTeam: prediction.homeTeam,
      awayTeam: prediction.awayTeam,
      league: prediction.league,
      prediction: prediction.prediction,
      confidence: prediction.confidence,
      marketBacked: prediction.marketBacked,
      createdAt: prediction.createdAt,
      user: {
        displayName: prediction.user?.username || prediction.user?.firstName || 'Unknown',
        rank: prediction.user?.rank
      }
    }));
    
    res.json(feed);
  } catch (error) {
    console.error('Error fetching feed:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Create or update user
app.post('/api/user', async (req, res) => {
  try {
    const { telegramId, username, firstName } = req.body;
    
    const existingUser = await db.query.users.findFirst({
      where: eq(users.telegramId, telegramId)
    });
    
    if (existingUser) {
      // Update existing user
      const updatedUser = await db
        .update(users)
        .set({
          username,
          firstName,
          lastActive: new Date()
        })
        .where(eq(users.telegramId, telegramId))
        .returning();
      
      res.json(updatedUser[0]);
    } else {
      // Create new user
      const newUser = await db
        .insert(users)
        .values({
          telegramId,
          username,
          firstName
        })
        .returning();
      
      res.json(newUser[0]);
    }
  } catch (error) {
    console.error('Error creating/updating user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Serve static files for the Mini App
app.use(express.static('client'));

// Handle client-side routing for the Mini App
app.get('*', (req, res) => {
  res.sendFile('index.html', { root: 'client' });
});

const PORT = process.env.PORT || 5000;

httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Mini App server running on port ${PORT}`);
  console.log(`📱 WebSocket server ready for real-time updates`);
});

export default app;