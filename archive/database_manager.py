"""
Database manager for authentic user data in community features
"""
import os
import asyncio
import asyncpg
from typing import Dict, List, Optional
from datetime import datetime

class DatabaseManager:
    """Manages authentic user data from PostgreSQL database"""
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)
            await self.create_tables_if_not_exist()
        except Exception as e:
            print(f"Database initialization error: {e}")
            self.pool = None
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def create_tables_if_not_exist(self):
        """Create tables if they don't exist"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            # Create users table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    telegram_id TEXT UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    joined_date TIMESTAMP DEFAULT NOW(),
                    last_active TIMESTAMP DEFAULT NOW(),
                    confidence_points INTEGER DEFAULT 1000,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    current_streak INTEGER DEFAULT 0,
                    best_streak INTEGER DEFAULT 0,
                    rank TEXT DEFAULT 'Beginner'
                )
            ''')
            
            # Create predictions table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT,
                    match_date TIMESTAMP,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    market_backed BOOLEAN DEFAULT FALSE,
                    actual_result TEXT,
                    points_staked INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
    
    async def get_or_create_user(self, telegram_id: str, username: str = None, first_name: str = None) -> Optional[Dict]:
        """Get or create user with authentic data only"""
        if not self.pool:
            return None
            
        async with self.pool.acquire() as conn:
            # Try to get existing user
            user = await conn.fetchrow(
                'SELECT * FROM users WHERE telegram_id = $1', 
                str(telegram_id)
            )
            
            if user:
                # Update last active
                await conn.execute(
                    'UPDATE users SET last_active = NOW() WHERE telegram_id = $1',
                    str(telegram_id)
                )
                return dict(user)
            else:
                # Create new user with default values
                user_id = await conn.fetchval('''
                    INSERT INTO users (telegram_id, username, first_name, joined_date, last_active, 
                                     confidence_points, total_predictions, correct_predictions, 
                                     current_streak, best_streak, rank)
                    VALUES ($1, $2, $3, NOW(), NOW(), 1000, 0, 0, 0, 0, 'Beginner')
                    RETURNING id
                ''', str(telegram_id), username, first_name)
                
                # Get the newly created user
                new_user = await conn.fetchrow('SELECT * FROM users WHERE id = $1', user_id)
                return dict(new_user)
    
    async def get_user_dashboard(self, telegram_id: str) -> Optional[Dict]:
        """Get authentic user dashboard data"""
        if not self.pool:
            return None
            
        user = await self.get_or_create_user(telegram_id)
        if not user:
            return None
        
        async with self.pool.acquire() as conn:
            # Get user's recent predictions
            recent_predictions = await conn.fetch('''
                SELECT home_team, away_team, prediction, confidence, actual_result, created_at
                FROM predictions 
                WHERE user_id = $1 
                ORDER BY created_at DESC 
                LIMIT 3
            ''', user['id'])
            
            # Calculate accuracy
            accuracy = 0.0
            if user['total_predictions'] > 0:
                accuracy = (user['correct_predictions'] / user['total_predictions']) * 100
            
            return {
                'user': user,
                'accuracy': round(accuracy, 1),
                'recent_predictions': [dict(pred) for pred in recent_predictions]
            }
    
    async def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get authentic leaderboard data"""
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            # Get top users by accuracy (minimum 5 predictions)
            users = await conn.fetch('''
                SELECT 
                    telegram_id,
                    COALESCE(username, first_name, 'Anonymous') as display_name,
                    total_predictions,
                    correct_predictions,
                    CASE 
                        WHEN total_predictions > 0 
                        THEN ROUND((correct_predictions::FLOAT / total_predictions::FLOAT) * 100, 1)
                        ELSE 0 
                    END as accuracy,
                    current_streak,
                    confidence_points,
                    rank
                FROM users 
                WHERE total_predictions >= 3
                ORDER BY 
                    (correct_predictions::FLOAT / GREATEST(total_predictions::FLOAT, 1)) DESC,
                    total_predictions DESC
                LIMIT $1
            ''', limit)
            
            return [dict(user) for user in users]
    
    async def get_community_stats(self) -> Dict:
        """Get authentic community statistics"""
        if not self.pool:
            return {
                'total_users': 0,
                'active_users': 0,
                'total_predictions': 0,
                'community_accuracy': 0.0
            }
        
        async with self.pool.acquire() as conn:
            # Get community stats
            stats = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN total_predictions > 0 THEN 1 END) as active_users,
                    COALESCE(SUM(total_predictions), 0) as total_predictions,
                    CASE 
                        WHEN SUM(total_predictions) > 0 
                        THEN ROUND((SUM(correct_predictions)::FLOAT / SUM(total_predictions)::FLOAT) * 100, 1)
                        ELSE 0 
                    END as community_accuracy
                FROM users
            ''')
            
            return dict(stats) if stats else {
                'total_users': 0,
                'active_users': 0, 
                'total_predictions': 0,
                'community_accuracy': 0.0
            }
    
    async def record_prediction(self, telegram_id: str, home_team: str, away_team: str, 
                              prediction: str, confidence: float, league: str = None) -> bool:
        """Record a new prediction"""
        if not self.pool:
            return False
            
        user = await self.get_or_create_user(telegram_id)
        if not user:
            return False
        
        async with self.pool.acquire() as conn:
            try:
                # Insert prediction
                await conn.execute('''
                    INSERT INTO predictions (user_id, home_team, away_team, prediction, 
                                           confidence, league, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ''', user['id'], home_team, away_team, prediction, confidence, league)
                
                # Update user's total predictions
                await conn.execute('''
                    UPDATE users 
                    SET total_predictions = total_predictions + 1,
                        last_active = NOW()
                    WHERE id = $1
                ''', user['id'])
                
                return True
            except Exception as e:
                print(f"Error recording prediction: {e}")
                return False