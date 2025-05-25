"""
Social Features for Sports Prediction Bot
=========================================

User tracking, leaderboards, and community features with authentic data only
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SocialFeaturesManager:
    """Manages social features, user tracking, and leaderboards"""
    
    def __init__(self):
        self.data_file = "user_data.json"
        self.users_data = self._load_user_data()
        
    def _load_user_data(self) -> Dict:
        """Load user data from persistent storage"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
        
        return {
            "users": {},
            "global_stats": {
                "total_predictions": 0,
                "total_users": 0,
                "community_accuracy": 0.0
            }
        }
    
    def _save_user_data(self):
        """Save user data to persistent storage"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.users_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
    
    def register_user(self, user_id: str, username: str = None, first_name: str = None):
        """Register a new user or update existing user info"""
        user_id = str(user_id)
        
        if user_id not in self.users_data["users"]:
            self.users_data["users"][user_id] = {
                "username": username,
                "first_name": first_name,
                "joined_date": datetime.now().isoformat(),
                "predictions": [],
                "stats": {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy": 0.0,
                    "current_streak": 0,
                    "best_streak": 0,
                    "confidence_points": 1000,  # Starting points
                    "rank": "Beginner"
                },
                "badges": [],
                "last_active": datetime.now().isoformat()
            }
            self.users_data["global_stats"]["total_users"] += 1
            self._save_user_data()
            logger.info(f"New user registered: {username or first_name} (ID: {user_id})")
        else:
            # Update user info
            self.users_data["users"][user_id]["username"] = username
            self.users_data["users"][user_id]["first_name"] = first_name
            self.users_data["users"][user_id]["last_active"] = datetime.now().isoformat()
            self._save_user_data()
    
    def record_prediction(self, user_id: str, match_info: Dict, prediction: Dict, confidence: float):
        """Record a user's prediction for tracking"""
        user_id = str(user_id)
        
        if user_id not in self.users_data["users"]:
            return False
        
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "match": {
                "home_team": match_info.get("home_team"),
                "away_team": match_info.get("away_team"),
                "date": match_info.get("date"),
                "league": match_info.get("league")
            },
            "prediction": prediction.get("prediction"),
            "confidence": confidence,
            "market_backed": prediction.get("market_data", False),
            "result": None,  # To be updated when match finishes
            "points_staked": min(100, int(confidence))  # Stake points based on confidence
        }
        
        self.users_data["users"][user_id]["predictions"].append(prediction_record)
        self.users_data["users"][user_id]["stats"]["total_predictions"] += 1
        self.users_data["global_stats"]["total_predictions"] += 1
        
        self._save_user_data()
        return True
    
    def update_prediction_result(self, user_id: str, prediction_index: int, actual_result: str):
        """Update a prediction with the actual match result"""
        user_id = str(user_id)
        
        if user_id not in self.users_data["users"]:
            return False
        
        user = self.users_data["users"][user_id]
        if prediction_index >= len(user["predictions"]):
            return False
        
        prediction = user["predictions"][prediction_index]
        prediction["result"] = actual_result
        
        # Check if prediction was correct
        was_correct = prediction["prediction"] == actual_result
        
        if was_correct:
            user["stats"]["correct_predictions"] += 1
            user["stats"]["current_streak"] += 1
            user["stats"]["best_streak"] = max(user["stats"]["best_streak"], user["stats"]["current_streak"])
            
            # Award confidence points
            points_won = prediction["points_staked"]
            user["stats"]["confidence_points"] += points_won
        else:
            user["stats"]["current_streak"] = 0
            # Lose staked points
            user["stats"]["confidence_points"] -= prediction["points_staked"]
        
        # Ensure points don't go below 0
        user["stats"]["confidence_points"] = max(0, user["stats"]["confidence_points"])
        
        # Update accuracy
        user["stats"]["accuracy"] = (user["stats"]["correct_predictions"] / user["stats"]["total_predictions"]) * 100
        
        # Update rank
        self._update_user_rank(user_id)
        
        # Check for new badges
        self._check_badges(user_id)
        
        self._save_user_data()
        return True
    
    def _update_user_rank(self, user_id: str):
        """Update user rank based on accuracy and predictions"""
        user = self.users_data["users"][user_id]
        accuracy = user["stats"]["accuracy"]
        total_predictions = user["stats"]["total_predictions"]
        
        if total_predictions < 5:
            rank = "Beginner"
        elif accuracy >= 80 and total_predictions >= 20:
            rank = "Expert Predictor"
        elif accuracy >= 70 and total_predictions >= 15:
            rank = "Advanced Analyst"
        elif accuracy >= 60 and total_predictions >= 10:
            rank = "Skilled Predictor"
        elif total_predictions >= 5:
            rank = "Learning Predictor"
        else:
            rank = "Beginner"
        
        user["stats"]["rank"] = rank
    
    def _check_badges(self, user_id: str):
        """Check and award badges to user"""
        user = self.users_data["users"][user_id]
        current_badges = set(user["badges"])
        
        # Define badge criteria
        badge_criteria = {
            "🎯 First Prediction": user["stats"]["total_predictions"] >= 1,
            "🔥 Hot Streak": user["stats"]["current_streak"] >= 5,
            "⚡ Lightning Streak": user["stats"]["best_streak"] >= 10,
            "🎖️ Century Club": user["stats"]["total_predictions"] >= 100,
            "🏆 Accuracy Master": user["stats"]["accuracy"] >= 75 and user["stats"]["total_predictions"] >= 10,
            "💎 Diamond Predictor": user["stats"]["accuracy"] >= 80 and user["stats"]["total_predictions"] >= 20,
            "🎰 High Roller": user["stats"]["confidence_points"] >= 2000,
        }
        
        new_badges = []
        for badge, earned in badge_criteria.items():
            if earned and badge not in current_badges:
                user["badges"].append(badge)
                new_badges.append(badge)
        
        return new_badges
    
    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """Get user statistics"""
        user_id = str(user_id)
        if user_id not in self.users_data["users"]:
            return None
        
        user = self.users_data["users"][user_id]
        return {
            "username": user.get("username") or user.get("first_name", "Unknown"),
            "stats": user["stats"],
            "badges": user["badges"],
            "recent_predictions": user["predictions"][-5:] if user["predictions"] else []
        }
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top users leaderboard"""
        users_with_stats = []
        
        for user_id, user_data in self.users_data["users"].items():
            if user_data["stats"]["total_predictions"] >= 3:  # Minimum predictions to qualify
                users_with_stats.append({
                    "user_id": user_id,
                    "username": user_data.get("username") or user_data.get("first_name", "Unknown"),
                    "accuracy": user_data["stats"]["accuracy"],
                    "total_predictions": user_data["stats"]["total_predictions"],
                    "current_streak": user_data["stats"]["current_streak"],
                    "confidence_points": user_data["stats"]["confidence_points"],
                    "rank": user_data["stats"]["rank"]
                })
        
        # Sort by accuracy (with minimum predictions), then by total predictions
        users_with_stats.sort(key=lambda x: (-x["accuracy"], -x["total_predictions"]))
        
        return users_with_stats[:limit]
    
    def get_community_insights(self) -> Dict:
        """Get community-wide statistics and insights"""
        total_users = len(self.users_data["users"])
        active_users = sum(1 for user in self.users_data["users"].values() 
                          if user["stats"]["total_predictions"] > 0)
        
        # Calculate community accuracy
        total_predictions = sum(user["stats"]["total_predictions"] for user in self.users_data["users"].values())
        total_correct = sum(user["stats"]["correct_predictions"] for user in self.users_data["users"].values())
        
        community_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_predictions": total_predictions,
            "community_accuracy": round(community_accuracy, 1),
            "top_accuracy": max((user["stats"]["accuracy"] for user in self.users_data["users"].values() 
                               if user["stats"]["total_predictions"] >= 5), default=0)
        }
    
    def get_user_rank_position(self, user_id: str) -> Optional[int]:
        """Get user's position in the leaderboard"""
        leaderboard = self.get_leaderboard(limit=1000)  # Get full leaderboard
        
        for position, user in enumerate(leaderboard, 1):
            if user["user_id"] == str(user_id):
                return position
        
        return None