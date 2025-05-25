"""
Enhanced Feature Engine for All Deep Learning Frameworks
Provides 27 sophisticated features using authentic 228K+ match dataset
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedFeatureEngine:
    """Unified 27-feature extraction engine for all ML frameworks"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_names = [
            # Core strength features (4)
            'home_elo', 'away_elo', 'elo_difference', 'elo_ratio',
            # Form features (8)
            'home_form3', 'away_form3', 'home_form5', 'away_form5',
            'home_ppg', 'away_ppg', 'home_momentum', 'away_momentum',
            # Goal features (6)
            'home_goals_pg', 'away_goals_pg', 'home_conceded_pg', 'away_conceded_pg',
            'home_goal_diff', 'away_goal_diff',
            # Performance features (4)
            'home_advantage', 'away_performance', 'home_win_rate', 'away_win_rate',
            # Head-to-head features (3)
            'h2h_home_wins', 'h2h_total', 'h2h_ratio',
            # Context features (2)
            'league_competitiveness', 'match_importance'
        ]
    
    def extract_features(self, home_team: str, away_team: str) -> np.ndarray:
        """Extract 27 features for a match"""
        try:
            # Find team matches
            home_mask = self.df['HomeTeam'].str.contains(home_team, na=False, case=False)
            away_mask = self.df['AwayTeam'].str.contains(away_team, na=False, case=False)
            
            home_matches = self.df[home_mask].copy()
            away_matches = self.df[away_mask].copy()
            
            if len(home_matches) == 0 or len(away_matches) == 0:
                return None
            
            features = []
            
            # Core strength (4 features)
            home_elo = self._safe_get(home_matches, 'HomeElo', 1500)
            away_elo = self._safe_get(away_matches, 'AwayElo', 1500)
            features.extend([
                home_elo, away_elo, 
                home_elo - away_elo,
                home_elo / max(away_elo, 1)
            ])
            
            # Form analysis (8 features)
            home_form3 = self._safe_get(home_matches, 'Form3Home', 0)
            away_form3 = self._safe_get(away_matches, 'Form3Away', 0)
            home_form5 = self._safe_get(home_matches, 'Form5Home', 0)
            away_form5 = self._safe_get(away_matches, 'Form5Away', 0)
            
            home_ppg = self._calc_points_per_game(home_matches, True)
            away_ppg = self._calc_points_per_game(away_matches, False)
            home_momentum = self._calc_momentum(home_matches, True)
            away_momentum = self._calc_momentum(away_matches, False)
            
            features.extend([home_form3, away_form3, home_form5, away_form5,
                           home_ppg, away_ppg, home_momentum, away_momentum])
            
            # Goal patterns (6 features)
            home_goals = self._calc_avg_goals(home_matches, 'FTHome')
            away_goals = self._calc_avg_goals(away_matches, 'FTAway')
            home_conceded = self._calc_avg_goals(home_matches, 'FTAway')
            away_conceded = self._calc_avg_goals(away_matches, 'FTHome')
            
            features.extend([
                home_goals, away_goals, home_conceded, away_conceded,
                home_goals - home_conceded, away_goals - away_conceded
            ])
            
            # Performance metrics (4 features)
            home_advantage = self._calc_home_advantage(home_matches)
            away_performance = self._calc_away_performance(away_matches)
            home_win_rate = self._calc_win_rate(home_matches, True)
            away_win_rate = self._calc_win_rate(away_matches, False)
            
            features.extend([home_advantage, away_performance, home_win_rate, away_win_rate])
            
            # Head-to-head (3 features)
            h2h_home, h2h_total, h2h_ratio = self._calc_h2h(home_team, away_team)
            features.extend([h2h_home, h2h_total, h2h_ratio])
            
            # Context (2 features)
            features.extend([self._calc_competitiveness(), 0.5])  # match_importance placeholder
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def prepare_training_data(self, sample_size: int = 15000) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with 27 features"""
        try:
            # Filter complete matches
            complete_matches = self.df[
                (self.df['FTHome'].notna()) & 
                (self.df['FTAway'].notna()) &
                (self.df['HomeTeam'].notna()) &
                (self.df['AwayTeam'].notna())
            ].copy()
            
            # Sample for training efficiency
            if len(complete_matches) > sample_size:
                step = len(complete_matches) // sample_size
                complete_matches = complete_matches.iloc[::step].head(sample_size)
            
            logger.info(f"Preparing 27-feature training data from {len(complete_matches)} matches")
            
            X_list = []
            y_list = []
            
            for idx, (_, match) in enumerate(complete_matches.iterrows()):
                if idx % 1000 == 0:
                    logger.info(f"Processing {idx}/{len(complete_matches)} matches")
                
                features = self.extract_features(match['HomeTeam'], match['AwayTeam'])
                
                if features is not None:
                    home_score = match['FTHome']
                    away_score = match['FTAway']
                    
                    if home_score > away_score:
                        result = 0  # Home win
                    elif away_score > home_score:
                        result = 1  # Away win
                    else:
                        result = 2  # Draw
                    
                    X_list.append(features)
                    y_list.append(result)
            
            if len(X_list) == 0:
                logger.error("No training samples generated")
                return np.array([]), np.array([])
            
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            
            logger.info(f"Generated {len(X)} samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def _safe_get(self, matches: pd.DataFrame, column: str, default: float) -> float:
        """Safely get latest value from column"""
        try:
            if len(matches) > 0 and column in matches.columns:
                latest = matches.iloc[-1][column]
                return float(latest) if pd.notna(latest) else default
        except:
            pass
        return default
    
    def _calc_points_per_game(self, matches: pd.DataFrame, is_home: bool) -> float:
        """Calculate points per game"""
        try:
            recent = matches.tail(10)
            points = 0
            games = 0
            
            for _, match in recent.iterrows():
                home_goals = match.get('FTHome', 0)
                away_goals = match.get('FTAway', 0)
                
                if pd.notna(home_goals) and pd.notna(away_goals):
                    if is_home:
                        if home_goals > away_goals:
                            points += 3
                        elif home_goals == away_goals:
                            points += 1
                    else:
                        if away_goals > home_goals:
                            points += 3
                        elif home_goals == away_goals:
                            points += 1
                    games += 1
            
            return points / max(games, 1)
        except:
            return 1.0
    
    def _calc_momentum(self, matches: pd.DataFrame, is_home: bool) -> float:
        """Calculate recent momentum"""
        try:
            recent = matches.tail(5)
            momentum = 0
            
            for i, (_, match) in enumerate(recent.iterrows()):
                home_goals = match.get('FTHome', 0)
                away_goals = match.get('FTAway', 0)
                
                if pd.notna(home_goals) and pd.notna(away_goals):
                    if is_home:
                        if home_goals > away_goals:
                            momentum += (i + 1) * 0.3
                    else:
                        if away_goals > home_goals:
                            momentum += (i + 1) * 0.3
            
            return momentum / 5.0
        except:
            return 0.0
    
    def _calc_avg_goals(self, matches: pd.DataFrame, column: str) -> float:
        """Calculate average goals"""
        try:
            recent = matches.tail(10)
            goals = recent[column].fillna(0)
            return float(goals.mean())
        except:
            return 1.5
    
    def _calc_home_advantage(self, matches: pd.DataFrame) -> float:
        """Calculate home advantage"""
        try:
            wins = 0
            total = 0
            for _, match in matches.tail(15).iterrows():
                home_goals = match.get('FTHome', 0)
                away_goals = match.get('FTAway', 0)
                if pd.notna(home_goals) and pd.notna(away_goals):
                    if home_goals > away_goals:
                        wins += 1
                    total += 1
            return wins / max(total, 1)
        except:
            return 0.5
    
    def _calc_away_performance(self, matches: pd.DataFrame) -> float:
        """Calculate away performance"""
        try:
            wins = 0
            total = 0
            for _, match in matches.tail(15).iterrows():
                home_goals = match.get('FTHome', 0)
                away_goals = match.get('FTAway', 0)
                if pd.notna(home_goals) and pd.notna(away_goals):
                    if away_goals > home_goals:
                        wins += 1
                    total += 1
            return wins / max(total, 1)
        except:
            return 0.3
    
    def _calc_win_rate(self, matches: pd.DataFrame, is_home: bool) -> float:
        """Calculate win rate"""
        try:
            wins = 0
            total = 0
            for _, match in matches.iterrows():
                home_goals = match.get('FTHome', 0)
                away_goals = match.get('FTAway', 0)
                if pd.notna(home_goals) and pd.notna(away_goals):
                    if is_home and home_goals > away_goals:
                        wins += 1
                    elif not is_home and away_goals > home_goals:
                        wins += 1
                    total += 1
            return wins / max(total, 1)
        except:
            return 0.33
    
    def _calc_h2h(self, home_team: str, away_team: str) -> Tuple[float, float, float]:
        """Calculate head-to-head"""
        try:
            h2h = self.df[
                ((self.df['HomeTeam'].str.contains(home_team, na=False, case=False)) & 
                 (self.df['AwayTeam'].str.contains(away_team, na=False, case=False))) |
                ((self.df['HomeTeam'].str.contains(away_team, na=False, case=False)) & 
                 (self.df['AwayTeam'].str.contains(home_team, na=False, case=False)))
            ]
            
            if len(h2h) == 0:
                return 0.5, 0.0, 1.0
            
            home_wins = 0
            for _, match in h2h.iterrows():
                home_goals = match.get('FTHome', 0)
                away_goals = match.get('FTAway', 0)
                if pd.notna(home_goals) and pd.notna(away_goals):
                    home_team_playing_home = home_team.lower() in str(match['HomeTeam']).lower()
                    if home_team_playing_home and home_goals > away_goals:
                        home_wins += 1
                    elif not home_team_playing_home and away_goals > home_goals:
                        home_wins += 1
            
            return home_wins / len(h2h), float(len(h2h)), 1.0
        except:
            return 0.5, 0.0, 1.0
    
    def _calc_competitiveness(self) -> float:
        """Calculate league competitiveness"""
        try:
            recent = self.df.tail(1000)
            goal_diff = recent['FTHome'].fillna(0) - recent['FTAway'].fillna(0)
            return min(float(np.var(goal_diff)) / 4.0, 1.0)
        except:
            return 0.5
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names.copy()