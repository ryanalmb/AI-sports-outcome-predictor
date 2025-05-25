"""
Enhanced Feature Extractor for Rich Dataset Analysis
Extracts maximum value from the 228K+ authentic match dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedFeatureExtractor:
    """Extract rich features from the comprehensive match dataset"""
    
    def __init__(self):
        self.team_stats_cache = {}
        self.league_characteristics = {}
        
    def extract_comprehensive_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract comprehensive features using all available authentic data"""
        try:
            features = {}
            
            # Core team strength from ELO ratings
            features.update(self._extract_elo_features(home_team, away_team, df))
            
            # Advanced form analysis using Form3/Form5 data
            features.update(self._extract_form_features(home_team, away_team, df))
            
            # Shot efficiency and attacking patterns
            features.update(self._extract_performance_features(home_team, away_team, df))
            
            # Disciplinary and tactical indicators
            features.update(self._extract_tactical_features(home_team, away_team, df))
            
            # Market intelligence from betting odds
            features.update(self._extract_market_features(home_team, away_team, df))
            
            # Head-to-head with enhanced context
            features.update(self._extract_h2h_features(home_team, away_team, df))
            
            # League-specific characteristics
            features.update(self._extract_league_features(home_team, away_team, df))
            
            # Temporal and seasonal patterns
            features.update(self._extract_temporal_features(home_team, away_team, df))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            return self._fallback_basic_features(home_team, away_team, df)
    
    def _extract_elo_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract ELO-based strength features"""
        home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)]
        away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)]
        
        features = {}
        
        if len(home_matches) > 0:
            home_elo = home_matches['HomeElo'].fillna(1500).mean() if home_team in home_matches['HomeTeam'].values else home_matches['AwayElo'].fillna(1500).mean()
            features['home_avg_elo'] = home_elo
            features['home_elo_trend'] = self._calculate_elo_trend(home_team, df)
        else:
            features['home_avg_elo'] = 1500
            features['home_elo_trend'] = 0
            
        if len(away_matches) > 0:
            away_elo = away_matches['HomeElo'].fillna(1500).mean() if away_team in away_matches['HomeTeam'].values else away_matches['AwayElo'].fillna(1500).mean()
            features['away_avg_elo'] = away_elo
            features['away_elo_trend'] = self._calculate_elo_trend(away_team, df)
        else:
            features['away_avg_elo'] = 1500
            features['away_elo_trend'] = 0
            
        features['elo_difference'] = features['home_avg_elo'] - features['away_avg_elo']
        
        return features
    
    def _extract_form_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract form-based features using Form3/Form5 data"""
        features = {}
        
        home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].tail(10)
        away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].tail(10)
        
        # Home team form analysis
        if len(home_matches) > 0:
            home_form3 = home_matches[home_matches['HomeTeam'] == home_team]['Form3Home'].fillna(0).mean()
            home_form5 = home_matches[home_matches['HomeTeam'] == home_team]['Form5Home'].fillna(0).mean()
            features['home_form3'] = home_form3
            features['home_form5'] = home_form5
            features['home_form_momentum'] = home_form3 - home_form5  # Recent vs longer form
        else:
            features['home_form3'] = 0
            features['home_form5'] = 0
            features['home_form_momentum'] = 0
            
        # Away team form analysis
        if len(away_matches) > 0:
            away_form3 = away_matches[away_matches['AwayTeam'] == away_team]['Form3Away'].fillna(0).mean()
            away_form5 = away_matches[away_matches['AwayTeam'] == away_team]['Form5Away'].fillna(0).mean()
            features['away_form3'] = away_form3
            features['away_form5'] = away_form5
            features['away_form_momentum'] = away_form3 - away_form5
        else:
            features['away_form3'] = 0
            features['away_form5'] = 0
            features['away_form_momentum'] = 0
            
        features['form_difference'] = features['home_form5'] - features['away_form5']
        
        return features
    
    def _extract_performance_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract performance metrics from shots, goals, etc."""
        features = {}
        
        # Home team performance
        home_home_matches = df[(df['HomeTeam'] == home_team) & (df['HomeShots'].notna())]
        home_away_matches = df[(df['AwayTeam'] == home_team) & (df['AwayShots'].notna())]
        
        if len(home_home_matches) > 0:
            features['home_shots_per_game'] = home_home_matches['HomeShots'].mean()
            features['home_shots_on_target_ratio'] = (home_home_matches['HomeTarget'] / home_home_matches['HomeShots'].replace(0, 1)).mean()
            features['home_goals_per_shot'] = (home_home_matches['FTHome'] / home_home_matches['HomeShots'].replace(0, 1)).mean()
        else:
            features['home_shots_per_game'] = 10
            features['home_shots_on_target_ratio'] = 0.3
            features['home_goals_per_shot'] = 0.1
            
        # Away team performance  
        away_away_matches = df[(df['AwayTeam'] == away_team) & (df['AwayShots'].notna())]
        away_home_matches = df[(df['HomeTeam'] == away_team) & (df['HomeShots'].notna())]
        
        if len(away_away_matches) > 0:
            features['away_shots_per_game'] = away_away_matches['AwayShots'].mean()
            features['away_shots_on_target_ratio'] = (away_away_matches['AwayTarget'] / away_away_matches['AwayShots'].replace(0, 1)).mean()
            features['away_goals_per_shot'] = (away_away_matches['FTAway'] / away_away_matches['AwayShots'].replace(0, 1)).mean()
        else:
            features['away_shots_per_game'] = 10
            features['away_shots_on_target_ratio'] = 0.3
            features['away_goals_per_shot'] = 0.1
            
        # Performance differentials
        features['shots_advantage'] = features['home_shots_per_game'] - features['away_shots_per_game']
        features['accuracy_advantage'] = features['home_shots_on_target_ratio'] - features['away_shots_on_target_ratio']
        
        return features
    
    def _extract_tactical_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract tactical indicators from fouls, corners, cards"""
        features = {}
        
        # Home team tactical profile
        home_matches = df[(df['HomeTeam'] == home_team) & (df['HomeFouls'].notna())]
        if len(home_matches) > 0:
            features['home_fouls_per_game'] = home_matches['HomeFouls'].mean()
            features['home_corners_per_game'] = home_matches['HomeCorners'].fillna(0).mean()
            features['home_yellow_cards'] = home_matches['HomeYellow'].fillna(0).mean()
            features['home_red_cards'] = home_matches['HomeRed'].fillna(0).mean()
            features['home_aggression_index'] = features['home_fouls_per_game'] + features['home_yellow_cards'] * 2
        else:
            features['home_fouls_per_game'] = 12
            features['home_corners_per_game'] = 5
            features['home_yellow_cards'] = 2
            features['home_red_cards'] = 0.1
            features['home_aggression_index'] = 16
            
        # Away team tactical profile
        away_matches = df[(df['AwayTeam'] == away_team) & (df['AwayFouls'].notna())]
        if len(away_matches) > 0:
            features['away_fouls_per_game'] = away_matches['AwayFouls'].mean()
            features['away_corners_per_game'] = away_matches['AwayCorners'].fillna(0).mean()
            features['away_yellow_cards'] = away_matches['AwayYellow'].fillna(0).mean()
            features['away_red_cards'] = away_matches['AwayRed'].fillna(0).mean()
            features['away_aggression_index'] = features['away_fouls_per_game'] + features['away_yellow_cards'] * 2
        else:
            features['away_fouls_per_game'] = 12
            features['away_corners_per_game'] = 5
            features['away_yellow_cards'] = 2
            features['away_red_cards'] = 0.1
            features['away_aggression_index'] = 16
            
        # Tactical matchup analysis
        features['aggression_clash'] = abs(features['home_aggression_index'] - features['away_aggression_index'])
        features['corner_advantage'] = features['home_corners_per_game'] - features['away_corners_per_game']
        
        return features
    
    def _extract_market_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract market intelligence from betting odds"""
        features = {}
        
        # Home team market perception
        home_matches = df[(df['HomeTeam'] == home_team) & (df['OddHome'].notna())]
        if len(home_matches) > 0:
            features['home_avg_odds'] = home_matches['OddHome'].mean()
            features['home_market_confidence'] = 1 / features['home_avg_odds'] if features['home_avg_odds'] > 0 else 0.5
            features['home_over25_tendency'] = home_matches['Over25'].fillna(0.5).mean()
        else:
            features['home_avg_odds'] = 2.0
            features['home_market_confidence'] = 0.5
            features['home_over25_tendency'] = 0.5
            
        # Away team market perception
        away_matches = df[(df['AwayTeam'] == away_team) & (df['OddAway'].notna())]
        if len(away_matches) > 0:
            features['away_avg_odds'] = away_matches['OddAway'].mean()
            features['away_market_confidence'] = 1 / features['away_avg_odds'] if features['away_avg_odds'] > 0 else 0.5
            features['away_over25_tendency'] = away_matches['Over25'].fillna(0.5).mean()
        else:
            features['away_avg_odds'] = 2.0
            features['away_market_confidence'] = 0.5
            features['away_over25_tendency'] = 0.5
            
        # Market intelligence
        features['market_home_advantage'] = features['away_market_confidence'] - features['home_market_confidence']
        features['total_goals_expectation'] = (features['home_over25_tendency'] + features['away_over25_tendency']) / 2
        
        return features
    
    def _extract_h2h_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract enhanced head-to-head features"""
        features = {}
        
        h2h_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
        
        if len(h2h_matches) > 0:
            # Recent H2H bias (last 5 meetings)
            recent_h2h = h2h_matches.tail(5)
            home_wins = len(recent_h2h[((recent_h2h['HomeTeam'] == home_team) & (recent_h2h['FTResult'] == 'H')) |
                                     ((recent_h2h['AwayTeam'] == home_team) & (recent_h2h['FTResult'] == 'A'))])
            
            features['h2h_home_win_rate'] = home_wins / len(recent_h2h)
            features['h2h_avg_goals'] = (recent_h2h['FTHome'] + recent_h2h['FTAway']).mean()
            features['h2h_meetings'] = len(h2h_matches)
            features['h2h_variance'] = (recent_h2h['FTHome'] - recent_h2h['FTAway']).var()
        else:
            features['h2h_home_win_rate'] = 0.5
            features['h2h_avg_goals'] = 2.5
            features['h2h_meetings'] = 0
            features['h2h_variance'] = 1.0
            
        return features
    
    def _extract_league_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract league-specific characteristics"""
        features = {}
        
        # Determine leagues (handle missing Div column)
        try:
            home_matches = df[df['HomeTeam'] == home_team]
            away_matches = df[df['AwayTeam'] == away_team]
            
            if 'Div' in df.columns and len(home_matches) > 0:
                home_league = home_matches['Div'].iloc[0] if pd.notna(home_matches['Div'].iloc[0]) else 'Unknown'
            else:
                home_league = 'Unknown'
                
            if 'Div' in df.columns and len(away_matches) > 0:
                away_league = away_matches['Div'].iloc[0] if pd.notna(away_matches['Div'].iloc[0]) else 'Unknown'
            else:
                away_league = 'Unknown'
        except:
            home_league = 'Unknown'
            away_league = 'Unknown'
        
        features['same_league'] = 1 if home_league == away_league else 0
        features['home_league_goals_avg'] = df[df['Div'] == home_league]['FTHome'].mean() if home_league != 'Unknown' else 1.5
        features['away_league_goals_avg'] = df[df['Div'] == away_league]['FTAway'].mean() if away_league != 'Unknown' else 1.2
        
        # League competitiveness
        if home_league != 'Unknown':
            league_matches = df[df['Div'] == home_league]
            features['league_competitiveness'] = (league_matches['FTResult'] == 'D').mean()  # Draw rate indicates competitiveness
        else:
            features['league_competitiveness'] = 0.25
            
        return features
    
    def _extract_temporal_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Extract temporal and seasonal patterns"""
        features = {}
        
        # This would require date parsing - simplified for now
        features['home_advantage'] = 0.15  # Standard home advantage
        features['fixture_congestion'] = 0.0  # Could be calculated from match frequency
        features['seasonal_form'] = 1.0  # Could extract from recent months
        
        return features
    
    def _calculate_elo_trend(self, team: str, df: pd.DataFrame) -> float:
        """Calculate ELO rating trend over recent matches"""
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)
        
        if len(team_matches) < 2:
            return 0
            
        elo_values = []
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                elo_values.append(match['HomeElo'])
            else:
                elo_values.append(match['AwayElo'])
                
        if len(elo_values) >= 2:
            return elo_values[-1] - elo_values[0]  # Trend over recent matches
        return 0
    
    def _fallback_basic_features(self, home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
        """Fallback to basic features if enhanced extraction fails"""
        return {
            'home_avg_elo': 1500,
            'away_avg_elo': 1500,
            'elo_difference': 0,
            'home_advantage': 0.15,
            'h2h_meetings': 0
        }
    
    def get_feature_vector(self, features: Dict) -> np.ndarray:
        """Convert features dictionary to numpy array for ML training"""
        feature_order = [
            'home_avg_elo', 'away_avg_elo', 'elo_difference', 'elo_trend',
            'home_form3', 'home_form5', 'away_form3', 'away_form5', 'form_difference',
            'home_shots_per_game', 'away_shots_per_game', 'shots_advantage',
            'home_shots_on_target_ratio', 'away_shots_on_target_ratio', 'accuracy_advantage',
            'home_aggression_index', 'away_aggression_index', 'aggression_clash',
            'home_market_confidence', 'away_market_confidence', 'market_home_advantage',
            'h2h_home_win_rate', 'h2h_avg_goals', 'h2h_meetings',
            'same_league', 'league_competitiveness', 'home_advantage'
        ]
        
        vector = []
        for feature in feature_order:
            if feature in features:
                vector.append(features[feature])
            elif feature == 'elo_trend':
                vector.append(features.get('home_elo_trend', 0) - features.get('away_elo_trend', 0))
            else:
                vector.append(0.0)  # Default value for missing features
                
        return np.array(vector, dtype=np.float32)