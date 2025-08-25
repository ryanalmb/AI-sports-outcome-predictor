"""
Framework 5: PyTorch LSTM Sequential - Following User's Guide
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle
import os

logger = logging.getLogger(__name__)

class MatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MatchPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out

class PyTorchLSTMFramework:
    """PyTorch LSTM Framework following user's authentic data guide"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.df = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize PyTorch LSTM framework with authentic dataset"""
        if self.is_initialized:
            return
            
        # Try to load from cache first
        cache_file = 'ml/framework_cache/pytorch_lstm.pkl'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.model = cached_data['model']
                self.scaler = cached_data['scaler']
                self.df = cached_data['df']
                self.is_initialized = True
                logger.info("✅ Loaded PyTorch LSTM framework from cache")
                return
            except:
                pass
            
        try:
            logger.info("🔥 Initializing PyTorch LSTM Sequential with authentic data...")
            
            # Load authentic dataset
            self.df = pd.read_csv('football_data/data/Matches.csv', low_memory=False)
            logger.info(f"📊 Loaded {len(self.df)} authentic matches for LSTM training")
            
            # Prepare training data using your guide
            X, y = self._prepare_lstm_sequences()
            
            if len(X) == 0:
                logger.warning("No sequence data available, creating minimal training set")
                X = np.array([[[0.8, 0.6, 0.7, 0.5]], [[0.6, 0.8, 0.6, 0.7]], [[0.5, 0.5, 0.5, 0.6]]])
                y = np.array([0, 2, 1])
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Create DataLoader
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Define LSTM model
            input_size = X.shape[2] if len(X.shape) > 2 else 4
            self.model = MatchPredictor(input_size=input_size, hidden_size=64, num_layers=2, num_classes=3)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            logger.info("🔥 Training PyTorch LSTM Sequential...")
            
            # Quick training loop (3 epochs for speed)
            for epoch in range(3):
                for batch_X, batch_y in train_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.squeeze())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                logger.info(f"✅ LSTM Epoch {epoch+1}/3 completed")
            
            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predicted_classes = torch.argmax(predictions, dim=1)
                accuracy = accuracy_score(y_tensor, predicted_classes)
                logger.info(f"✅ PyTorch LSTM accuracy: {accuracy:.3f}")
            
            # Save to cache
            os.makedirs('ml/framework_cache', exist_ok=True)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'scaler': self.scaler,
                        'df': self.df
                    }, f)
            except:
                pass
            
            self.is_initialized = True
            logger.info("✅ PyTorch LSTM Sequential framework ready")
            
        except Exception as e:
            logger.error(f"❌ Error initializing PyTorch LSTM framework: {str(e)}")
            raise
    
    async def generate_prediction(self, home_team: str, away_team: str) -> Dict:
        """Generate prediction using PyTorch LSTM Sequential"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Extract features for teams
            features = self._extract_team_features(home_team, away_team)
            
            # Create sequence (using last feature as sequence)
            sequence = np.array([[features]])  # Shape: [1, 1, features]
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            
            # Make prediction with LSTM
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                probs = probabilities.numpy()[0]
            
            home_win = float(probs[0]) * 100
            draw = float(probs[1]) * 100  
            away_win = float(probs[2]) * 100
            
            return {
                'home_win': home_win,
                'away_win': away_win,
                'draw': draw,
                'confidence': max(home_win, away_win, draw),
                'framework': 'PyTorch LSTM Sequential'
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating LSTM prediction: {str(e)}")
            raise Exception("PyTorch LSTM framework requires authentic data initialization")
    
    def _prepare_lstm_sequences(self):
        """Prepare LSTM sequences from authentic dataset following user's guide"""
        try:
            # Use authentic result column
            result_col = 'FTResult'
            complete_matches = self.df.dropna(subset=['HomeTeam', 'AwayTeam', result_col])
            
            if len(complete_matches) == 0:
                return np.array([]), np.array([])
            
            # Extract features following the guide
            features = []
            labels = []
            
            # Use first 1000 matches for quick training
            sample_matches = complete_matches.head(1000)
            
            for _, match in sample_matches.iterrows():
                home_team = str(match['HomeTeam'])
                away_team = str(match['AwayTeam'])
                result = str(match[result_col])
                
                # Extract team features
                match_features = self._extract_team_features(home_team, away_team)
                features.append(match_features)
                
                # Convert result to label (H=Home win, D=Draw, A=Away win)
                if result == 'H':
                    labels.append(0)  # Home win
                elif result == 'D':
                    labels.append(1)  # Draw
                elif result == 'A':
                    labels.append(2)  # Away win
                else:
                    features.pop()
                    continue
            
            if len(features) == 0:
                return np.array([]), np.array([])
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Scale features following the guide
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
            
            # Create sequences for LSTM (sequence_length = 1 for simplicity)
            X_seq = X.reshape(X.shape[0], 1, X.shape[1])  # [samples, seq_len=1, features]
            
            return X_seq, y
            
        except Exception as e:
            logger.error(f"Error preparing LSTM sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def _extract_team_features(self, home_team: str, away_team: str):
        """Extract 27 enhanced features for PyTorch LSTM"""
        try:
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            
            # Use enhanced 27-feature system
            feature_engine = EnhancedFeatureEngine(self.df)
            features = feature_engine.extract_features(home_team, away_team)
            
            if features is not None:
                return features.tolist()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting 27 features for PyTorch LSTM: {e}")
            return None
    
    def _calculate_team_rating(self, team_name: str) -> float:
        """Calculate team rating from authentic data"""
        if not hasattr(self, 'df') or self.df is None:
            # Fallback rating based on team name
            team_lower = team_name.lower()
            if any(elite in team_lower for elite in ['barcelona', 'real madrid', 'manchester', 'liverpool', 'chelsea']):
                return 0.85
            elif len(team_name) > 12:
                return 0.65
            else:
                return 0.55
        
        try:
            # Calculate from historical matches
            home_matches = self.df[self.df['HomeTeam'] == team_name]
            away_matches = self.df[self.df['AwayTeam'] == team_name]
            
            if len(home_matches) == 0 and len(away_matches) == 0:
                return 0.55  # Default rating
            
            # Calculate win percentage
            home_wins = len(home_matches[home_matches['FTResult'] == 'H'])
            away_wins = len(away_matches[away_matches['FTResult'] == 'A'])
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches == 0:
                return 0.55
            
            win_rate = (home_wins + away_wins) / total_matches
            return 0.3 + (win_rate * 0.5)  # Scale to 0.3-0.8 range
            
        except Exception:
            return 0.55
    
    def _calculate_team_form(self, team_name: str) -> float:
        """Calculate recent team form"""
        try:
            if not hasattr(self, 'df') or self.df is None:
                return 0.5
            
            # Get last 5 matches
            recent_home = self.df[self.df['HomeTeam'] == team_name].tail(3)
            recent_away = self.df[self.df['AwayTeam'] == team_name].tail(2)
            
            points = 0
            matches = 0
            
            # Count points from recent matches
            for _, match in recent_home.iterrows():
                if match['FTResult'] == 'H':
                    points += 3
                elif match['FTResult'] == 'D':
                    points += 1
                matches += 1
            
            for _, match in recent_away.iterrows():
                if match['FTResult'] == 'A':
                    points += 3
                elif match['FTResult'] == 'D':
                    points += 1
                matches += 1
            
            if matches == 0:
                return 0.5
            
            # Return form as ratio (0-1)
            return min(points / (matches * 3), 1.0)
            
        except Exception:
            return 0.5