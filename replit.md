# Sports Prediction Telegram Bot

## Overview

This is a Python-based Telegram bot that provides sports predictions using machine learning. The bot integrates with multiple sports APIs to collect real-time data and uses trained ML models to predict match outcomes for various sports including football (soccer), UFC, and boxing.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular, service-oriented architecture with clear separation of concerns:

### Core Components
- **Main Application (`main.py`)**: Entry point that orchestrates all components
- **Telegram Bot Interface (`bot/`)**: Handles user interactions via Telegram
- **Data Collection Layer (`data/`)**: Fetches sports data from external APIs
- **Machine Learning Engine (`ml/`)**: Provides prediction capabilities
- **Configuration Management (`config/`)**: Centralizes settings and API keys
- **Utilities (`utils/`)**: Common helper functions for data formatting

### Architecture Pattern
The system uses an async/await pattern throughout for non-blocking operations, essential for handling multiple concurrent users and API calls.

## Key Components

### 1. Telegram Bot (`bot/telegram_bot.py`)
- **Purpose**: Primary user interface for the bot
- **Features**: Command handling, inline keyboards, user session management
- **Commands**: `/start`, `/help`, `/leagues`, `/upcoming`, `/predict`, `/stats`
- **Design**: Event-driven architecture using python-telegram-bot library

### 2. Sports Data Collector (`data/sports_api.py`)
- **Purpose**: Aggregates data from multiple sports APIs
- **APIs Supported**: ESPN, Football-Data.org, SportsRadar
- **Features**: Rate limiting, caching, error handling
- **Caching Strategy**: 5-minute cache duration to reduce API calls

### 3. ML Prediction Engine (`ml/`)
- **Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Features**: Sport-specific models, confidence scoring, prediction history
- **Training Data**: Historical match results, team statistics, player data
- **Outcomes**: Match winners, score predictions, confidence levels

### 4. Configuration System (`config/settings.py`)
- **Environment Variables**: API keys, bot tokens, feature flags
- **Fallback Values**: Sensible defaults for development
- **Security**: API keys stored as environment variables

## Data Flow

1. **User Request**: User sends command via Telegram
2. **Command Processing**: Bot parses command and determines required data
3. **Data Collection**: System fetches current sports data from APIs
4. **Feature Engineering**: Raw data transformed into ML-ready features
5. **Prediction Generation**: ML models generate predictions with confidence scores
6. **Response Formatting**: Results formatted for Telegram display
7. **User Response**: Formatted prediction sent back to user

### Caching Strategy
- API responses cached for 5 minutes to reduce external calls
- Prediction results cached to avoid redundant calculations
- User session state maintained for conversation context

## External Dependencies

### Required APIs
- **Telegram Bot API**: Core bot functionality
- **ESPN API**: General sports data (free tier)
- **Football-Data.org**: Detailed football statistics
- **SportsRadar**: Professional sports data (premium features)

### Python Libraries
- `python-telegram-bot`: Telegram integration
- `aiohttp`: Async HTTP client for API calls
- `scikit-learn`: Machine learning models
- `pandas`/`numpy`: Data manipulation
- `pickle`: Model serialization

### Environment Variables Required
```
TELEGRAM_BOT_TOKEN=your_bot_token
FOOTBALL_DATA_API_KEY=your_api_key
SPORTS_RADAR_API_KEY=your_api_key
ESPN_API_KEY=your_api_key (optional)
```

## Deployment Strategy

### Current Setup (Replit)
- **Runtime**: Python 3.11 with Nix package management
- **Auto-install**: Dependencies installed via pip on startup
- **Process**: Single-threaded async application
- **Port**: Configured for port 5000 (though bot doesn't serve HTTP)

### Production Considerations
- **Scalability**: Currently single-instance, could be horizontally scaled
- **Database**: Currently in-memory, SQLite configured for persistence
- **Monitoring**: Basic logging configured, could add metrics
- **Rate Limiting**: Built-in API rate limiting to respect external service limits

### Key Design Decisions

1. **Async Architecture**: Chosen for handling multiple concurrent users efficiently
2. **Modular Design**: Separated concerns for maintainability and testing
3. **Multiple API Sources**: Reduces dependency on single data provider
4. **Caching Strategy**: Balances data freshness with API rate limits
5. **Environment-based Config**: Enables easy deployment across environments
6. **Error Handling**: Comprehensive error handling to maintain bot stability

The system is designed to be resilient, scalable, and maintainable while providing accurate sports predictions through a user-friendly Telegram interface.