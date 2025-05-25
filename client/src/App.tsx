import React, { useState, useEffect } from 'react';
import { Leaderboard } from './components/Leaderboard';
import { UserProfile } from './components/UserProfile';
import { CommunityFeed } from './components/CommunityFeed';
import { CommunityStats } from './components/CommunityStats';
import './App.css';

declare global {
  interface Window {
    Telegram: {
      WebApp: {
        ready: () => void;
        close: () => void;
        expand: () => void;
        initData: string;
        initDataUnsafe: {
          user?: {
            id: number;
            first_name: string;
            last_name?: string;
            username?: string;
          };
        };
        themeParams: {
          bg_color: string;
          text_color: string;
          hint_color: string;
          link_color: string;
          button_color: string;
          button_text_color: string;
        };
      };
    };
  }
}

type Tab = 'profile' | 'leaderboard' | 'feed' | 'community';

interface User {
  id: number;
  telegramId: string;
  username?: string;
  firstName?: string;
  totalPredictions: number;
  correctPredictions: number;
  currentStreak: number;
  bestStreak: number;
  confidencePoints: number;
  rank: string;
  accuracy: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('profile');
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Initialize Telegram Web App
    if (window.Telegram?.WebApp) {
      const tgApp = window.Telegram.WebApp;
      tgApp.ready();
      tgApp.expand();

      // Apply Telegram theme
      const theme = tgApp.themeParams;
      if (theme.bg_color) {
        document.documentElement.style.setProperty('--tg-bg-color', theme.bg_color);
        document.documentElement.style.setProperty('--tg-text-color', theme.text_color);
        document.documentElement.style.setProperty('--tg-hint-color', theme.hint_color);
        document.documentElement.style.setProperty('--tg-link-color', theme.link_color);
        document.documentElement.style.setProperty('--tg-button-color', theme.button_color);
        document.documentElement.style.setProperty('--tg-button-text-color', theme.button_text_color);
      }

      // Get user info and load profile
      const telegramUser = tgApp.initDataUnsafe.user;
      if (telegramUser) {
        loadUserProfile(telegramUser.id.toString());
      }
    }

    // Setup WebSocket connection for real-time updates
    setupWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const setupWebSocket = () => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    const websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
      console.log('Connected to Mini App WebSocket');
      setWs(websocket);
      
      // Subscribe to leaderboard updates
      websocket.send(JSON.stringify({ type: 'subscribe_leaderboard' }));
    };
    
    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'leaderboard_update') {
          // Handle real-time leaderboard updates
          console.log('Leaderboard updated:', data.data);
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };
    
    websocket.onclose = () => {
      console.log('Disconnected from Mini App WebSocket');
      // Attempt to reconnect after 3 seconds
      setTimeout(setupWebSocket, 3000);
    };
  };

  const loadUserProfile = async (telegramId: string) => {
    try {
      setLoading(true);
      
      // First, ensure user exists in database
      await fetch('/api/user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          telegramId,
          username: window.Telegram?.WebApp?.initDataUnsafe?.user?.username,
          firstName: window.Telegram?.WebApp?.initDataUnsafe?.user?.first_name
        })
      });

      // Then fetch user profile
      const response = await fetch(`/api/user/${telegramId}`);
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      }
    } catch (error) {
      console.error('Error loading user profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderTabContent = () => {
    if (loading) {
      return (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your profile...</p>
        </div>
      );
    }

    switch (activeTab) {
      case 'profile':
        return <UserProfile user={user} />;
      case 'leaderboard':
        return <Leaderboard currentUser={user} />;
      case 'feed':
        return <CommunityFeed />;
      case 'community':
        return <CommunityStats />;
      default:
        return <UserProfile user={user} />;
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏆 Sports Prediction Community</h1>
        <p>Market-driven predictions & social competition</p>
      </header>

      <nav className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          👤 Profile
        </button>
        <button 
          className={`tab-button ${activeTab === 'leaderboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('leaderboard')}
        >
          🏆 Rankings
        </button>
        <button 
          className={`tab-button ${activeTab === 'feed' ? 'active' : ''}`}
          onClick={() => setActiveTab('feed')}
        >
          📱 Feed
        </button>
        <button 
          className={`tab-button ${activeTab === 'community' ? 'active' : ''}`}
          onClick={() => setActiveTab('community')}
        >
          📊 Stats
        </button>
      </nav>

      <main className="tab-content">
        {renderTabContent()}
      </main>
    </div>
  );
}