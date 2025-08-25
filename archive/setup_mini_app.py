"""
Setup script to configure Telegram Mini App menu button
"""
import os
import requests

def setup_mini_app_menu():
    """Set up the Mini App menu button for the bot"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    mini_app_url = "https://5000-workspace-d7c95a73-1f4b-479f-af8b-54764b7b4a28-00-39tfhp7mxq1gm.janeway.replit.dev"
    
    # Set the chat menu button to open the Mini App
    url = f"https://api.telegram.org/bot{bot_token}/setChatMenuButton"
    
    payload = {
        "menu_button": {
            "type": "web_app",
            "text": "Community Hub",
            "web_app": {
                "url": mini_app_url
            }
        }
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("✅ Mini App menu button configured successfully!")
        print(f"🚀 Mini App URL: {mini_app_url}")
        print("Users can now access the Community Hub via the menu button in your bot!")
    else:
        print(f"❌ Error setting up menu button: {response.text}")
    
    return response.json()

if __name__ == "__main__":
    result = setup_mini_app_menu()
    print(f"Result: {result}")