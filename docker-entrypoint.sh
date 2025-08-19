#!/bin/bash
set -e

echo "🚀 Starting Sports Prediction Bot..."

# Wait for database to be ready
echo "⏳ Waiting for database connection..."
until pg_isready -h ${PGHOST:-localhost} -p ${PGPORT:-5432} -U ${PGUSER:-postgres}; do
  echo "Database is unavailable - sleeping"
  sleep 2
done

echo "✅ Database is ready!"

# Run database migrations if needed
echo "📊 Setting up database..."
python -c "
import asyncio
from database_manager import DatabaseManager

async def setup():
    db = DatabaseManager()
    await db.initialize()
    await db.create_tables_if_not_exist()
    await db.close()
    print('✅ Database setup complete!')

asyncio.run(setup())
"

# Start the main application
echo "🤖 Starting Sports Prediction Bot..."
exec python -m telegram_bot.bot