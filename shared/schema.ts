import { pgTable, text, integer, timestamp, real, boolean, serial } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// Users table for social features
export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  telegramId: text('telegram_id').unique().notNull(),
  username: text('username'),
  firstName: text('first_name'),
  joinedDate: timestamp('joined_date').defaultNow(),
  lastActive: timestamp('last_active').defaultNow(),
  confidencePoints: integer('confidence_points').default(1000),
  totalPredictions: integer('total_predictions').default(0),
  correctPredictions: integer('correct_predictions').default(0),
  currentStreak: integer('current_streak').default(0),
  bestStreak: integer('best_streak').default(0),
  rank: text('rank').default('Beginner'),
});

// Predictions table
export const predictions = pgTable('predictions', {
  id: serial('id').primaryKey(),
  userId: integer('user_id').references(() => users.id),
  homeTeam: text('home_team').notNull(),
  awayTeam: text('away_team').notNull(),
  league: text('league'),
  matchDate: timestamp('match_date'),
  prediction: text('prediction').notNull(),
  confidence: real('confidence').notNull(),
  marketBacked: boolean('market_backed').default(false),
  actualResult: text('actual_result'),
  pointsStaked: integer('points_staked').default(0),
  createdAt: timestamp('created_at').defaultNow(),
});

// Badges table
export const badges = pgTable('badges', {
  id: serial('id').primaryKey(),
  userId: integer('user_id').references(() => users.id),
  badgeName: text('badge_name').notNull(),
  earnedAt: timestamp('earned_at').defaultNow(),
});

// User relations
export const usersRelations = relations(users, ({ many }) => ({
  predictions: many(predictions),
  badges: many(badges),
}));

// Predictions relations
export const predictionsRelations = relations(predictions, ({ one }) => ({
  user: one(users, {
    fields: [predictions.userId],
    references: [users.id],
  }),
}));

// Badges relations
export const badgesRelations = relations(badges, ({ one }) => ({
  user: one(users, {
    fields: [badges.userId],
    references: [users.id],
  }),
}));

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;
export type Prediction = typeof predictions.$inferSelect;
export type InsertPrediction = typeof predictions.$inferInsert;
export type Badge = typeof badges.$inferSelect;
export type InsertBadge = typeof badges.$inferInsert;