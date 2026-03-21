-- BESPOKE V0 Database Schema
-- SQLite + sqlite-vec

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ============================================================
-- Stage 1: Raw interaction warehouse
-- ============================================================

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Source metadata
    provider TEXT NOT NULL,              -- 'claude', 'openai', 'gemini', etc.
    model TEXT NOT NULL,                 -- 'claude-opus-4-6', 'claude-sonnet-4-6', etc.
    source TEXT NOT NULL,                -- 'claude-code', 'cursor', 'claude-web', etc.
    session_id TEXT,                     -- Links multi-turn conversations

    -- Content
    system_prompt TEXT,                  -- System prompt if present
    user_message TEXT NOT NULL,          -- The user's input
    assistant_response TEXT NOT NULL,    -- The model's output

    -- Token counts
    input_tokens INTEGER,
    output_tokens INTEGER,

    -- Feedback signals
    user_followup TEXT,                  -- Next real user message (accept/reject signal for Stage 2a)
    feedback_class TEXT,                 -- 'strong_accept', 'accept', 'neutral', 'reject', 'strong_reject'
    feedback_raw TEXT,                   -- Stage 2a's reasoning for the feedback classification

    -- Stage 2a outputs
    domain TEXT,                         -- 'code', 'strategy', 'planning', 'organizing', etc.
    quality_score TEXT,                  -- 'high', 'medium', 'low', 'exclude'
    reasoning_primitives TEXT,           -- JSON array: ['decomposition', 'tradeoff_analysis', ...]

    -- Stage 2b outputs (retroactive tags)
    meta_patterns TEXT,                  -- JSON array of meta-pattern IDs
    curriculum_weight REAL DEFAULT 1.0,  -- Weighting for training data selection

    -- Tracking
    captured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    processed_2a_at TEXT,               -- When Stage 2a processed this record
    processed_2b_at TEXT,               -- When Stage 2b tagged this record

    -- Training usage
    used_in_sft INTEGER DEFAULT 0,      -- Has been used in SFT training
    used_in_dpo INTEGER DEFAULT 0,      -- Has been used in DPO training
    generation_source TEXT DEFAULT 'frontier',  -- 'frontier' or 'specialist' (for flywheel tracking)
    content_hash TEXT                        -- SHA-256 of session_id + user_message + assistant_response
);

-- ============================================================
-- Stage 2a: Training pairs (extracted from interactions)
-- ============================================================

CREATE TABLE IF NOT EXISTS training_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL REFERENCES interactions(id),
    pair_type TEXT NOT NULL,             -- 'sft' or 'dpo'
    domain TEXT NOT NULL,

    -- SFT pair
    instruction TEXT,
    response TEXT,

    -- DPO pair (preference)
    prompt TEXT,
    chosen TEXT,                         -- Preferred response
    rejected TEXT,                       -- Rejected response

    -- Quality
    quality_score TEXT NOT NULL,
    reasoning_primitives TEXT,           -- JSON array

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================
-- Stage 2b: Meta-pattern registry
-- ============================================================

CREATE TABLE IF NOT EXISTS meta_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    trigger_conditions TEXT,             -- When this pattern applies
    reasoning_steps TEXT,                -- The reasoning sequence
    occurrence_count INTEGER DEFAULT 0,
    domains TEXT,                        -- JSON array of domains where observed

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================
-- Experiment tracking (Stage 3)
-- ============================================================

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adapter_name TEXT NOT NULL,          -- 'code-v1', 'planner-v3', etc.

    -- Configuration
    base_model TEXT NOT NULL,
    training_config TEXT NOT NULL,       -- JSON: rank, lr, epochs, etc.
    data_config TEXT NOT NULL,           -- JSON: domain filter, quality threshold, etc.

    -- Results
    benchmark_version INTEGER,
    scorecard TEXT,                      -- JSON: full scorecard output
    reward_vector TEXT,                  -- JSON: per-dimension scores
    predicted_accept_rate REAL,

    -- Outcome
    decision TEXT NOT NULL,              -- 'keep' or 'revert'
    decision_reasoning TEXT,

    -- Previous baseline
    previous_experiment_id INTEGER REFERENCES experiments(id),
    previous_reward_vector TEXT,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================
-- Pipeline state tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS pipeline_state (
    stage TEXT PRIMARY KEY,              -- 'stage1', 'stage2a', 'stage2b', 'stage3'
    last_successful_run TEXT,            -- ISO timestamp
    last_record_processed INTEGER,       -- interaction.id of last processed record
    metadata TEXT                        -- JSON: any stage-specific state
);

-- Initialize pipeline state
INSERT OR IGNORE INTO pipeline_state (stage, metadata) VALUES
    ('stage1', '{}'),
    ('stage2a', '{}'),
    ('stage2b', '{}'),
    ('stage3', '{}');

-- ============================================================
-- Embedding vector table (sqlite-vec)
-- ============================================================

-- This is created programmatically after loading sqlite-vec extension
-- See bespoke/db/init.py for the vec0 virtual table creation

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_interactions_captured_at ON interactions(captured_at);
CREATE INDEX IF NOT EXISTS idx_interactions_domain ON interactions(domain);
CREATE INDEX IF NOT EXISTS idx_interactions_quality ON interactions(quality_score);
CREATE INDEX IF NOT EXISTS idx_interactions_feedback ON interactions(feedback_class);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_processed_2a ON interactions(processed_2a_at);
CREATE INDEX IF NOT EXISTS idx_interactions_generation ON interactions(generation_source);
CREATE INDEX IF NOT EXISTS idx_training_pairs_domain ON training_pairs(domain);
CREATE INDEX IF NOT EXISTS idx_training_pairs_type ON training_pairs(pair_type);
CREATE INDEX IF NOT EXISTS idx_experiments_adapter ON experiments(adapter_name);
