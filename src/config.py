"""
Configuration parameters for HireLens Recruiter.
"""

import os

# LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

# Matching Weights
SKILL_MATCH_WEIGHT = 0.50     # 50% weight on explicit skill match
SEMANTIC_MATCH_WEIGHT = 0.25  # 25% weight on TF-IDF semantic similarity
EXP_MATCH_WEIGHT = 0.15       # 15% weight on experience level match
DOMAIN_MATCH_WEIGHT = 0.10    # 10% weight on domain keyword match

# Evaluation & Scoring Thresholds
EXCELLENT_MATCH_THRESHOLD = 80.0
GOOD_MATCH_THRESHOLD = 60.0
FAIR_MATCH_THRESHOLD = 40.0

# Privacy Settings
STRICT_DEMOGRAPHIC_SCRUBBING = True
