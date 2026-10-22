"""Edit these to match your extract. Canonical name -> column name in your CSV."""

COLUMNS = {
    "date": "date",
    "org": "org",
    "team": "team",
    "sid": "SID",
    "model": "model",
    "token_type": "token_type",
    "tokens": "tokens",
}

# Used only when the CSV is wide (one column per token type, no token_type column).
WIDE_TOKEN_COLUMNS = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_tokens": "cache_read",
    "cache_write_tokens": "cache_write",
}

TOKEN_TYPE_ALIASES = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache read": "cache_read",
    "cache_read_input_tokens": "cache_read",
    "cache_read_tokens": "cache_read",
    "cache write": "cache_write",
    "cache_creation": "cache_write",
    "cache_creation_input_tokens": "cache_write",
    "cache_write_tokens": "cache_write",
}

TOKEN_TYPES = ["input", "output", "cache_read", "cache_write"]

# "start" if the date column is the first day of the week, "end" if it is the last.
WEEK_DATE_IS = "start"

# Relative weight of a Saturday/Sunday vs a weekday when splitting weeks into calendar months.
WEEKEND_WEIGHT = 0.15

HORIZON_WEEKS = 13
FIRST_TRAIN_WEEKS = 20
ORIGIN_STEP_WEEKS = 2

LEVELS = {
    "org": ["org"],
    "team": ["org", "team"],
    "team_model": ["org", "team", "model"],
    "sid": ["org", "team", "sid"],
}
