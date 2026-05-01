import os

# Must run before any src imports — pydantic-settings reads env at class instantiation time.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://synthesizr:synthesizr@localhost:5432/synthesizr")
