import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_database_engine():
    """
    Create and return the SQLAlchemy engine used
    to connect Python with Neon PostgreSQL.

    The database URL is loaded from the .env file
    located in the project root.
    """

 
    project_root = Path(__file__).resolve().parent.parent

    # Build the complete path to the .env file
    env_path = project_root / ".env"

    # Verify that .env exists
    if not env_path.exists():
        raise FileNotFoundError(
            f".env file was not found at: {env_path}"
        )

    # 3. Load environment variables
    load_dotenv(
        dotenv_path=env_path,
        override=True
    )

    # Read DATABASE_URL from .env
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError(
            "DATABASE_URL was not found inside the .env file."
        )

    # 4. Create SQLAlchemy engine
    engine = create_engine(db_url)

    return engine