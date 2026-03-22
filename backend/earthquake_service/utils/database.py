"""Async database connection via SQLAlchemy."""
import logging
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, QueuePool

from earthquake_service.config import settings

logger = logging.getLogger(__name__)

# Determine database type from URL
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
is_postgresql = settings.DATABASE_URL.startswith("postgresql")

# Configure engine based on database type
engine_kwargs = {
    "echo": False,  # Set to True for SQL debugging
    "future": True,
}

if is_sqlite:
    # SQLite configuration
    engine_kwargs.update({
        "connect_args": {"check_same_thread": False},
        "poolclass": NullPool,  # SQLite doesn't need connection pooling
    })
    logger.info("Configuring SQLite database engine")
    
elif is_postgresql:
    # PostgreSQL configuration with connection pooling
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "poolclass": QueuePool,
    })
    logger.info("Configuring PostgreSQL database engine with connection pooling")
    
else:
    # Default/fallback configuration
    logger.warning(f"Unknown database type: {settings.DATABASE_URL}. Using conservative settings.")
    engine_kwargs.update({
        "poolclass": NullPool,  # Safe default
    })

# Create the engine
try:
    engine = create_async_engine(
        settings.DATABASE_URL,
        **engine_kwargs
    )
    logger.info(f"Database engine created successfully for {settings.DATABASE_URL.split('://')[0]} database")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


async def init_db():
    """Create tables on startup."""
    try:
        # Import models here to avoid circular imports
        from earthquake_service.models.db_models import EarthquakeRecord, PredictionLog  # noqa
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database tables initialized successfully.")
        
        # For SQLite, log the database file location and size
        if is_sqlite:
            db_file = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
            if os.path.exists(db_file):
                file_size = os.path.getsize(db_file) / 1024  # KB
                logger.info(f"📁 SQLite database file: {db_file} ({file_size:.2f} KB)")
            else:
                logger.warning(f"⚠️ SQLite database file not found: {db_file}")
                
    except Exception as e:
        logger.error(f"❌ Failed to initialize database tables: {e}")
        raise


async def get_db():
    """
    Dependency for getting database sessions.
    Usage: Depends(get_db) in FastAPI routes
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db():
    """Close database connections on shutdown."""
    await engine.dispose()
    logger.info("Database connections closed.")


# Health check function
async def check_db_health() -> bool:
    """Check if database is reachable and working."""
    try:
        async with engine.connect() as conn:
            if is_sqlite:
                # SQLite health check
                result = await conn.execute("SELECT 1")
            else:
                # PostgreSQL health check
                result = await conn.execute("SELECT 1")
            await result.fetchone()
        logger.debug("Database health check passed")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False