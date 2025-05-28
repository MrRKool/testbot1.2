import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Optional, Generator

Base = declarative_base()

class DatabaseManager:
    """Database manager for handling database connections and operations."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize database manager with connection settings."""
        self.logger = logging.getLogger(__name__)
        
        # Use provided URL or default to SQLite
        self.db_url = db_url or 'sqlite:///trading_bot.db'
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        self.logger.info("Database initialized successfully")
    
    @contextmanager
    def get_session(self) -> Generator:
        """Get a database session with automatic cleanup."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """Execute a raw SQL query."""
        with self.get_session() as session:
            result = session.execute(query, params or {})
            return result.fetchall()
    
    def add_record(self, record: Base) -> None:
        """Add a new record to the database."""
        with self.get_session() as session:
            session.add(record)
    
    def update_record(self, record: Base) -> None:
        """Update an existing record."""
        with self.get_session() as session:
            session.merge(record)
    
    def delete_record(self, record: Base) -> None:
        """Delete a record from the database."""
        with self.get_session() as session:
            session.delete(record)
    
    def get_record(self, model: Base, **kwargs) -> Optional[Base]:
        """Get a single record by filter criteria."""
        with self.get_session() as session:
            return session.query(model).filter_by(**kwargs).first()
    
    def get_records(self, model: Base, **kwargs) -> list:
        """Get multiple records by filter criteria."""
        with self.get_session() as session:
            return session.query(model).filter_by(**kwargs).all()
    
    def close(self) -> None:
        """Close all database connections."""
        self.engine.dispose()
        self.logger.info("Database connections closed") 