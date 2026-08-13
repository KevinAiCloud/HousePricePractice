"""Script to create all database tables."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.base import Base
from data_models.users import User
from data_models.chunks import Document, Chunk, Embedding
from data_models.session import engine

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")
