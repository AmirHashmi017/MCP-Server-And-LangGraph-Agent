from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.config import settings

class Database:
    client: AsyncIOMotorClient = None
    
db = Database()

async def get_database():
    return db.client[settings.MONGODB_DB]

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.MONGO_DB_URI)
    # Force initial connection to surface SRV/DNS errors early
    try:
        await db.client.admin.command("ping")
        print("Connected to MongoDB!")
    except Exception as e:
        # Provide a concise hint for SRV/DNS issues
        print(f"MongoDB connection failed: {e}")
        raise

async def close_mongo_connection():
    db.client.close()
    print("MongoDB connection closed!")

async def get_collection(collection_name: str):
    database = await get_database()
    return database[collection_name]

from typing import Optional

async def get_gridfs_bucket(bucket_name: Optional[str] = None) -> AsyncIOMotorGridFSBucket:
    database = await get_database()
    bucket = AsyncIOMotorGridFSBucket(database, bucket_name=bucket_name or settings.GRIDFS_BUCKET)
    return bucket
