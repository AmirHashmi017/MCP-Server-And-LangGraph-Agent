from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MONGO_DB_URI: str
    MONGODB_DB: str = "Volvox"
    GRIDFS_BUCKET: str = "fs"
    
    JWT_SECRET_KEY: str = "volvoxpersonalaiassistantresearc"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Volvox"
    
    ALLOWED_ORIGINS: List[str] = ["*"]

    USERS_COLLECTION: str = "users"
    RESEARCH_COLLECTION: str = "research"
    CHATHISTORY_COLLECTION: str= "chatHistory"

    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  
    )

    @field_validator("MONGO_DB_URI", mode="before")
    @classmethod
    def normalize_mongo_uri(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            return v.strip()
        return v

settings = Settings()
