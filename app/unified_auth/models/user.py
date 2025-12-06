from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator
from typing import Optional
from datetime import datetime,timezone

class UserModel(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    email: EmailStr
    hashed_password: str
    fullName: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        populate_by_name=True,
    )

    # Convert Mongo ObjectId to string when loading from DB
    @field_validator("id", mode="before")
    @classmethod
    def _convert_objectid_to_str(cls, v):
        if v is None:
            return v
        return str(v)
