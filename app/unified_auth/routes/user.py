from fastapi import APIRouter, HTTPException, status, Depends
from app.unified_auth.schemas.user import (
    UserSignupRequest, 
    UserLoginRequest, 
    TokenResponse, 
    UserResponse,
    MessageResponse
)
from app.unified_auth.models.user import UserModel
from app.unified_auth.utils.password import hash_password, verify_password
from app.unified_auth.utils.auth import create_access_token
from app.database import get_collection
from app.config import settings
from app.unified_auth.middleware.auth import get_current_user
from datetime import datetime, timezone
from bson import ObjectId

async def signup(user_data: UserSignupRequest):
    users_collection = await get_collection(settings.USERS_COLLECTION)
    
    existing_email = await users_collection.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    
    user_dict = {
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "fullName": user_data.fullName,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    
    result = await users_collection.insert_one(user_dict)
    user_dict["_id"] = result.inserted_id
    
    access_token = create_access_token(data={"sub": str(result.inserted_id)})
    
    user_response = UserResponse(
        _id=str(user_dict["_id"]),
        email=user_dict["email"],
        fullName= user_dict['fullName'],
        created_at=user_dict["created_at"]
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )

async def login(credentials: UserLoginRequest):
    users_collection = await get_collection(settings.USERS_COLLECTION)
    
    user = await users_collection.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    
    access_token = create_access_token(data={"sub": str(user["_id"])})
    
    user_response = UserResponse(
        _id=str(user["_id"]),
        email=user["email"],
        fullName= user["fullName"],
        created_at=user["created_at"]
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )
