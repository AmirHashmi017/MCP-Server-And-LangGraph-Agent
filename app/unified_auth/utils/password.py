from passlib.context import CryptContext

# Use bcrypt_sha256 to avoid 72-byte bcrypt password limit
# Keep bcrypt in the schemes for backward compatibility if any hashes exist
pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
