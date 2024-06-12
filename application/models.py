from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, TIMESTAMP, func

from application.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, nullable=False)
    username = Column(String, index=True, unique=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    email_verified = Column(Boolean, nullable=False, default=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)


class File(Base):
    __tablename__ = "files"

    file_id = Column(String, primary_key=True, unique=True, index=True, nullable=False)

    filename = Column(String, index=True, unique=False, nullable=False)
    file_ext = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, nullable=True)
    split_pages = Column(Boolean, default=False)

    deleted = Column(Boolean, default=False)

    owner_id = Column(Integer, ForeignKey("users.id"))

    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
