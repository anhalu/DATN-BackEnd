from sqlalchemy.orm import Session

from . import models, schemas


def get_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate, hashed_password: str) -> models.User:
    db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_files(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.File).filter(models.File.owner_id == user_id).offset(skip).limit(limit).all()


def get_file(db: Session, user_id: int, file_id: str):
    return db.query(models.File).filter(models.File.owner_id == user_id).filter(models.File.file_id == file_id).first()


def create_user_file(db: Session, item: schemas.FileCreate, user_id: int):
    db_item = models.File(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def save_file(db: Session, file: models.File):
    db.add(file)
    db.commit()
    db.refresh(file)
    return file
