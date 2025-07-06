from datetime import datetime
from typing import Optional
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, BigInteger, Boolean, DateTime, func, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Mapped, mapped_column

DATABASE_URL = "sqlite:///./marketplace.db"

Base = declarative_base()

class UserTask(Base):
    __tablename__ = 'user_tasks'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    task_id = Column(String, index=True) # e.g., 'subscribe_channel'

    user = relationship("User", back_populates="completed_tasks")


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(BigInteger, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    balance = Column(Float, default=0.0)
    frozen_balance = Column(Float, default=0.0)
    photo_url = Column(String, nullable=True)
    bonuses = Column(Integer, default=0)
    last_daily_bonus: Mapped[Optional[datetime]] = Column(DateTime, nullable=True)
    first_purchase_bonus_received: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa.false())
    first_sale_bonus_received: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa.false())

    items = relationship('Item', back_populates='seller')
    deals_as_buyer = relationship('Deal', foreign_keys='Deal.buyer_id', back_populates='buyer')
    deals_as_seller = relationship('Deal', foreign_keys='Deal.seller_id', back_populates='seller')
    completed_tasks = relationship("UserTask", back_populates="user")

class Game(Base):
    __tablename__ = 'games'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)
    cover_image_url = Column(String, nullable=True)

    items = relationship("Item", back_populates="game")

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Float)
    currency = Column(String, default='USD')
    image_url = Column(String, nullable=True)
    is_sold = Column(Boolean, default=False, nullable=False)
    seller_id = Column(Integer, ForeignKey('users.id'))
    game_id = Column(Integer, ForeignKey('games.id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    seller = relationship("User", back_populates="items")
    game = relationship("Game", back_populates="items")

class Deal(Base):
    __tablename__ = 'deals'
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey('items.id'))
    buyer_id = Column(Integer, ForeignKey('users.id'))
    seller_id = Column(Integer, ForeignKey('users.id'))
    price = Column(Float) # Price at the time of the deal
    status = Column(String, default='frozen') # frozen, shipped, completed, disputed, canceled
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    item = relationship("Item")
    buyer = relationship("User", foreign_keys=[buyer_id], back_populates="deals_as_buyer")
    seller = relationship("User", foreign_keys=[seller_id], back_populates="deals_as_seller")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # This function creates all database tables
    Base.metadata.create_all(bind=engine)

    # --- Developer convenience: Set balance for dev user on startup ---
    DEV_USER_TELEGRAM_ID = 8043784997
    DEV_USER_BALANCE = 99999.0

    db = SessionLocal()
    try:
        # Find the developer user by their Telegram ID
        dev_user = db.query(User).filter(User.telegram_id == DEV_USER_TELEGRAM_ID).first()
        if dev_user:
            # To avoid unnecessary database writes, only update if the balance is lower than desired
            if dev_user.balance < DEV_USER_BALANCE:
                dev_user.balance = DEV_USER_BALANCE
                db.commit()
                print(f"INFO:     Set balance for dev user {DEV_USER_TELEGRAM_ID} to {DEV_USER_BALANCE}")
        # Optional: create the user if they don't exist? For now, we just log it.
        else:
            print(f"INFO:     Dev user {DEV_USER_TELEGRAM_ID} not found, balance not set.")
    except Exception as e:
        print(f"ERROR:    Could not set dev user balance during init: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Resetting database...")
    # Drop all tables first
    Base.metadata.drop_all(bind=engine)
    print("Tables dropped.")
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database re-created successfully.")