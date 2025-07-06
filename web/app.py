import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import defaultdict
from itertools import cycle
import shutil
import uuid
import io
import os
import requests
from sqlalchemy import or_
from fastapi import FastAPI, Request, Depends, HTTPException, Response, Form, File, UploadFile, BackgroundTasks
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from database import SessionLocal, engine, init_db, User, Item, Deal, Game, UserTask
from contextlib import asynccontextmanager
import httpx

# Initialize database
init_db()

# Create a directory for uploads if it doesn't exist
UPLOADS_DIR = "web/uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# --- CONSTANTS ---
SUPPORTED_GAMES = [
    "Dota 2",
    "Counter-Strike 2",
    "CoD: Warzone",
    "Black Russia",
    "Fortnite",
    "Overwatch 2",
    "Sea of Thieves",
    "Apex Legends",
    "Genshin Impact",
    "Team Fortress 2",
    "Valorant",
    "Rust",
    "World of Warcraft",
    "Escape from Tarkov",
    "Telegram Premium",
    "Spotify",
    "Discord"
]

app = FastAPI()

# --- Middleware, Static Files, and Templates Setup ---
origins = ["*"]  # Allows all origins. For production, you should restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories and mount static files
# --- Ensure legacy avatar path exists ---
os.makedirs("web/static/images/catalog", exist_ok=True)
avatar_src = "web/static/images/my-avatar.png"
avatar_dst = "web/static/images/catalog/my-avatar.png"
if os.path.exists(avatar_src) and not os.path.exists(avatar_dst):
    try:
        shutil.copy(avatar_src, avatar_dst)
    except Exception as e:
        print(f"Could not copy default avatar to catalog path: {e}")
UPLOADS_DIR = "web/uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="web/templates")

# --- DATABASE DEPENDENCY ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# --- PRODUCT IMPORT LOGIC ---
import os
import shutil
import uuid
import random
import json
import re

# --- CONFIGURATION for IMPORTER ---
# The root directory where each sub-folder is a game to be imported.
SOURCE_GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "import_data")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")

@app.get("/import-products")
def run_product_import(db: Session = Depends(get_db)):
    """
    An endpoint to trigger a flexible product import process.
    - Scans the `import_data` directory for game folders.
    - For each game folder, it creates the game in the DB if it doesn't exist.
    - It then imports all product sub-folders, reading title/price from info.json.
    - Assigns sellers sequentially from a shuffled list to ensure even distribution.
    """
    # --- Read and prepare seller names ---
    names_file_path = os.path.join(os.path.dirname(__file__), "..", "names.txt")
    if not os.path.exists(names_file_path):
        return {"status": "error", "message": "names.txt not found in the project root."}
    with open(names_file_path, 'r', encoding='utf-8') as f:
        seller_names = [re.sub(r'^\d+\.\s*', '', line.strip()) for line in f if line.strip()]
    if not seller_names:
        return {"status": "error", "message": "names.txt is empty or contains no valid names."}

    # Shuffle the list once and create an infinite cycle for even distribution
    random.shuffle(seller_names)
    seller_cycle = cycle(seller_names)

    if not os.path.exists(SOURCE_GAMES_DIR):
        return {"status": "error", "message": f"Source directory not found at '{SOURCE_GAMES_DIR}'"}

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    total_imported_count = 0
    total_skipped_count = 0
    import_log = defaultdict(list)
    error_log = defaultdict(list)

    # --- Iterate over each game directory in the source directory ---
    for game_name in os.listdir(SOURCE_GAMES_DIR):
        game_folder_path = os.path.join(SOURCE_GAMES_DIR, game_name)
        if not os.path.isdir(game_folder_path):
            continue

        # --- Find or create the game in the database ---
        game_name_from_folder = os.path.basename(game_folder_path)
        
        # Normalize the folder name for a more robust lookup
        normalized_folder_name = re.sub(r'[^a-zA-Z0-9]', '', game_name_from_folder).lower()

        # Find the game by comparing normalized names
        all_games = db.query(Game).all()
        game = None
        for g in all_games:
            normalized_db_name = re.sub(r'[^a-zA-Z0-9]', '', g.name).lower()
            if normalized_db_name == normalized_folder_name:
                game = g
                break
        if not game:
            game = Game(name=game_name, image_url=f'/static/images/games/{game_name.lower().replace(" ", "-")}.png')
            db.add(game)
            db.flush() # Flush to get the new game's ID
            import_log[game_name].append(f"Created new game '{game_name}' in the database.")

        # --- Iterate over each product folder within the game directory ---
        for product_folder_name in os.listdir(game_folder_path):
            product_folder_path = os.path.join(game_folder_path, product_folder_name)
            if not os.path.isdir(product_folder_path):
                continue

            json_path = os.path.join(product_folder_path, 'info.json')
            if not os.path.exists(json_path):
                error_log[game_name].append(f"{product_folder_name}: info.json not found.")
                total_skipped_count += 1
                continue

            image_file = next((f for f in os.listdir(product_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
            if not image_file:
                error_log[game_name].append(f"{product_folder_name}: No image file found.")
                total_skipped_count += 1
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            product_name = info.get('title')
            price_str = info.get('price')

            if not product_name or price_str is None:
                error_log[game_name].append(f"{product_folder_name}: 'title' or 'price' missing.")
                total_skipped_count += 1
                continue

            cleaned_price_str = str(price_str).replace('₽', '').replace(' ', '').strip()
            try:
                price = float(cleaned_price_str)
            except ValueError:
                error_log[game_name].append(f"Skipping '{product_folder_name}': Invalid price format '{price_str}'.")
                total_skipped_count += 1
                continue

            # --- Seller Logic: Get the next seller from the shuffled cycle ---
            seller_username = next(seller_cycle)
            seller = db.query(User).filter(User.username == seller_username).first()
            if not seller:
                new_telegram_id = -random.randint(1000000, 9999999)
                seller = User(
                    telegram_id=new_telegram_id,
                    username=seller_username,
                    first_name=seller_username,
                    last_name="",
                    photo_url='/static/images/my-avatar.png'
                )
                db.add(seller)
                db.flush()

            original_image_path = os.path.join(product_folder_path, image_file)
            unique_filename = f"{uuid.uuid4()}{os.path.splitext(image_file)[1]}"
            destination_image_path = os.path.join(UPLOADS_DIR, unique_filename)
            shutil.copy(original_image_path, destination_image_path)

            new_item = Item(
                name=product_name,
                description=f"Аккаунт: {product_name}",
                price=price,
                currency="RUB",
                seller_id=seller.id,
                game_id=game.id,
                image_url=f"/uploads/{unique_filename}",
                is_sold=False
            )
            db.add(new_item)
            import_log[game_name].append(f"{product_name} (Seller: {seller.username})")
            total_imported_count += 1

    db.commit()

    return {
        "status": "success",
        "total_imported": total_imported_count,
        "total_skipped": total_skipped_count,
        "details": import_log,
        "errors": error_log
    }
    return {
        "status": "success",
        "message": "Import complete.",
        "imported_count": len(imported_items),
        "imported_items": imported_items,
        "skipped_count": len(skipped_items),
        "skipped_details": skipped_items
    }



# --- SCHEMAS (Pydantic Models) ---
class UserSchema(BaseModel):
    id: int
    telegram_id: int
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    balance: float
    frozen_balance: float
    photo_url: Optional[str] = None
    bonuses: Optional[int] = None

    class Config:
        from_attributes = True
        extra='ignore'

class UserProfile(BaseModel):
    id: int
    username: str
    photo_url: Optional[str] = None
    deals_as_buyer_count: int
    deals_as_seller_count: int
    balance: float
    frozen_balance: float

    class Config:
        from_attributes = True

class GameResponse(BaseModel):
    id: int
    name: str
    cover_image_url: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class UserResponse(BaseModel):
    telegram_id: int
    username: Optional[str]
    photo_url: Optional[str]
    class Config:
        from_attributes = True

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    currency: str
    seller: UserResponse
    image_url: Optional[str] = None
    is_sold: bool
    game: GameResponse
    model_config = ConfigDict(from_attributes=True)

class ItemCreateRequest(BaseModel):
    name: str
    description: str
    price: float
    game_id: int
    currency: str
    seller_telegram_id: int

class DealCreateRequest(BaseModel):
    item_id: int
    buyer_telegram_id: int

class DealActionRequest(BaseModel):
    telegram_id: int

class ProfileResponse(BaseModel):
    telegram_id: int
    username: Optional[str]
    first_name: Optional[str]
    balance: float
    frozen_balance: float
    total_spent: float = 0.0
    total_earned: float = 0.0
    photo_url: Optional[str]
    bonuses: int
    model_config = ConfigDict(from_attributes=True)

class UserSyncRequest(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None
    photo_url: Optional[str] = None
    initData: str

class CompleteTaskRequest(BaseModel):
    telegram_id: int
    task_id: str

class DealResponse(BaseModel):
    id: int
    item: ItemResponse
    seller: UserResponse
    buyer: UserResponse
    price: float
    status: str
    class Config:
        from_attributes = True

class AddFundsRequest(BaseModel):
    telegram_id: int
    amount: float

class UserUpdate(BaseModel):
    telegram_id: int

# --- ENDPOINTS ---
# --- MOCK FINANCE DATA ---
# In a real app, this would come from a database.
FINANCE_DATA = {
    "RU": {
        "currency": "RUB",
        "banks": {
            "Sberbank": ["4276 1600 0000 1111", "4276 1600 0000 2222"],
            "Tinkoff": ["5536 9100 0000 3333", "5536 9100 0000 4444"],
            "Raiffeisenbank": ["4081 7810 0000 5555", "4081 7810 0000 6666"]
        }
    },
    "UA": {
        "currency": "UAH",
        "banks": {
            "PrivatBank": ["5168 7550 0000 7777", "5168 7550 0000 8888"],
            "Monobank": ["5375 4141 0000 9999", "5375 4141 0000 0000"]
        }
    }
}

RECIPIENT_NAMES = ["Ivan I.", "Anna P.", "Petr S.", "Olga K.", "Dmitry M."]

# --- FINANCE API ENDPOINTS ---

@app.get("/api/finance/info")
def get_finance_info():
    """Returns available countries and their banks."""
    return {country: list(data["banks"].keys()) for country, data in FINANCE_DATA.items()}

@app.post("/api/finance/deposit-details")
async def get_deposit_details(request: Request):
    data = await request.json()
    country = data.get("country")
    bank = data.get("bank")

    if not all([country, bank]):
        raise HTTPException(status_code=400, detail="Country and bank are required.")

    country_data = FINANCE_DATA.get(country)
    if not country_data or bank not in country_data["banks"]:
        raise HTTPException(status_code=404, detail="Invalid country or bank.")

    card_number = random.choice(country_data["banks"][bank])
    recipient_name = random.choice(RECIPIENT_NAMES)
    currency = country_data["currency"]

    return {
        "card_number": card_number,
        "recipient_name": recipient_name,
        "currency": currency
    }

@app.post("/api/finance/withdraw")
async def create_withdrawal_request(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    user_id = data.get("user_id")
    amount = data.get("amount")
    card_number = data.get("card_number") # User's card to withdraw to

    if not all([user_id, amount, card_number]):
        raise HTTPException(status_code=400, detail="User ID, amount, and card number are required.")

    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid amount specified.")

    user = db.query(User).filter(User.telegram_id == str(user_id)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient funds.")

    # In a real app, you would create a withdrawal record and process it.
    # For now, we just confirm the request is valid.
    user.balance -= amount
    user.frozen_balance += amount
    db.commit()

    return {"message": "Withdrawal request created successfully."}


@app.get("/api/user/profile/{user_id}", response_model=UserProfile)
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == str(user_id)).first()
    if not user:
        # If user doesn't exist, create them with a default state
        user = User(
            telegram_id=str(user_id),
            username=f"user_{user_id}",  # Placeholder username
            first_name=f"User {user_id}",
            photo_url='/static/images/my-avatar.png',
            balance=0.0,  # Starting balance
            frozen_balance=0.0
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    deals_as_buyer_count = db.query(Deal).filter(Deal.buyer_id == user.id).count()
    deals_as_seller_count = db.query(Deal).filter(Deal.seller_id == user.id).count()

    return UserProfile(
        id=user.id,
        username=user.username,
        photo_url=user.photo_url,
        deals_as_buyer_count=deals_as_buyer_count,
        deals_as_seller_count=deals_as_seller_count,
        balance=user.balance,
        frozen_balance=user.frozen_balance
    )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

BOT_TOKEN = os.getenv('BOT_TOKEN')
CHANNEL_ID = os.getenv('CHANNEL_ID', '@your_channel_name') # Замените на ID или @имя вашего канала

@app.post("/api/sync_user")
async def sync_user(sync_data: UserSyncRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == sync_data.id).first()
    if not user:
        user = User(
            telegram_id=sync_data.id,
            username=sync_data.username,
            first_name=sync_data.first_name,
            last_name=sync_data.last_name,
            photo_url=sync_data.photo_url
        )
        db.add(user)
    else:
        user.username = sync_data.username
        user.first_name = sync_data.first_name
        user.last_name = sync_data.last_name
        user.photo_url = sync_data.photo_url
    
    db.commit()
    db.refresh(user)
    return get_profile_data(user, db)

@app.get("/api/profile/{telegram_id}")
def get_profile(telegram_id: int, db: Session = Depends(get_db)) -> ProfileResponse:
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return get_profile_data(user, db)

@app.get("/api/tasks/{telegram_id}")
async def get_user_tasks(telegram_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        # A new user might not be in the DB yet, but we should still show them the tasks page.
        # The sync_user endpoint will create them shortly.
        return []

    tasks = []

    # Task 1: Daily Bonus
    completed_today = False
    if user.last_daily_bonus:
        if user.last_daily_bonus.date() == datetime.utcnow().date():
            completed_today = True

    tasks.append({
        "id": "daily_bonus",
        "title": "Ежедневный бонус",
        "description": "Получайте 10 бонусов каждый день.",
        "reward": 10,
        "completed": completed_today
    })

    # Task 2: First Purchase
    tasks.append({
        "id": "first_purchase",
        "title": "Первая покупка",
        "description": "Совершите свою первую покупку, чтобы получить награду.",
        "reward": 50,
        "completed": user.first_purchase_bonus_received
    })

    # Task 3: First Sale
    tasks.append({
        "id": "first_sale",
        "title": "Первая продажа",
        "description": "Совершите свою первую продажу, чтобы получить награду.",
        "reward": 50,
        "completed": user.first_sale_bonus_received
    })

    return tasks



@app.post("/api/tasks/daily_bonus", response_model=ProfileResponse)
async def claim_daily_bonus(request: CompleteTaskRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == request.telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.last_daily_bonus and user.last_daily_bonus.date() == datetime.utcnow().date():
        raise HTTPException(status_code=400, detail="Daily bonus already claimed today.")

    # Grant the bonus
    reward = 10 # Hardcoded reward for daily bonus
    user.bonuses += reward
    user.last_daily_bonus = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return get_profile_data(user, db)

@app.get("/api/games", response_model=List[GameResponse])
def get_games(db: Session = Depends(get_db)):
    return db.query(Game).all()

@app.get("/api/items/{game_id}")
def get_game_items(game_id: int, db: Session = Depends(get_db)):
    items = db.query(Item).options(joinedload(Item.seller), joinedload(Item.game)).filter(Item.game_id == game_id, Item.is_sold == False).order_by(Item.created_at.desc()).all()
    return items

def process_and_save_item(
    name: str,
    description: str,
    price: float,
    game_id: int,
    currency: str,
    seller_telegram_id: int,
    image_content: bytes,
    original_filename: str
):
    """
    Background task to process an image, save it, and create an item in the database.
    """
    db = SessionLocal()
    try:
        seller = db.query(User).filter(User.telegram_id == seller_telegram_id).first()
        if not seller:
            print(f"BACKGROUND TASK ERROR: Seller with telegram_id {seller_telegram_id} not found.")
            return

        # Standardize on JPEG format for all uploaded images
        unique_filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)

        # Process image from in-memory bytes
        with Image.open(io.BytesIO(image_content)) as img:
            img.thumbnail((800, 800))
            # Convert to RGB if it has an alpha channel (like PNGs) to be JPEG-compatible
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(file_path, 'JPEG', optimize=True, quality=85)
        
        image_url_path = f"/uploads/{unique_filename}"

        new_item = Item(
            name=name,
            description=description,
            price=price,
            game_id=game_id,
            currency=currency,
            seller_id=seller.id,
            image_url=image_url_path
        )
        db.add(new_item)
        db.commit()
        print(f"BACKGROUND TASK SUCCESS: Item '{name}' created successfully.")
        # Notify seller via Telegram
        bot_token = os.getenv("BOT_TOKEN")
        if bot_token:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    data={
                        "chat_id": seller_telegram_id,
                        "text": f"Ваш товар '{name}' успешно опубликован в магазине!"
                    }, timeout=5
                )
            except Exception as tel_err:
                print(f"Failed to send Telegram notification: {tel_err}")
    except Exception as e:
        print(f"BACKGROUND TASK FAILED: Could not create item '{name}'. Reason: {e}")
    finally:
        db.close()

@app.post("/api/items", status_code=202)
async def create_item(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    name: str = Form(...),
    description: str = Form(""),
    price: float = Form(...),
    game_id: int = Form(...),
    currency: str = Form(...),
    seller_telegram_id: int = Form(...),
    image: UploadFile = File(...)
):
    if not image:
        raise HTTPException(status_code=400, detail="Image upload is required.")

    seller = db.query(User).filter(User.telegram_id == seller_telegram_id).first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")

    image_content = await image.read()

        # Offload heavy image processing to background task
    background_tasks.add_task(process_and_save_item,
        name,
        description,
        price,
        game_id,
        currency,
        seller_telegram_id,
        image_content,
        image.filename
    )

    return {"message": "Item submitted for review."}

@app.post("/api/deals", status_code=201)
def create_deal(request: DealCreateRequest, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == request.item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.is_sold:
        raise HTTPException(status_code=400, detail="Item is already sold")

    buyer = db.query(User).filter(User.telegram_id == request.buyer_telegram_id).first()
    if not buyer:
        raise HTTPException(status_code=404, detail="Buyer not found")

    # The seller is already linked to the item, no need for a separate query.
    seller = item.seller
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found for this item")
        
    if buyer.id == seller.id:
        raise HTTPException(status_code=400, detail="You cannot buy your own item")

    if buyer.balance < item.price:
        raise HTTPException(status_code=400, detail="Not enough funds")

    # Atomically update balances and create the deal
    buyer.balance -= item.price
    buyer.frozen_balance += item.price
    item.is_sold = True # Mark item as sold

    new_deal = Deal(
        item_id=item.id,
        buyer_id=buyer.id,
        seller_id=seller.id,
        price=item.price,
        status='frozen'
    )
    db.add(new_deal)
    db.commit()
    db.refresh(new_deal)

    # Notify buyer and seller via Telegram bot
    try:
        bot_token = os.getenv("BOT_TOKEN")
        if bot_token:
            # Notify buyer
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={
                    "chat_id": buyer.telegram_id,
                    "text": f"Вы успешно купили товар '{item.name}' за {item.price} {item.currency}."
                }, timeout=5
            )
            # Notify seller
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={
                    "chat_id": seller.telegram_id,
                    "text": f"Ваш товар '{item.name}' был куплен пользователем @{buyer.username or buyer.first_name}."
                }, timeout=5
            )
    except Exception as tel_err:
        print(f"Failed to send Telegram purchase notification: {tel_err}")

    return {"message": "Deal created successfully", "deal_id": new_deal.id}

@app.get("/api/my_sales/{telegram_id}")
async def get_my_sales(telegram_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    active_items = db.query(Item).filter(Item.seller_id == user.id, Item.is_sold == False).order_by(Item.created_at.desc()).all()
    deals = db.query(Deal).options(
        joinedload(Deal.item).options(joinedload(Item.game), joinedload(Item.seller)),
        joinedload(Deal.buyer),
        joinedload(Deal.seller)
    ).filter(Deal.seller_id == user.id).order_by(Deal.created_at.desc()).all()
    return {
        "active_items": [ItemResponse.model_validate(i).model_dump() for i in active_items],
        "deals": [DealResponse.model_validate(d).model_dump() for d in deals]
    }

@app.get("/api/deals/{telegram_id}", response_model=List[DealResponse])
def get_deals(telegram_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Correctly query deals using the internal user.id
    deals = db.query(Deal).options(
        joinedload(Deal.item).options(
            joinedload(Item.game),
            joinedload(Item.seller)
        ),
        joinedload(Deal.buyer),
        joinedload(Deal.seller)
    ).filter(
        or_(Deal.seller_id == user.id, Deal.buyer_id == user.id)
    ).order_by(Deal.created_at.desc()).all()

    # Ensure every deal item has an image URL
    for deal in deals:
        if not deal.item.image_url:
            deal.item.image_url = "/static/images/default-placeholder.png"
            
    return deals

@app.post("/api/deals/{deal_id}/ship", status_code=200)
def ship_deal(deal_id: int, request: DealActionRequest, db: Session = Depends(get_db)):
    deal = db.query(Deal).options(joinedload(Deal.seller)).filter(Deal.id == deal_id).first()
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    if deal.seller.telegram_id != request.telegram_id:
        raise HTTPException(status_code=403, detail="Forbidden: You are not the seller.")
    if deal.status != 'frozen':
        raise HTTPException(status_code=400, detail=f"Cannot ship a deal with status '{deal.status}'")
    
    deal.status = 'shipped'
    db.commit()
    db.refresh(deal)
    return {"message": "Deal has been shipped"}

@app.post("/api/deals/{deal_id}/complete", response_model=DealResponse)
def complete_deal(deal_id: int, user_data: UserUpdate, db: Session = Depends(get_db)):
    deal = db.query(Deal).options(joinedload(Deal.buyer), joinedload(Deal.seller), joinedload(Deal.item)).filter(Deal.id == deal_id).first()
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")

    user = db.query(User).filter(User.telegram_id == user_data.telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if deal.buyer_id != user.id:
        raise HTTPException(status_code=403, detail="Only the buyer can complete the deal")

    if deal.status != 'shipped':
        raise HTTPException(status_code=400, detail=f"Cannot complete a deal with status '{deal.status}'")

    deal.status = 'completed'
    deal.updated_at = datetime.utcnow()

    # --- NEW LOGIC FOR FIRST DEAL BONUS ---
    buyer = deal.buyer
    seller = deal.seller
    deal_bonus = 100

    # Award bonus for the first purchase if applicable
    if not buyer.first_purchase_bonus_received:
        buyer.bonuses += 50
        buyer.first_purchase_bonus_received = True

    # Award bonus for the first sale if applicable
    if not seller.first_sale_bonus_received:
        seller.bonuses += 50
        seller.first_sale_bonus_received = True

    item = deal.item
    item.owner_id = deal.buyer_id

    db.commit()
    db.refresh(deal)
    db.refresh(buyer)
    db.refresh(seller)
    
    return DealResponse.model_validate(deal).model_dump()

def get_profile_data(user: User, db: Session) -> dict:
    # Helper function to structure profile data
    # Calculate total spent and earned
    total_spent = db.query(func.sum(Deal.price)).filter(Deal.buyer_id == user.id, Deal.status == 'completed').scalar() or 0.0
    total_earned = db.query(func.sum(Deal.price)).filter(Deal.seller_id == user.id, Deal.status == 'completed').scalar() or 0.0

    return {
        "telegram_id": user.telegram_id,
        "username": user.username,
        "first_name": user.first_name,
        "balance": user.balance,
        "frozen_balance": user.frozen_balance,
        "total_spent": total_spent,
        "total_earned": total_earned,
        "photo_url": user.photo_url,
        "bonuses": user.bonuses,
        "first_purchase_bonus_received": user.first_purchase_bonus_received,
        "first_sale_bonus_received": user.first_sale_bonus_received
    }

# --- DEV ENDPOINTS (for testing) ---
@app.post("/api/dev/add_funds")
def add_funds(request: AddFundsRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.telegram_id == request.telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.balance += request.amount
    db.commit()
    db.refresh(user)
    
    return {"message": f"Successfully added {request.amount} to user {user.telegram_id}. New balance: {user.balance}"}