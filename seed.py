from sqlalchemy.orm import Session
from database import SessionLocal, engine, init_db, User, Item, Game, Deal

# --- Cover Images Mapping ---
# Make sure these files exist in /web/static/images/catalog/
GAME_COVERS = {
    "Counter-Strike 2": "/static/images/catalog/cs2.jpg",
    "Dota 2": "/static/images/catalog/dota2.jpg",
    "Apex Legends": "/static/images/catalog/apex.jpg",
    "Call of Duty: Mobile": "/static/images/catalog/cod_mobile.jpg",
    "Valorant": "/static/images/catalog/valorant.jpg",
    "Telegram Premium": "/static/images/catalog/telegram.jpg",
    "Spotify": "/static/images/catalog/spotify.jpg",
    "Discord": "/static/images/catalog/discord.jpg",
    "Black Russia": "/static/images/catalog/blackrussia.jpg",
    "Fortnite": "/static/images/catalog/fortnite.jpg",
    "Overwatch 2": "/static/images/catalog/overwatch.jpg",
    "Sea of Thieves": "/static/images/catalog/sot.jpg",
    "Genshin Impact": "/static/images/catalog/genshin.jpg",
    "Team Fortress 2": "/static/images/catalog/tf2.jpg",
    "Rust": "/static/images/catalog/rust.jpg",
    "World of Warcraft": "/static/images/catalog/wow.jpg",
    "Escape from Tarkov": "/static/images/catalog/tarkov.jpg"
}

def seed_database(db: Session):
    """
    Seeds the database with initial data.
    - Deletes all existing data from tables.
    - Creates Game entries from the GAME_COVERS dictionary.
    """
    print("Clearing all existing data from the database...")
    # Clear data in reverse order of dependencies to avoid foreign key constraints
    db.query(Item).delete()
    db.query(Deal).delete()
    db.query(User).delete()
    db.query(Game).delete()
    db.commit()
    print("All tables have been cleared.")

    print("Seeding database with games...")
    # Create Game objects
    for game_name, cover_url in GAME_COVERS.items():
        new_game = Game(name=game_name, cover_image_url=cover_url)
        db.add(new_game)
    
    db.commit()
    print(f"Successfully created {len(GAME_COVERS)} game entries.")
    print("Database seeding complete.")

if __name__ == "__main__":
    print("Initializing DB for seeding...")
    init_db()  # Ensure tables are created
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
