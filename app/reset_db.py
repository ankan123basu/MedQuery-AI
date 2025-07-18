import os
import shutil
from pathlib import Path

def reset_database():
    """Reset the ChromaDB database by removing all files."""
    db_path = Path("d:/MedQueryAI/app/db")
    
    # Remove all files in the db directory
    for item in db_path.glob('*'):
        if item.is_file():
            item.unlink()
            print(f"Deleted file: {item}")
        elif item.is_dir():
            shutil.rmtree(item)
            print(f"Deleted directory: {item}")
    
    # Recreate the directory structure
    db_path.mkdir(exist_ok=True)
    print("Database has been reset successfully!")

if __name__ == "__main__":
    reset_database()
