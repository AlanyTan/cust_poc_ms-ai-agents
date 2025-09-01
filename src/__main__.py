# Add current directory to sys.path so modules can be imported as if started from src dir

import sys
import os
import uvicorn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .api import main  # noqa: E402


if __name__ == "__main__":
    app = main.create_app()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
