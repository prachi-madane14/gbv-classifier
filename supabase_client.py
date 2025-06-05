from supabase import create_client  # type: ignore
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
