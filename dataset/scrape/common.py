from dotenv import load_dotenv
import os

load_dotenv()

DATA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data_files")
KEY = os.getenv("STEAM_WEB_API_KEY")
