from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(), override=True)


class Settings(BaseSettings):
    """Settings for the FastAPI application."""

    model_config = {
        "env_file": find_dotenv(),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }
    CONFIG_DIR: str = "../../config"
    OPENAI_API_KEY: str
    NEO4J_PASSWORD: str
    NEO4J_USER: str
    NEO4J_URI: str


SETTINGS = Settings()
