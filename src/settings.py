from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_api_key: str
    qdrant_url: str

    class Config:
        env_file = "../dev.env"

settings = Settings()
