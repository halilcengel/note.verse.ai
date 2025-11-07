from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = "../dev.env"

settings = Settings()
