from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    whisper_model_size: str = "small"   
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  

    class Config:
        env_file = ".env"

settings = Settings()