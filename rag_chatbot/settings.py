from pydantic import BaseSettings, Field, validator
from typing import Optional, Literal
import os

class AzureOpenAISettings(BaseSettings):
    api_key: str = Field(..., env="AZURE_OPENAI_KEY")
    endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    version: str = Field(..., env="AZURE_OPENAI_VERSION")
    deployment: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT")

class DeepSeekSettings(BaseSettings):
    api_key: str = Field(..., env="DEEPSEEK_API_KEY")
    endpoint: str = Field(..., env="DEEPSEEK_ENDPOINT")

class AppSettings(BaseSettings):
    api_type: Literal["azure", "deepseek"] = Field(..., env="API_TYPE")
    azure: Optional[AzureOpenAISettings] = None
    deepseek: Optional[DeepSeekSettings] = None

    @validator('azure', always=True)
    def validate_azure(cls, v, values):
        if values.get('api_type') == 'azure' and v is None:
            return AzureOpenAISettings()
        return v

    @validator('deepseek', always=True)
    def validate_deepseek(cls, v, values):
        if values.get('api_type') == 'deepseek' and v is None:
            return DeepSeekSettings()
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
