from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from fastapi import APIRouter, HTTPException, Request

import litellm as lm

T = TypeVar("T")


@dataclass
class RagaApiResponse(Generic[T]):
    success: bool
    data: T
    message: Optional[str]


router = APIRouter(prefix="/raga/internal")


@router.get("/providers")
async def get_providers() -> RagaApiResponse:
    return RagaApiResponse(True, {"providers": list(lm.models_by_provider.keys())}, None)


@router.get("/providers/{provider}/models")
async def get_model_by_provider(request: Request) -> RagaApiResponse:
    provider_name = request.path_params.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")

    model_list = lm.models_by_provider.get(provider_name)
    if model_list is None:
        raise HTTPException(status_code=400, detail="provider not supported")
    return RagaApiResponse(True, {"models": model_list}, None)


@router.get("/providers/{provider}/models/{model}/params")
async def get_model_params_by_model(request: Request) -> RagaApiResponse:
    provider_name = request.path_params.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")
    model_name = request.path_params.get("model")
    if model_name is None:
        raise HTTPException(status_code=400, detail="model is required")
    if model_name not in lm.models_by_provider.get(provider_name, []):
        raise HTTPException(status_code=400, detail="model is not valid")

    param_list = lm.get_supported_openai_params(model=f"{provider_name}/{model_name}")
    return RagaApiResponse(True, {"params": param_list}, None)
