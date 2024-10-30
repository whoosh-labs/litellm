import json
from functools import cache
from typing import Any, Dict

import litellm as lm


def get_params(provider_name: str, model_name: str):
    if provider_name == "azure":
        return lm.get_supported_openai_params(model="", custom_llm_provider="azure")

    return lm.get_supported_openai_params(model=f"{provider_name}/{model_name}")


@cache
def get_supported_providers():
    import yaml

    with open("litellm/proxy/raga/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    providers = []
    for item in config.get("model_list"):
        providers.append(item.get("litellm_params").get("model").split("/")[0])
    return providers


raw_data = {}
provider_data: Dict[str, Any] = {}
with open("model_prices_and_context_window.json", "r") as f:
    raw_data: dict = json.loads(f.read())
    raw_data.pop("sample_spec", None)
    for model_name, model_data in raw_data.items():
        if model_data.pop("mode", "chat") != "chat":
            continue

        provider_name: str = model_data.pop("litellm_provider", None)
        if provider_name is None or provider_name not in get_supported_providers():
            continue

        if provider_name not in provider_data:
            provider_data[provider_name] = {
                "keys": lm.validate_environment(f"{provider_name}/").get("missing_keys"),
                "models": {},
            }
        provider_data[provider_name]["models"][model_name] = get_params(provider_name, model_name)

# {
#     "azure": {
#         "keys": [],
#         "models": {
#             "gpt-35-turbo-instruct-0914": [],
#             "gpt-35-turbo-16k": []
#         }
#     },
#     "openai": {
#         "keys": [],
#         "models": {
#             "gpt-4": []
#         }
#     }
# }
