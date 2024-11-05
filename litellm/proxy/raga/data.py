import json
from typing import Any, Dict

import yaml

import litellm as lm


def get_params(provider_name: str, model_name: str):
    if provider_name == "azure":
        return lm.get_supported_openai_params(model="", custom_llm_provider="azure")

    return lm.get_supported_openai_params(model=f"{provider_name}/{model_name}")


provider_keys_dict = {}
with open("litellm/proxy/raga/model_config.yaml", "r") as f:
    model_config = yaml.safe_load(f)
    for item in model_config.get("model_list"):
        provider_config = item.get("litellm_params")
        provider_name = provider_config.pop("model").split("/")[0]
        provider_keys_dict[provider_name] = list(map(lambda x: x.split("/")[1], provider_config.values()))


raw_data = {}
provider_data: Dict[str, Any] = {}
with open("model_prices_and_context_window.json", "r") as f:
    raw_data: dict = json.loads(f.read())
    raw_data.pop("sample_spec", None)
    for model_name, model_data in raw_data.items():
        if model_data.pop("mode", "chat") != "chat":
            continue

        provider_name: str = model_data.pop("litellm_provider", None)
        if provider_name is None or provider_name not in list(provider_keys_dict.keys()):
            continue

        if provider_name not in provider_data:
            provider_data[provider_name] = {
                "keys": provider_keys_dict.get(provider_name),
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
