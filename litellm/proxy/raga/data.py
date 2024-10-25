import json
from typing import Any, Dict

import litellm as lm


def get_params(provider_name: str, model_name: str):
    if provider_name == "azure":
        return lm.get_supported_openai_params(model='', custom_llm_provider='azure')

    return lm.get_supported_openai_params(model=f"{provider_name}/{model_name}")


provider_data: Dict[str, Any] = {}
with open("model_prices_and_context_window.json", "r") as f:
    data: dict = json.loads(f.read())
    data.pop("sample_spec")
    for model_name, model_data in data.items():
        provider_name: str = model_data.get("litellm_provider")
        if provider_name is None:
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