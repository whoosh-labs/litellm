import traceback

from fastapi import HTTPException
import warnings
import os
import json
import tempfile
from base64 import b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# Suppress serialization warning for vertex_ai
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Pydantic serializer warnings"
)

API_KEY = "api_key"
API_BASE = "api_base"
API_VERSION = "api_version"

AZURE_API_KEY = "AZURE_API_KEY"
AZURE_API_BASE = "AZURE_API_BASE"
AZURE_API_VERSION = "AZURE_API_VERSION"

AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_REGION_NAME = "AWS_REGION_NAME"
OLLAMA_API_BASE = "OLLAMA_API_BASE"

# VERTEX_AI
VERTEXAI_CREDENTIALS = "VERTEXAI_CREDENTIALS"
VERTEXAI_PROJECT = "VERTEXAI_PROJECT"
VERTEXAI_LOCATION = "VERTEXAI_LOCATION"

AES_KEY = "c7e415394750cde6"  # os.getenv("AES_ENCRYPTION_KEY")


def modify_user_request(data):
    try:
        if "provider" in data:
            data["model"] = data["provider"] + "/" + data["model"]
            del data["provider"]
        if "encrypted_secrets_map" in data:
            set_api_keys_from_vault(data)
            del data["user_id"]
        return data
    except Exception as e:
        print(f"exception in getting api keys: {str(e)}")
        traceback.print_exc()
        raise e


def set_api_keys_from_vault(data):
    print(f"getting api keys for user: {data['user_id']}")

    secrets = decrypt_secrets_map(data["encrypted_secrets_map"], AES_KEY)

    model_name = data["model"]
    if model_name.startswith("azure"):
        validate_api_keys(secrets, model_name, [AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION])
        data[API_KEY] = secrets.get(AZURE_API_KEY)
        data[API_BASE] = secrets.get(AZURE_API_BASE)
        data[API_VERSION] = secrets.get(AZURE_API_VERSION)
    elif model_name.startswith("bedrock"):
        validate_api_keys(secrets, model_name, [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME])
        data["aws_access_key_id"] = secrets.get(AWS_ACCESS_KEY_ID)
        data["aws_secret_access_key"] = secrets.get(AWS_SECRET_ACCESS_KEY)
        data["aws_region_name"] = secrets.get(AWS_REGION_NAME)
    elif model_name.startswith("ollama"):
        validate_api_keys(secrets, model_name, [OLLAMA_API_BASE])
        data[API_BASE] = secrets.get(OLLAMA_API_BASE)
    elif model_name.startswith("vertex_ai"):
        handle_vertex_ai_model(data, secrets, model_name)
    else:
        from litellm.proxy.raga.data import get_model_keys

        keys = get_model_keys(model_name)
        print(f"keys: {keys}")
        if len(keys) == 1:
            validate_api_keys(secrets, model_name, keys)
            data[API_KEY] = secrets.get(keys[0])
        else:
            raise Exception(f"Model {model_name} is not supported")

    del data['encrypted_secrets_map']


def handle_vertex_ai_model(data, vault_secrets, model_name):
    """Handle Vertex AI model configuration"""
    vertex_creds = vault_secrets.get(VERTEXAI_CREDENTIALS)
    if "messages" in data:
        for message in data["messages"]:
            if message.get("name") is None:
                message.pop("name", None)
            if message.get("function_call") is None:
                message.pop("function_call", None)

    if "vertex_ai/openai/" in model_name:
        # Model Garden endpoint
        if vertex_creds and vertex_creds.strip():
            validate_api_keys(vault_secrets, model_name, [VERTEXAI_CREDENTIALS])

            # Set vertex parameters
            data["vertex_credentials"] = vertex_creds

        data["vertex_project"] = vault_secrets.get(VERTEXAI_PROJECT)
        data["vertex_location"] = vault_secrets.get(VERTEXAI_LOCATION)
        # Transform using simple handler
        from litellm.proxy.raga.vertex_model_garden_handler import VertexModelGardenHandler
        handler = VertexModelGardenHandler()
        handler.transform_request(data)
    else:

        # Standard Vertex AI model
        if vertex_creds and vertex_creds.strip():
            validate_api_keys(vault_secrets, model_name, [VERTEXAI_CREDENTIALS])
            credentials = json.loads(vertex_creds)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(credentials, f)
                temp_file_path = f.name
            data["vertex_credentials"] = temp_file_path

        data["vertex_project"] = vault_secrets.get(VERTEXAI_PROJECT)
        data["vertex_location"] = vault_secrets.get(VERTEXAI_LOCATION)
        data["api_key"] = "dummy-vertex"


def validate_api_keys(secrets, model_name, required_keys):
    print(f"secrets: {secrets}")
    print(f"req secrets: {required_keys}")
    not_set_keys = []
    for key in required_keys:
        if secrets.get(key, "") == "":
            not_set_keys.append(key)

    if len(not_set_keys) > 0:
        raise HTTPException(status_code=401, detail=f"Required API Keys are not set for {model_name}: {not_set_keys}")


def decrypt_secrets_map(secrets_map: dict, encryption_key: str) -> dict:
    key = encryption_key.encode()
    cipher = AES.new(key, AES.MODE_ECB)

    decrypted_map = {}

    for k, v in secrets_map.items():
        if v is None:
            decrypted_map[k] = None
            continue

        decrypted_bytes = cipher.decrypt(b64decode(v))
        decrypted_value = unpad(decrypted_bytes, AES.block_size).decode()

        decrypted_map[k] = decrypted_value

    return decrypted_map
