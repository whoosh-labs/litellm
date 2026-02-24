import hashlib
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

AES_KEY = os.getenv("AES_ENCRYPTION_KEY")

# Cache for Vertex AI credentials to avoid redundant temp file creation
# and to detect credential changes for singleton reset
_vertex_creds_hash = None
_vertex_creds_temp_file = None


def modify_user_request(data):
    try:
        if "provider" in data:
            data["model"] = data["provider"] + "/" + data["model"]
            del data["provider"]
        if "encrypted_secrets_map" in data:
            set_api_keys(data)
            data.pop("user_id", None)
        return data
    except Exception as e:
        print(f"exception in getting api keys: {str(e)}")
        traceback.print_exc()
        raise e


def set_api_keys(data):
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


def _get_or_update_vertex_creds_file(vertex_creds: str) -> str:
    """
    Returns the path to a temp file containing the Vertex AI credentials JSON.
    If the credentials haven't changed since the last call, reuses the existing temp file.
    If they have changed, writes a new temp file, cleans up the old one, and resets
    the VertexLLM singleton so it reloads credentials on the next request.
    """
    global _vertex_creds_hash, _vertex_creds_temp_file

    new_hash = hashlib.sha256(vertex_creds.encode()).hexdigest()

    if new_hash == _vertex_creds_hash and _vertex_creds_temp_file and os.path.exists(_vertex_creds_temp_file):
        return _vertex_creds_temp_file

    # Credentials changed — clean up old temp file
    if _vertex_creds_temp_file and os.path.exists(_vertex_creds_temp_file):
        try:
            os.unlink(_vertex_creds_temp_file)
        except OSError:
            pass

    # Write new temp file
    credentials = json.loads(vertex_creds)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(credentials, f)
        _vertex_creds_temp_file = f.name

    _vertex_creds_hash = new_hash

    # Reset the VertexLLM singleton so it picks up the new credentials
    # Only vertex_chat_completion caches credentials on the instance;
    # vertex_partner_models and vertex_model_garden create new VertexLLM() per call.
    try:
        from litellm.main import vertex_chat_completion
        vertex_chat_completion.reset_credentials()
        print("Vertex AI credentials changed — reset VertexLLM singleton")
    except Exception as e:
        print(f"Warning: could not reset VertexLLM singleton: {e}")

    return _vertex_creds_temp_file


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
            data["vertex_credentials"] = _get_or_update_vertex_creds_file(vertex_creds)

        data["vertex_project"] = vault_secrets.get(VERTEXAI_PROJECT)
        data["vertex_location"] = vault_secrets.get(VERTEXAI_LOCATION)
        data["api_key"] = "dummy-vertex"


def validate_api_keys(secrets, model_name, required_keys):
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
