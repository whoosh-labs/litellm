"""
Simple Vertex AI Model Garden Handler for LiteLLM Proxy
"""

import json
import os
import requests
import tempfile
from typing import Optional, Tuple
from google.auth import default
import google.auth.transport.requests
from datetime import datetime, timedelta
from litellm._logging import verbose_logger


class VertexModelGardenHandler:
    """Simple handler for Vertex AI Model Garden endpoints"""
    
    def __init__(self):
        self._cache = {}
        self._cache_expiry = {}
    
    def get_oauth_token(self, credentials_json: Optional[str] = None) -> str:
        """Get OAuth token"""
        if credentials_json:
            creds_dict = json.loads(credentials_json)
            
            # Use proper temp file handling
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(creds_dict, f)
                temp_file = f.name
            
            old_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file
            
            try:
                credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                credentials.refresh(google.auth.transport.requests.Request())
                return credentials.token
            finally:
                # Always cleanup
                os.unlink(temp_file)
                if old_env:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_env
                else:
                    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        else:
            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
            return credentials.token
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid (5 minutes TTL)"""
        if cache_key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[cache_key]
    
    def find_deployment_by_name(self, deployment_name: str, project_id: str, location: str, credentials_json: Optional[str] = None) -> Tuple[str, str]:
        """Find endpoint ID and dedicated host by deployment name"""
        cache_key = f"{project_id}:{location}"
        
        # Check cache first
        if cache_key not in self._cache or not self._is_cache_valid(cache_key):
            try:
                token = self.get_oauth_token(credentials_json)
                
                # List all endpoints
                url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints"
                headers = {"Authorization": f"Bearer {token}"}
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                endpoints = response.json().get('endpoints', [])
                
                # Cache the results
                self._cache[cache_key] = {}
                for endpoint in endpoints:
                    display_name = endpoint.get('displayName', '').lower()
                    endpoint_id = endpoint.get('name', '').split('/')[-1]
                    dedicated_host = endpoint.get('dedicatedEndpointDns', '')
                    
                    if display_name and dedicated_host:
                        self._cache[cache_key][display_name] = {
                            'endpoint_id': endpoint_id,
                            'dedicated_host': dedicated_host
                        }
                
                # Set cache expiry (5 minutes)
                self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to list endpoints: {str(e)}")
            except Exception as e:
                raise Exception(f"Error accessing Vertex AI: {str(e)}")
        
        # Find by deployment name
        deployment_name_lower = deployment_name.lower()
        if deployment_name_lower in self._cache[cache_key]:
            info = self._cache[cache_key][deployment_name_lower]
            return info['endpoint_id'], info['dedicated_host']
        
        # If not found, list available deployments
        available = list(self._cache[cache_key].keys())
        raise ValueError(f"Deployment '{deployment_name}' not found. Available: {available}")
    
    def transform_request(self, data: dict) -> dict:
        """Transform vertex_ai/openai/* request to Model Garden format"""
        model_name = data.get("model", "")
        
        # Extract deployment name
        deployment_name = model_name.replace("vertex_ai/openai/", "")
        
        # Get required parameters
        project_id = data.get("vertex_project")
        location = data.get("vertex_location", "us-central1")
        credentials = data.get("vertex_credentials")
        
        # Handle credentials file path
        if credentials and credentials.startswith("/") and os.path.exists(credentials):
            with open(credentials, 'r') as f:
                credentials = f.read()
        
        if not project_id:
            raise ValueError("vertex_project is required")
        
        try:
            # Find endpoint ID and dedicated host by deployment name
            endpoint_id, dedicated_host = self.find_deployment_by_name(
                deployment_name, project_id, location, credentials
            )
            
            # Build URL
            api_base = f"https://{dedicated_host}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
            
            # Get OAuth token
            oauth_token = self.get_oauth_token(credentials)
            
            # Transform the request
            data["model"] = "openai/" + deployment_name
            data["api_base"] = api_base
            data["api_key"] = oauth_token
            data["max_tokens"] = data.get("max_tokens", 100)
            
            verbose_logger.debug(f"api_base: {api_base}")
            verbose_logger.debug(f"deployment_name: {deployment_name}")
            verbose_logger.debug(f"project_id: {project_id}")
            verbose_logger.debug(f"location: {location}")
            verbose_logger.debug(f"credentials: {credentials}")
            
            # Clean up
            data.pop("vertex_project", None)
            data.pop("vertex_location", None)
            data.pop("vertex_credentials", None)
        except Exception as e:
            print(f"Error transforming Model Garden request: {e}")
            raise


# Global instance
handler = VertexModelGardenHandler()


