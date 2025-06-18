"""
Utility functions for WANDS Product Search
Centralized helper functions for vector search, MLflow, and data processing operations.
"""

import time
import pandas as pd
from typing import Dict, Any, Optional
from databricks.vector_search.client import VectorSearchClient
from mlflow.tracking import MlflowClient


def endpoint_exists(vsc: VectorSearchClient, endpoint_name: str) -> bool:
    """
    Check if a Vector Search endpoint exists
    
    Args:
        vsc: Vector Search client
        endpoint_name: Name of the endpoint to check
        
    Returns:
        bool: True if endpoint exists, False otherwise
    """
    try:
        vsc.get_endpoint(endpoint_name)
        return True
    except Exception as e:
        # Check the error message instead of relying on specific exception class
        if "ENDPOINT_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
            return False
        # For other errors, log and re-raise
        print(f"Error checking if endpoint exists: {e}")
        raise e


def wait_for_vs_endpoint_to_be_ready(vsc: VectorSearchClient, vs_endpoint_name: str, timeout_seconds: int = 1800) -> Optional[Dict]:
    """
    Wait for a Vector Search endpoint to be ready
    
    Args:
        vsc: Vector Search client
        vs_endpoint_name: Name of the endpoint
        timeout_seconds: Timeout in seconds
        
    Returns:
        dict: Endpoint information or None if REQUEST_LIMIT_EXCEEDED
    """
    for i in range(timeout_seconds // 10):
        try:
            endpoint = vsc.get_endpoint(vs_endpoint_name)
            status = endpoint.get("endpoint_status", endpoint.get("status", {}))["state"].upper()
            
            if "ONLINE" in status:
                print(f"Endpoint {vs_endpoint_name} is ready")
                return endpoint
            elif "PROVISIONING" in status or i < 6:
                if i % 20 == 0:
                    print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
                time.sleep(10)
            else:
                raise Exception(f"Error with endpoint {vs_endpoint_name}: {endpoint}")
        except Exception as e:
            # Handle potential REQUEST_LIMIT_EXCEEDED issue
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
                return None
            else:
                raise e
                
    raise Exception(f"Timeout waiting for endpoint {vs_endpoint_name} to be ready")


def index_exists(vsc: VectorSearchClient, endpoint_name: str, index_name: str) -> bool:
    """
    Check if a Vector Search index exists
    
    Args:
        vsc: Vector Search client
        endpoint_name: Name of the endpoint
        index_name: Name of the index
        
    Returns:
        bool: True if index exists, False otherwise
    """
    try:
        vsc.get_index(endpoint_name, index_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


def wait_for_index_to_be_ready(vsc: VectorSearchClient, vs_endpoint_name: str, index_name: str, timeout_seconds: int = 1800) -> None:
    """
    Wait for a Vector Search index to be ready
    
    Args:
        vsc: Vector Search client
        vs_endpoint_name: Name of the endpoint
        index_name: Name of the index
        timeout_seconds: Timeout in seconds
        
    Returns:
        None
    """
    for i in range(timeout_seconds // 10):
        try:
            idx = vsc.get_index(vs_endpoint_name, index_name).describe()
            index_status = idx.get('status', idx.get('index_status', {}))
            status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
            url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
            
            if "ONLINE" in status:
                return
            if "UNKNOWN" in status:
                print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
                return
            elif "PROVISIONING" in status:
                if i % 40 == 0: 
                    print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
                time.sleep(10)
            else:
                raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
        except Exception as e:
            # Handle potential REQUEST_LIMIT_EXCEEDED issue
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                print("WARN: couldn't get index status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your index status")
                return
            else:
                raise e
                
    raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(vs_endpoint_name, index_name)}")


def get_vs_results_df(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Helper function to return results of a vector search as a dataframe
    
    Args:
        results: Vector search results dictionary
        
    Returns:
        pd.DataFrame: Results formatted as DataFrame
        
    Raises:
        KeyError: If results format is invalid
        Exception: If DataFrame creation fails
    """
    try:
        result_columns = [col['name'] for col in results['manifest']['columns']]
    except KeyError as e:
        raise KeyError(f"KeyError: {e}. 'manifest' or 'columns' key not found in results.") from e

    try:
        result_data = results['result']['data_array']
    except KeyError as e:
        raise KeyError(f"KeyError: {e}. 'result' or 'data_array' key not found in results.") from e

    try:
        result_df = pd.DataFrame(result_data, columns=result_columns)
    except Exception as e:
        raise Exception(f"Error creating DataFrame: {e}") from e

    return result_df


def get_workspace_url() -> str:
    """
    Extract workspace URL from Spark configuration.
    
    Returns:
        str: Full workspace URL with https prefix
    """
    # Import spark here to avoid issues if not available
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active Spark session found")
    
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    return f"https://{workspace_url}"


def get_latest_model_version(model_name: str):
    """
    Get the latest version of a registered MLflow model
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        ModelVersion: Latest model version object
        
    Raises:
        ValueError: If no versions found for the model
    """
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    
    latest_version = max(model_versions, key=lambda mv: int(mv.version))
    return latest_version


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    Load and validate YAML configuration
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Loaded configuration
        
    Raises:
        ValueError: If required configuration sections are missing
    """
    import yaml
    
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # Validate required sections
    required_sections = ['unity_catalog', 'base_data', 'search_configs', 'auth_creds']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config





def create_vector_search_client(config: Dict[str, Any]) -> VectorSearchClient:
    """
    Create Vector Search client using automatic authentication
    
    Args:
        config: Configuration dictionary (not used for automatic auth)
        
    Returns:
        VectorSearchClient: Client with automatic authentication
    """
    # Use automatic authentication in notebook environment
    return VectorSearchClient()


print("✅ Utils module loaded successfully") 