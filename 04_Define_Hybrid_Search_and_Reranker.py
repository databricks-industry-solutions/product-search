# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our basic vector search implemented, we can now explore two additional search capabilities available in Databricks Vector Search: **Hybrid Search** and **Reranking**. These are separate but complementary techniques that can be used together to improve search relevance.
# MAGIC
# MAGIC **Hybrid Search** combines semantic vector search with traditional keyword search (BM25) to leverage the strengths of both approaches:
# MAGIC * Vector search excels at semantic similarity and handling synonyms
# MAGIC * Keyword search excels at exact matches, names, and specific terms
# MAGIC
# MAGIC **Reranking** is a separate post-processing step that reorders search results using additional relevance signals:
# MAGIC * Takes initial search results and applies more sophisticated scoring
# MAGIC * Uses product descriptions as additional context for relevance scoring
# MAGIC * Computationally more expensive but provides better ranking accuracy
# MAGIC
# MAGIC **Key Configuration Differences:**
# MAGIC * `query_type: "HYBRID"` - Enables both vector and keyword search
# MAGIC * `columns_to_rerank: ["embedding_column"]` - Enables reranking using product descriptions

# COMMAND ----------
%md
# Enabling Private Preview for Reranker feature as of June 2025
If you want to enroll additional workspaces to this preview, please reach out to your account team. 


![Reranker PrPr preview.png](images/Reranker%20PrPr%20preview.png "Reranker PrPr preview.png")

Once your workspace is enrolled, a workspace admin can toggle on/off this preview feature from the Preview UI page.

![Reranker PrPr.png](images/Reranker%20PrPr.png "Reranker PrPr.png")

Toggle on the "Vector Search Reranker" Private Preview. After a few minutes, you should be able to query your Vector Search indices with our Reranker!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install & load libraries

# COMMAND ----------

# Install the vectorsearch package to enable vector search
%pip install -qU databricks-vectorsearch
%pip install -qU databricks-sdk
%pip install -qU tenacity
dbutils.library.restartPython()




# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import mlflow
import pandas as pd
from mlflow.models import infer_signature

# COMMAND ----------

# MAGIC %md ##Step 1: Setup Hybrid Search and Reranking Configuration
# MAGIC
# MAGIC In this step, we configure our search to use both hybrid search and reranking capabilities. Hybrid search combines semantic similarity with keyword matching using `query_type: "HYBRID"`. Reranking is enabled by adding `columns_to_rerank` which specifies which text fields to use for additional relevance scoring.

# COMMAND ----------
%run ./00_Setup

# COMMAND ----------
print("✅ Configuration loaded from 00_Setup")
print(f"Using catalog: {config['catalog_name']}, schema: {config['schema_name']}")

# Initialize Vector Search client and import helper functions
from utils import (
    create_vector_search_client,
    index_exists,
    wait_for_index_to_be_ready
)

vsc = create_vector_search_client(config)
print("✅ Vector Search client initialized")

# COMMAND ----------

# Variables - using simplified config structure
VS_ENDPOINT = config['vs_endpoint_name']
VS_INDEX = config['vs_index_basic']  # Reuses the same index as basic search
VS_COLS_TO_RETURN = config['columns_to_return']
VS_COLS_TO_RERANK = [config['embedding_column']]  # Rerank using product descriptions
VS_NUM_RESULTS = config['num_results']
EMBEDDING_ENDPOINT = config['embedding_endpoint']
INDEX_SOURCE_TABLE = config['products_table']
MODEL_NAME = config['hybrid_reranker_model_name']
MODEL_SERVING_ENDPOINT_NAME = config['hybrid_reranker_endpoint_name']

# Print values
print(f"VS_ENDPOINT: {VS_ENDPOINT}")
print(f"VS_INDEX: {VS_INDEX}")
print(f"VS_COLS_TO_RETURN: {VS_COLS_TO_RETURN}")
print(f"VS_COLS_TO_RERANK: {VS_COLS_TO_RERANK}")
print(f"VS_NUM_RESULTS: {VS_NUM_RESULTS}")
print(f"EMBEDDING_ENDPOINT: {EMBEDDING_ENDPOINT}")
print(f"INDEX_SOURCE_TABLE: {INDEX_SOURCE_TABLE}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"MODEL_SERVING_ENDPOINT_NAME: {MODEL_SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md From the configuration above, notice the key differences:
# MAGIC * `query_type: "HYBRID"` enables hybrid search (vector + keyword search)
# MAGIC * `columns_to_rerank: ["embedding_column"]` enables reranking using product descriptions
# MAGIC * The same vector search index is used, but with different search and ranking methodology

# COMMAND ----------

# MAGIC %md ##Step 2: Test Hybrid Search with Reranking
# MAGIC
# MAGIC Let's test our hybrid search combined with reranking. Hybrid search retrieves results using both vector similarity and keyword matching (BM25). Then reranking reorders these results using product descriptions as additional context for more accurate relevance scoring.

# COMMAND ----------

# DBTITLE 1,Setup Helper Functions for Hybrid Search
# Helper functions 

# Use automatic authentication in notebook environment

# 1. Search Vector Search index based on query string
def query_vs_index(endpoint_name, index_name, query_text, columns_to_return=None, query_type='ANN', num_results=5, columns_to_rerank=None):
    from databricks.vector_search.client import VectorSearchClient
    from pyspark.sql.utils import AnalysisException
    vsc = VectorSearchClient()
    index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)

    if not columns_to_return: # If columns to return array isn't passed in, use all columns except embedding column from the index's source table
        # Get columns of source table
        try:
            index_source_data_table = index.describe()['delta_sync_index_spec']['source_table']
            index_columns = spark.table(index_source_data_table)
            columns_to_return = [column for column in index_columns.columns if column not in ['embedding', 'embeddings', 'embedding_vector', 'text_embedding', 'text_embeddings', 'embedding_column']]

        except AnalysisException as e:
            raise AnalysisException(f"Failed to read table {index_source_data_table}: {e}")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}. 'delta_sync_index_spec' or 'source_table' key not found in index description.")
    
    if not columns_to_rerank:
        # Query index
        results = index.similarity_search(
            query_text=query_text,
            columns=columns_to_return,
            query_type=query_type,
            num_results=num_results
        )
    else:
        results = index.similarity_search(
            query_text=query_text,
            columns=columns_to_return,
            query_type=query_type,
            num_results=num_results,
            columns_to_rerank=columns_to_rerank
        )
    return results


# 2. Import helper function from utils
from utils import get_vs_results_df

# COMMAND ----------

# DBTITLE 1,Test Hybrid Search with Reranking
# Test with WANDS-relevant product search queries
query_text = 'modern office chair'

# Query index with hybrid search and reranking
results = query_vs_index(
    endpoint_name=VS_ENDPOINT, 
    index_name=VS_INDEX, 
    query_text=query_text, 
    columns_to_return=VS_COLS_TO_RETURN, 
    columns_to_rerank=VS_COLS_TO_RERANK,
    query_type='HYBRID', 
    num_results=VS_NUM_RESULTS)

# Show results as a dataframe
print(f"Hybrid search + reranking results for query: '{query_text}'")
get_vs_results_df(results).display()

# COMMAND ----------

# MAGIC %md ##Step 3: Package Hybrid Search + Reranking Model for Deployment
# MAGIC
# MAGIC Now we'll package our hybrid search and reranking functionality into an MLflow model. This model combines both capabilities: hybrid search (vector + keyword) for retrieval and reranking for improved result ordering. Our VectorSearchWrapper class includes parameters for both hybrid search and reranking.

# COMMAND ----------

# Pre-fetch index columns - these cannot be fetched dynamically within the model class, as this causes errors with serializing the model due to Spark session dependency (spark.table() call)
VS_COLS_TO_RETURN = config['columns_to_return']
print("Columns to return from vector search:")
print(VS_COLS_TO_RETURN)

# COMMAND ----------

# DBTITLE 1,Define Hybrid Search Model Class
# Use automatic authentication - no credentials needed

import mlflow.pyfunc
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from typing import List, Dict, Any
import pandas as pd
import os
from tenacity import Retrying, stop_after_attempt, wait_exponential


class VectorSearchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, vs_endpoint_name: str, vs_index_name: str, query_type: str = 'HYBRID', num_results: int = 5, vs_cols_to_rerank: List[str] = None):
        self.vs_endpoint_name = vs_endpoint_name
        self.vs_index_name = vs_index_name
        self.num_results = num_results
        self.query_type = query_type
        self.vs_cols_to_return = VS_COLS_TO_RETURN
        self.vs_cols_to_rerank = vs_cols_to_rerank
        self.index = None
    
    def load_context(self, context):
        """One-time initialization during endpoint creation"""
        # Use automatic authentication in notebook environment
        self.vs_client = VectorSearchClient()
        
        # Get index reference
        self.index = self.vs_client.get_index(
            endpoint_name=self.vs_endpoint_name,
            index_name=self.vs_index_name
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Batch query execution"""
        results = []
        
        # Process batch queries
        for query in model_input["query"]:
            # Query index
            search_result = self._perform_search(query_text=query)
            results.extend(self._format_results(search_result))
                
        return pd.DataFrame(results)
    
    def _perform_search(self, query_text: str) -> Dict:
        """Perform vector search with retry logic."""
        for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True):
            with attempt:
                if self.vs_cols_to_rerank:
                    return self.index.similarity_search(
                        query_text=query_text,
                        columns=self.vs_cols_to_return,
                        columns_to_rerank=self.vs_cols_to_rerank,
                        query_type=self.query_type,
                        num_results=self.num_results,
                        disable_notice=True
                    )
                else: # cols_to_rerank is not provided
                    return self.index.similarity_search(
                        query_text=query_text,
                        columns=self.vs_cols_to_return,
                        query_type=self.query_type,
                        num_results=self.num_results,
                        disable_notice=True
                    )

    def _format_results(self, raw_response: Dict) -> List[Dict]:
        """Optimized result formatting - return all results after reranking"""
        all_results = [
            dict(zip(
                [col['name'] for col in raw_response['manifest']['columns']],
                row
            ))
            for row in raw_response.get("result", {}).get("data_array", [])
        ]
        # Return all results (already limited to 5 by num_results)
        return all_results

# COMMAND ----------

# DBTITLE 1,Test Hybrid Search Model
# Sample data for signature inference with WANDS-relevant queries
sample_input = pd.DataFrame({"query": ["modern office chair"]})
    
# Initialize model with test configuration
test_model = VectorSearchWrapper(
    vs_endpoint_name=VS_ENDPOINT,
    vs_index_name=VS_INDEX,
    query_type='HYBRID',
    vs_cols_to_rerank=VS_COLS_TO_RERANK,
    num_results=VS_NUM_RESULTS
)
  
test_model.load_context(None)
sample_output = test_model.predict(None, sample_input)

print("Sample Input:")
display(sample_input)
print("\nSample Output:")
display(sample_output)

# COMMAND ----------

# MAGIC %md The results above demonstrate hybrid search combined with reranking. We still get 5 results like basic search, but with a two-stage process: first hybrid search (vector + keyword) retrieves candidates, then reranking reorders them using product descriptions for improved relevance.

# COMMAND ----------

# DBTITLE 1,Infer Model Signature
# Infer signature
auto_signature = mlflow.models.infer_signature(sample_input, sample_output)
print("Model signature:")
print(auto_signature)

# COMMAND ----------

# DBTITLE 1,Register Hybrid Search + Reranking Model
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.tracking.client import MlflowClient

# For logging to unity catalog
mlflow.set_registry_uri('databricks-uc')

# Log model
with mlflow.start_run(run_name="wands_hybrid_search_reranker_model") as run:
    model_info = mlflow.pyfunc.log_model(
        python_model=VectorSearchWrapper(
            vs_endpoint_name=VS_ENDPOINT,
            vs_index_name=VS_INDEX,
            query_type='HYBRID',
            vs_cols_to_rerank=VS_COLS_TO_RERANK,
            num_results=VS_NUM_RESULTS
        ),
        artifact_path="wands_hybrid_search_reranker_model",
        registered_model_name=MODEL_NAME,
        pip_requirements=[
            'databricks-sdk==0.57.0',
            'databricks-vectorsearch==0.56', 
            'mlflow==3.1.0',
            'tenacity==9.1.2',
            'pandas==1.5.3', 
            'numpy==1.26.4'
        ],
        input_example=sample_input,
        signature=auto_signature,
        resources=[
            DatabricksVectorSearchIndex(index_name=VS_INDEX),
        ]
    )

    # Add tags for better model management
    mlflow.set_tags({
        'vs_endpoint': VS_ENDPOINT,
        'vs_index': VS_INDEX,
        'vs_source_table': INDEX_SOURCE_TABLE,
        'embedding_endpoint': EMBEDDING_ENDPOINT,
        'model_type': 'VectorSearchWrapper',
        'dataset': 'WANDS',
        'search_type': 'HYBRID_RERANKER'
    })

    # Add model description
    client = MlflowClient()
    client.update_registered_model(
        name=MODEL_NAME,
        description="Vector Search model using hybrid search (vector + keyword) and reranking to find matching products from WANDS dataset"
    )

    print(f"Model logged to {MODEL_NAME}")
    print(f"Model info: {model_info}")
    
    # Create @newest alias using the model_info from registration
    version_number = model_info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "newest", version_number)
    print(f"✅ Created @newest alias pointing to version {version_number}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test registered model

# COMMAND ----------

# Import utility function from utils
from utils import get_latest_model_version

# COMMAND ----------

# Set the registry URI to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Verify the model is registered correctly
client = mlflow.tracking.MlflowClient()
latest_version_info = get_latest_model_version(MODEL_NAME)
model_version_info = client.get_model_version(name=MODEL_NAME, version=latest_version_info.version)
print("Model version info:")
print(model_version_info)

# COMMAND ----------

# Load model
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_version_info.name}/{model_version_info.version}")

print("Model loaded successfully:")
print(loaded_model)

# COMMAND ----------

# Input data with WANDS product search queries
sample_input = pd.DataFrame({"query": ["modern office chair"]})

# Generate predictions
predictions = loaded_model.predict(sample_input)
print(f"Got response of type {type(predictions)} with {len(predictions)} rows")
display(predictions)

# COMMAND ----------

# MAGIC %md ##Step 4: Deploy Hybrid Search + Reranking Endpoint (Optional)
# MAGIC
# MAGIC **⚠️ Note: This section is optional and can take >20 minutes to complete. Model serving endpoints are not required for the next steps in this solution accelerator.**
# MAGIC
# MAGIC Finally, we can deploy our hybrid search and reranking model to a serving endpoint. This creates a REST API that applications can call to perform product search using hybrid search (vector + keyword) combined with reranking for improved relevance.
# MAGIC
# MAGIC **You can skip this section and proceed directly to `05_Fine_Tune_Embedding_Model.py` if you want to continue with the core solution.**

# COMMAND ----------

# DBTITLE 1,Setup Model Serving Endpoint

# Get config for model serving - using hybrid + reranker configuration
MODEL_SERVING_ENDPOINT_NAME = config['hybrid_reranker_endpoint_name']  
MODEL_NAME = config['hybrid_reranker_model_name'] 
VS_ENDPOINT = config['vs_endpoint_name']
VS_INDEX = config['vs_index_basic']
INDEX_SOURCE_TABLE = config['products_table']
EMBEDDING_ENDPOINT = config['embedding_endpoint']

print(f"Setting up model serving endpoint: {MODEL_SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# Get latest model version
model_version_info = get_latest_model_version(MODEL_NAME)
MODEL_VERSION = model_version_info.version

print("Model: ", MODEL_NAME)
print(f"Model latest version: {MODEL_VERSION}")
print("Status of model: ", model_version_info.status)

# COMMAND ----------

# DBTITLE 1,Deploy Hybrid Search + Reranking Endpoint
# Get deploy client
import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

# Create or update model serving endpoint
try:
    # Try to get existing endpoint first
    endpoint = deploy_client.get_endpoint(MODEL_SERVING_ENDPOINT_NAME)
    print(f"✅ Model serving endpoint '{MODEL_SERVING_ENDPOINT_NAME}' already exists")
    
    # Update the endpoint with new model version
    endpoint = deploy_client.update_endpoint(
        endpoint=MODEL_SERVING_ENDPOINT_NAME,
        config={
            "served_entities": [{
                "entity_name": f"{MODEL_NAME}",
                "entity_version": f"{MODEL_VERSION}",
                "workload_type": "CPU",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }],
        }
    )
    print(f"✅ Updated endpoint with new model version {MODEL_VERSION}")
    
except Exception as e:
    if "does not exist" in str(e).lower() or "not found" in str(e).lower():
        # Endpoint doesn't exist, create it
        print(f"⚡ Creating new model serving endpoint: {MODEL_SERVING_ENDPOINT_NAME}")
        endpoint = deploy_client.create_endpoint(
            config={
                "name": f"{MODEL_SERVING_ENDPOINT_NAME}",
                "tags": [
                    {"key": "modeltype", "value": "VectorSearchWrapper"},
                    {"key": "vs_endpoint_name", "value": f"{VS_ENDPOINT}"},
                    {"key": "vs_index", "value": f"{VS_INDEX}"},
                    {"key": "vs_source_table", "value": f"{INDEX_SOURCE_TABLE}"},
                    {"key": "embedding_endpoint", "value": f"{EMBEDDING_ENDPOINT}"},
                    {"key": "dataset", "value": "WANDS"},
                    {"key": "search_type", "value": "HYBRID_RERANKER"}
                ],
                "config": {
                    "served_entities": [{
                        "entity_name": f"{MODEL_NAME}",
                        "entity_version": f"{MODEL_VERSION}",
                        "workload_type": "CPU",
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True
                    }],
                },

            }
        )
        print("✅ Model serving endpoint created successfully!")
    else:
        # Some other error occurred
        print(f"❌ Error with model serving endpoint: {str(e)}")
        raise e

# COMMAND ----------

# Get status of deployment
custom_model_endpoint = deploy_client.get_endpoint(MODEL_SERVING_ENDPOINT_NAME)
print("Endpoint status:")
print(custom_model_endpoint)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
# MAGIC
# MAGIC
