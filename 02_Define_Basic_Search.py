# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our data in place, we will now take an off-the-shelf model and apply it to perform product search. A key part of this work is the introduction of a vector database that our model will use during inference to rapidly search the product catalog.
# MAGIC
# MAGIC To understand the vector database, you first need to understand *embeddings*. An embedding is an array of numbers that indicate the degree to which a unit of text aligns with clusters of words frequently found together in a set of documents. The exact details as to how these numbers are estimated isn't terribly important here. What is important is to understand that the mathematical distance between two embeddings generated through the same model tells us something about the similarity of two documents. When we perform a search, the user's search phrase is used to generate an embedding and it's compared to the pre-existing embeddings associated with the products in our catalog to determine which ones the search is closest to. Those closest become the results of our search.
# MAGIC
# MAGIC To facilitate the fast retrieval of items using embedding similarities, we need a specialized database capable of not only storing embeddings but enabling a rapid search against numerical arrays. The class of data stores that addresses these needs are called vector stores. In this notebook, we'll use **Databricks Vector Search**, a fully managed vector database service that integrates seamlessly with Unity Catalog and Delta Lake.
# MAGIC
# MAGIC In this notebook, we will use a pre-trained BGE model, convert our product text to embeddings using this model, store our embeddings in a Databricks Vector Search index, and then package the search functionality for later deployment behind a REST API.

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# Install packages with version checking
%pip install --upgrade --quiet databricks-vectorsearch databricks-sdk mlflow tenacity
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries and Load Configuration
import yaml
import time
import json
import pandas as pd
from typing import Dict, Any, List, Optional

# Databricks imports
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput
from databricks.sdk.service.vectorsearch import EndpointType

# Spark and MLflow imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import mlflow
from mlflow.deployments import get_deploy_client
from mlflow.models import infer_signature
from mlflow.models.resources import DatabricksVectorSearchIndex
from mlflow.tracking.client import MlflowClient

# COMMAND ----------
%run ./00_Setup

# COMMAND ----------

print(f"✅ Configuration loaded from 00_Setup")
print(f"Using catalog: {config['catalog_name']}, schema: {config['schema_name']}")

# Initialize Vector Search client and import helper functions
from utils import (
    create_vector_search_client,
    endpoint_exists,
    wait_for_vs_endpoint_to_be_ready,
    index_exists,
    wait_for_index_to_be_ready,
    get_vs_results_df,
    get_workspace_url,
    get_latest_model_version
)

vsc = create_vector_search_client(config)
print("✅ Vector Search client initialized")
print("✅ Helper functions imported from utils")

# COMMAND ----------

# MAGIC %md ##Step 1: Assemble Product Info
# MAGIC
# MAGIC In this first step, we need to assemble the product text data against which we intend to search. We will use our product description as that text unless there is no description in which case we will use the product name.
# MAGIC
# MAGIC In addition to the searchable text, we will provide product metadata, such as product ids and names, that will be returned with our search results:

# COMMAND ----------

# DBTITLE 1,Verify Products Data for Vector Search
# Load products data - using existing table with embedding_column
products_table_name = config['products_table']
data_table_name = config['products_table']  # Same table for basic search

print(f"📊 Using products data from: {products_table_name}")
print(f"🎯 Vector search will use table: {data_table_name}")

# Verify products table has the required embedding_column
products_df = spark.table(products_table_name)
if "embedding_column" not in products_df.columns:
    raise ValueError("Products table is missing required 'embedding_column' for vector search")

print(f"✅ Products table verified. Records: {products_df.count()}")
print(f"✅ Using existing embedding_column for vector search")

# COMMAND ----------

# MAGIC %md ##Step 2: Convert Product Info into Embeddings
# MAGIC
# MAGIC We will now convert our product text into embeddings using the BGE model and create our vector search index. The instructions for converting text into an embedding is captured in a language model. The [*BGE-large* model](https://huggingface.co/BAAI/bge-large-en-v1.5) is a high-quality embedding model which has been trained on a large, well-rounded corpus of input text for excellent performance in a variety of document search scenarios. The BGE model generates rich, 1024-dimensional embeddings that capture semantic meaning effectively for our product search needs.
# MAGIC
# MAGIC Using our BGE model, we can now generate embeddings. We'll persist these to Databricks Vector Search, a fully managed vector database service that will allow us to retrieve vector data efficiently in future steps:

# COMMAND ----------

# DBTITLE 1,Setup Vector Search Endpoint
# Get vector search configuration for basic search
vs_endpoint_name = config['vs_endpoint_name']

print(f"🔧 Setting up Vector Search endpoint: {vs_endpoint_name}")

# Create Vector Search endpoint if it doesn't exist
try:
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    print(f"✅ Vector search endpoint '{vs_endpoint_name}' already exists")
except Exception:
    print(f"⚡ Creating new vector search endpoint '{vs_endpoint_name}'")
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# Wait for endpoint to be ready
wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
vs_index_name = config['vs_index_basic']
embedding_endpoint = config['embedding_endpoint']
index_column = config['embedding_column']

print(f"🔍 Setting up Vector Search index: {vs_index_name}")
print(f"🤖 Embedding endpoint: {embedding_endpoint}")
print(f"📝 Index column: {index_column}")

# Create or get vector search index
if index_exists(vsc, vs_endpoint_name, vs_index_name):
    print(f"✅ Vector search index '{vs_index_name}' already exists")
    index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_name)
    
    # Sync existing index to get latest data changes
    print(f"🔄 Syncing existing index with latest data...")
    index.sync()
    
else:
    print(f"⚡ Creating new vector search index '{vs_index_name}'")
    index = vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        source_table_name=data_table_name,
        index_name=vs_index_name,
        pipeline_type="TRIGGERED",
        primary_key="product_id",
        embedding_source_column=index_column,
        embedding_model_endpoint_name=embedding_endpoint
    )
    print(f"🚀 Vector search index '{vs_index_name}' creation initiated")

# Wait for index to be ready
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_name)
print(f"✅ Vector search index '{vs_index_name}' is ready!")

# COMMAND ----------

# MAGIC %md From the index statistics, we can see that every product entry from our original dataframe has been loaded into the vector search index:

# COMMAND ----------

# MAGIC %md We can examine our vector search index to see how our data has been transformed. Our product metadata such as product id and product name are stored as searchable columns in the index. The text we identified for embedding is processed by the BGE model to create 1024-dimensional embeddings that capture the semantic meaning of our product descriptions:

# COMMAND ----------

# MAGIC %md ##Step 3: Demonstrate Basic Search Capability
# MAGIC
# MAGIC To get a sense of how our search will work, we can perform a similarity search on our vector database:

# COMMAND ----------

# DBTITLE 1,Test Vector Search Functionality
print("🧪 Testing vector search functionality...")

# Test search using the vector search client directly
test_queries = [
    "comfortable office chair"
]

num_results = config['num_results']

print(f"🔍 Testing {len(test_queries)} sample queries (returning top {num_results} results each)")

for i, query in enumerate(test_queries, 1):
    print(f"\n--- Test Query {i}: '{query}' ---")
    
    try:
        # Direct Vector Search Client
        results = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_name).similarity_search(
            query_text=query,
            columns=config['columns_to_return'],
            num_results=num_results
        )
        
        # Show results as a dataframe using the simple helper function
        if results:
            print(f"✅ Found results for query: '{query}'")
            display(get_vs_results_df(results))
        else:
            print("⚠️ No results returned")
            
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")

print("\n✅ Vector search testing completed!")

# COMMAND ----------

# MAGIC %md Notice that while some of the results reflect key terms, some do not. This form of search is leveraging embeddings which understand that terms like *child*, *children*, *kid* and *kids* often are associated with each other. And while the exact term might not appear in every result, the presence of related concepts indicates that the results are close in concept to what we are searching for.

# COMMAND ----------

# MAGIC %md ##Step 4: Persist Model for Deployment
# MAGIC
# MAGIC At this point, we have all the elements in place to build a deployable model. In the Databricks environment, deployment typically takes place using [MLflow](https://www.databricks.com/product/managed-mlflow), which has the ability to build a containerized service from our model as one of its deployment patterns. Generic Python models deployed with MLflow typically support a standard API with a *predict* method that's called for inference. We will need to write a custom wrapper to map a standard interface to our Databricks Vector Search functionality as follows:

# COMMAND ----------

# DBTITLE 1,Define Wrapper Class for Model
# Get authentication credentials for model - using automatic notebook authentication
DATABRICKS_HOST = config['databricks_url']
DATABRICKS_TOKEN = config['databricks_token']

# Get model configuration
MODEL_NAME = config['basic_model_name']
VS_COLS_TO_RETURN = config['columns_to_return']
VS_NUM_RESULTS = config['num_results']

import mlflow.pyfunc
from tenacity import Retrying, stop_after_attempt, wait_exponential

class VectorSearchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, vs_endpoint_name: str, vs_index_name: str, query_type: str = 'ANN', num_results: int = 5):
        self.vs_endpoint_name = vs_endpoint_name
        self.vs_index_name = vs_index_name
        self.num_results = num_results
        self.query_type = query_type
        self.vs_cols_to_return = VS_COLS_TO_RETURN
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
                return self.index.similarity_search(
                    query_text=query_text,
                    columns=self.vs_cols_to_return,
                    num_results=self.num_results,
                    disable_notice=True
                )

    def _format_results(self, raw_response: Dict) -> List[Dict]:
        """Optimized result formatting"""
        return [
            dict(zip(
                [col['name'] for col in raw_response['manifest']['columns']],
                row
            ))
            for row in raw_response.get("result", {}).get("data_array", [])
        ]

# COMMAND ----------

# DBTITLE 1,Test Model
# Sample data for signature inference with WANDS-relevant queries
sample_input = pd.DataFrame({"query": ["modern office chair"]})
    
# Initialize model with test configuration
test_model = VectorSearchWrapper(
    vs_endpoint_name=vs_endpoint_name,
    vs_index_name=vs_index_name,
    num_results=VS_NUM_RESULTS
)
  
test_model.load_context(None)
sample_output = test_model.predict(None, sample_input)

print("Sample Input:")
display(sample_input)
print("\nSample Output:")
display(sample_output)


# COMMAND ----------

# DBTITLE 1,Infer Model Signature
# Infer signature
auto_signature = infer_signature(sample_input, sample_output)
print("Model signature:")
print(auto_signature)

# COMMAND ----------

# DBTITLE 1,Persist Model to MLflow
# For logging to unity catalog
mlflow.set_registry_uri('databricks-uc')

# Log model
with mlflow.start_run(run_name="wands_basic_search_model") as run:
    model_info = mlflow.pyfunc.log_model(
        python_model=VectorSearchWrapper(
            vs_endpoint_name=vs_endpoint_name,
            vs_index_name=vs_index_name,
            num_results=VS_NUM_RESULTS
        ),
        artifact_path="wands_basic_search_model",
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
            DatabricksVectorSearchIndex(index_name=vs_index_name),
        ]
    )

    # Add tags for better model management
    mlflow.set_tags({
        'vs_endpoint': vs_endpoint_name,
        'vs_index': vs_index_name,
        'vs_source_table': data_table_name,
        'embedding_endpoint': embedding_endpoint,
        'model_type': 'VectorSearchWrapper',
        'dataset': 'WANDS',
        'search_type': 'ANN'
    })

    # Add model description
    client = MlflowClient()
    client.update_registered_model(
        name=MODEL_NAME,
        description="Vector Search based model using ANN search to find matching products from WANDS dataset"
    )

    print(f"Model logged to {MODEL_NAME}")
    
    # Create @newest alias using the model_info from registration
    version_number = model_info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "newest", version_number)
    print(f"✅ Created @newest alias pointing to version {version_number}")

# COMMAND ----------

# DBTITLE 1,Test registered model
# Set the registry URI to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Verify the model is registered correctly
client = mlflow.tracking.MlflowClient()
latest_version_info = get_latest_model_version(MODEL_NAME)
model_version_info = client.get_model_version(name=MODEL_NAME, version=latest_version_info.version)
print("Model version info:")
print(model_version_info)

# COMMAND ----------

# DBTITLE 1,Load model
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

# MAGIC %md If you are curious why we are constructing a pandas dataframe for our search term, please understand that this aligns with how data will eventually be passed to our model when we host it in model serving. The logic in our *predict* function anticipates this as well.
# MAGIC
# MAGIC Inferencing a single record can take approximately 50-300 ms, allowing the model to be served and used by a user-facing webapp.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve model with Model Serving (Optional)
# MAGIC
# MAGIC **⚠️ Note: This section is optional and can take >20 minutes to complete. Model serving endpoints are not required for the next steps in this solution accelerator.**
# MAGIC
# MAGIC To deploy our model, we can use Databricks Model Serving to create a REST API endpoint. This allows applications to call our search functionality in real-time. The model serving infrastructure handles scaling, monitoring, and provides a standard API interface for our custom vector search wrapper.

# COMMAND ----------

# DBTITLE 1,Setup Model Serving Endpoint
# Get config for model serving
MODEL_SERVING_ENDPOINT_NAME = config['basic_endpoint_name']

print(f"Setting up model serving endpoint: {MODEL_SERVING_ENDPOINT_NAME}")

# Get latest model version
model_version_info = get_latest_model_version(MODEL_NAME)
MODEL_VERSION = model_version_info.version

print("Model: ", MODEL_NAME)
print(f"Model latest version: {MODEL_VERSION}")
print("Status of model: ", model_version_info.status)

# COMMAND ----------

# DBTITLE 1,Deploy Model Serving Endpoint
# Get deploy client
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
                    {"key": "vs_endpoint_name", "value": f"{vs_endpoint_name}"},
                    {"key": "vs_index", "value": f"{vs_index_name}"},
                    {"key": "vs_source_table", "value": f"{data_table_name}"},
                    {"key": "embedding_endpoint", "value": f"{embedding_endpoint}"},
                    {"key": "dataset", "value": "WANDS"},
                    {"key": "search_type", "value": "ANN"}
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
