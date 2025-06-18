# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tune all-MiniLM-L12-v2 Embeddings for WANDS Product Search
# MAGIC
# MAGIC Having demonstrated the basics of assembling a model and supporting data to enable a semantic search, we will now focus on fine-tuning the model. During fine-tuning, the model is fit against a set of data specific to a particular domain, such as our product catalog. The original knowledge accumulated by our model from its pre-training remains intact but is supplemented with information gleaned from the additional data provided. Once the model has been tuned to our satisfaction, it is packaged and persisted just like before.
# MAGIC
# MAGIC In this notebook, we will use the **all-MiniLM-L12-v2** model, fine-tune it on WANDS query-product pairs with relevance scores, and then deploy it using Databricks Vector Search for improved product search performance. This model provides excellent performance with 384-dimensional embeddings and has proven effectiveness in semantic search tasks.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Run `01_Data_Prep.py` (WANDS data loaded)
# MAGIC - Run `02_Define_Basic_Search.py` (baseline vector search setup)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install & Load Libraries

# COMMAND ----------

# Install required packages for fine-tuning
# Only install what's not pre-installed:
%pip install -qU sentence-transformers
%pip install -qU torch==2.6.0
%pip install -qU transformers
%pip install -qU huggingface_hub
%pip install -qU databricks-vectorsearch
%pip install -qU databricks-sdk
%pip install -qU tenacity
dbutils.library.restartPython()

# COMMAND ----------

import yaml
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import time
import json
from pathlib import Path
import os

# Sentence transformers for fine-tuning
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# Databricks imports
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.models import infer_signature
from mlflow.models.resources import DatabricksVectorSearchIndex

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

print("✅ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration and Setup

# COMMAND ----------
%run ./00_Setup

# COMMAND ----------
print(f"✅ Configuration loaded from 00_Setup")

# Initialize Vector Search client using utils
from utils import create_vector_search_client
vsc = create_vector_search_client(config)

# MLflow setup
mlflow.set_registry_uri("databricks-uc")
print("✅ Databricks clients initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Estimate Baseline Model Performance
# MAGIC
# MAGIC In this first step, we'll retrieve the queries and the products returned with each from the WANDS dataset. For each query-product combination, a numerical score assigned to each combination based on the perceived alignment of the product with the query is retrieved as well:

# COMMAND ----------

# DBTITLE 1,Get Search Results
# assemble product text relevant to search
search_pd = (
  spark   
    .table(config['products_table'])
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
    .join(
      spark.table(config['labels_table']),
        on='product_id'
      )
    .join(
      spark.table(config['queries_table']),
        on='query_id'
      )
      .selectExpr('query','product_text','label_score as score')
  ).toPandas()

display(search_pd)

# COMMAND ----------

# MAGIC %md We will then download the original model used in the last notebook so that we may convert both the queries and the product text information into embeddings:

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
# download embeddings model
original_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# DBTITLE 1,Convert Queries & Products to Embeddings
query_embeddings = (
  original_model
    .encode(
      search_pd['query'].tolist()
      )
  )

product_embeddings = (
  original_model
    .encode(
      search_pd['product_text'].tolist()
      )
  )

# COMMAND ----------

# MAGIC %md We can then calculate the cosine similarity between the queries and products associated with them. While we talk about similarity between embeddings as having to do with the distance between two vectors, cosine similarity refers to the angle separating two rays extending from the center of a space to the point identified by the vector (as if it were a coordinate). In a normalized vector space, this angle also captures the degree of similarity between two points:

# COMMAND ----------

# DBTITLE 1,Calculate Cosine Similarity Between Queries and Products
# determine cosine similarity for each query-product pair
from sentence_transformers import util
original_cos_sim_scores = (
  util.pairwise_cos_sim(
    query_embeddings, 
    product_embeddings
    )
  )

# COMMAND ----------

# MAGIC %md Averaging these values gives us a sense of how close the queries are to the products in the original embedding space. Please note that cosine similarity ranges from 0.0 to 1.0 with values improving as they approach 1.0:

# COMMAND ----------

# DBTITLE 1,Calculate Avg Cosine Similarity
# average the cosine similarity scores
original_cos_sim_score = torch.mean(original_cos_sim_scores).item()

# display result
print(original_cos_sim_score)

# COMMAND ----------

# MAGIC %md Examining the correlation between the label scores and the cosine similarity can provide us another measure of the model's performance:

# COMMAND ----------

# DBTITLE 1,Calculate Correlation with Scores 
# determine correlation between cosine similarities and relevancy scores
original_corr_coef_score = (
  np.corrcoef(
    original_cos_sim_scores,
    search_pd['score'].values
  )[0][1]
) 
# print results
print(original_corr_coef_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fine-Tune the Model
# MAGIC
# MAGIC With a baseline measurement of the original model's performance in-hand, we can now fine-tune it using our annotated search result data. We will start by restructuring our query results into a list of inputs as required by the model:

# COMMAND ----------

# DBTITLE 1,Restructure Data for Model Input
# define function to assemble an input
def create_input(doc1, doc2, score):
  return InputExample(texts=[doc1, doc2], label=score)

# convert each search result into an input
inputs = search_pd.apply(
  lambda s: create_input(s['query'], s['product_text'], s['score']), axis=1
  ).to_list()

# COMMAND ----------

# MAGIC %md We will then download a separate copy of our original model so that we may tune it:

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
tuned_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# MAGIC %md And we will then tune the model to minimize cosine similarity distances:
# MAGIC
# MAGIC **⏱️ Runtime Note:** This fine-tuning step typically takes **30 minutes** to complete. The process will run faster with GPU-enabled clusters (recommended: `g5.4xlarge` on AWS, `Standard_NC6s_v3` on Azure, or `n1-standard-4` with NVIDIA T4 on GCP).

# COMMAND ----------

# DBTITLE 1,Tune the Model
# define instructions for feeding inputs to model
input_dataloader = DataLoader(inputs, shuffle=True, batch_size=16) # feed 16 records at a time to the model

# define loss metric to optimize for
loss = losses.CosineSimilarityLoss(tuned_model)

# tune the model on the input data
tuned_model.fit(
  train_objectives=[(input_dataloader, loss)],
  epochs=1, # just make 1 pass over data
  warmup_steps=100 # controls how many steps over which learning rate increases to max before descending back to zero
  )

# COMMAND ----------

# MAGIC %md During model fitting, you will notice we are setting the model to perform just one pass (epoch) over the data. We will actually see pretty sizeable improvements from this process, but we may wish to increase that value to get multiple passes if we want to explore getting more. The setting for *warmup_steps* is just a common one used in this space. Feel free to experiment with other values or take the default.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Estimate Fine-Tuned Model Performance
# MAGIC
# MAGIC With our model tuned, we can assess its performance just like we did before:

# COMMAND ----------

# DBTITLE 1,Calculate Cosine Similarities for Queries & Products in Tuned Model
query_embeddings = (
  tuned_model
    .encode(
      search_pd['query'].tolist()
      )
  )

product_embeddings = (
  tuned_model
    .encode(
      search_pd['product_text'].tolist()
      )
  )

# determine cosine similarity for each query-product pair
tuned_cos_sim_scores = (
  util.pairwise_cos_sim(
    query_embeddings, 
    product_embeddings
    )
  )

# COMMAND ----------

# DBTITLE 1,Calculate Avg Cosine Similarity
# average the cosine similarity scores
tuned_cos_sim_score = torch.mean(tuned_cos_sim_scores).item()

# display result
print(f"With tuning, avg cosine similarity went from {original_cos_sim_score} to {tuned_cos_sim_score}")

# COMMAND ----------

# DBTITLE 1,Calculate Correlation Coefficient
# determine correlation between cosine similarities and relevancy scores
tuned_corr_coef_score = (
  np.corrcoef(
    tuned_cos_sim_scores,
    search_pd['score'].values
  )[0][1]
) 
# print results
print(f"With tuning, the correlation coefficient went from {original_corr_coef_score} to {tuned_corr_coef_score}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Persist Model for Deployment
# MAGIC
# MAGIC Just like before, we can package our tuned model with our data to enable its persistence (and eventual deployment). We'll save the fine-tuned model, create enhanced embeddings for vector search, and deploy it as an MLflow model:

# COMMAND ----------

# DBTITLE 1,Save Fine-Tuned Model to Unity Catalog Volume
# Save the tuned model to Unity Catalog Volume for deployment
volume_model_path = config['models_path'] + "/fine_tuned_minilm"
print("💾 Saving fine-tuned all-MiniLM-L12-v2 model to Unity Catalog Volume...")
tuned_model.save(volume_model_path)
print(f"✅ Model saved to: {volume_model_path}")

# COMMAND ----------

# DBTITLE 1,Create Enhanced Products Table with Fine-Tuned Embeddings
print("🔧 Creating enhanced products table with fine-tuned embeddings...")

# Load all products for embedding generation
products_df = spark.table(config['products_table'])
products_pd = products_df.toPandas()

print(f"   • Processing {len(products_pd)} products...")

# Generate embeddings using fine-tuned model
print("   • Generating fine-tuned embeddings...")
fine_tuned_embeddings = tuned_model.encode(
    products_pd['embedding_column'].fillna('').tolist(),
    show_progress_bar=True
)

print(f"   • Generated embeddings with shape: {fine_tuned_embeddings.shape}")

# Convert embeddings to list format for Spark
embeddings_list = [embedding.tolist() for embedding in fine_tuned_embeddings]

# Add embeddings to pandas dataframe
products_pd['finetuned_embedding_column'] = embeddings_list

# Convert back to Spark DataFrame
enhanced_products_df = spark.createDataFrame(products_pd)

# Save enhanced products table
enhanced_table_name = config['products_with_embeddings_table']
print(f"   • Saving enhanced products table: {enhanced_table_name}")

(enhanced_products_df
 .write
 .format('delta')
 .mode('overwrite')
 .option('overwriteSchema', 'true')
 .saveAsTable(enhanced_table_name)
)

# Enable Change Data Feed for vector search
spark.sql(f"ALTER TABLE {enhanced_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"✅ Enhanced products table created with fine-tuned embeddings")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index with Fine-Tuned Embeddings
print("🔍 Creating vector search index with fine-tuned embeddings...")

# Extract configuration
vs_endpoint_name = config['vs_endpoint_name']
vs_index_name = config['vs_index_finetuned']
embedding_vector_column = config['finetuned_embedding_column']

print(f"   • Endpoint: {vs_endpoint_name}")
print(f"   • Index: {vs_index_name}")
print(f"   • Data table: {enhanced_table_name}")
print(f"   • Embedding vector column: {embedding_vector_column}")

# all-MiniLM-L12-v2 embeddings have 384 dimensions
embedding_dimension = 384

# Import helper functions from utils
from utils import wait_for_index_to_be_ready, index_exists

# Create or get self-managed vector index
if index_exists(vsc, vs_endpoint_name, vs_index_name):
    print(f"   • Index {vs_index_name} already exists")
    index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_name)
    
    # Sync existing index with latest fine-tuned embeddings
    print("   • Syncing existing index with latest fine-tuned embeddings...")
    index.sync()
    
else:
    print(f"   • Creating new self-managed vector index...")
    index = vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        source_table_name=enhanced_table_name,
        index_name=vs_index_name,
        pipeline_type="TRIGGERED",
        primary_key="product_id",
        embedding_vector_column=embedding_vector_column,
        embedding_dimension=embedding_dimension
    )
    print(f"   • Index creation initiated")

# Wait for index to be ready
print("   • Waiting for index to be ready...")
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_name)
print("✅ Fine-tuned vector search index is ready")

# COMMAND ----------

# DBTITLE 1,Define Wrapper Class for Fine-Tuned Model
# Use automatic authentication - no credentials needed

# Get model configuration
MODEL_NAME = config['tuned_model_name']
VS_COLS_TO_RETURN = config['columns_to_return']

class FineTunedVectorSearchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, vs_endpoint_name: str, vs_index_name: str, num_results: int = 5, model_path: str = None):
        self.vs_endpoint_name = vs_endpoint_name
        self.vs_index_name = vs_index_name
        self.num_results = num_results
        self.vs_cols_to_return = VS_COLS_TO_RETURN
        self.model_path = model_path or f"{config['models_path']}/fine_tuned_minilm"
        self.index = None
        self.fine_tuned_model = None
    
    def load_context(self, context):
        """Initialize during endpoint creation"""
        from databricks.vector_search.client import VectorSearchClient
        from sentence_transformers import SentenceTransformer
        
        self.vs_client = VectorSearchClient()
        
        self.index = self.vs_client.get_index(
            endpoint_name=self.vs_endpoint_name,
            index_name=self.vs_index_name
        )
        
        # Load the fine-tuned model for query encoding from Unity Catalog Volume
        self.fine_tuned_model = SentenceTransformer(self.model_path)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Batch query execution using fine-tuned model"""
        results = []
        
        for query in model_input["query"]:
            # Encode query using fine-tuned model
            query_vector = self.fine_tuned_model.encode([query])[0].tolist()
            
            # Use WorkspaceClient for self-managed vector search
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            
            search_result = w.vector_search_indexes.query_index(
                index_name=self.vs_index_name,
                query_vector=query_vector,
                columns=self.vs_cols_to_return,
                num_results=self.num_results
            )
            
            # Format results with proper null checking
            if (hasattr(search_result, 'result') and 
                hasattr(search_result.result, 'data_array') and 
                search_result.result.data_array is not None and
                hasattr(search_result, 'manifest') and
                hasattr(search_result.manifest, 'columns')):
                
                # Get column names from manifest
                column_names = [col.name for col in search_result.manifest.columns]
                
                formatted_results = [
                    dict(zip(column_names, row))
                    for row in search_result.result.data_array
                ]
                results.extend(formatted_results)
            else:
                print(f"   ⚠️ No results returned for query: '{query}'")
                
        return pd.DataFrame(results)

# COMMAND ----------

# DBTITLE 1,Test Fine-Tuned Model
# Sample data for signature inference
sample_input = pd.DataFrame({"query": ["modern office chair"]})

# Initialize model with test configuration
test_model = FineTunedVectorSearchWrapper(
    vs_endpoint_name=vs_endpoint_name,
    vs_index_name=vs_index_name,
    model_path=volume_model_path
)

try:
    test_model.load_context(None)
    sample_output = test_model.predict(None, sample_input)
    print(f"✅ Test successful: {len(sample_output)} results returned")
    print("Sample Input:")
    display(sample_input)
    print("\nSample Output:")
    display(sample_output)
except Exception as e:
    print(f"⚠️ Test failed: {str(e)}")
    print("   • This might be due to index still syncing. Proceeding with registration...")
    # Create a dummy output for signature inference
    sample_output = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Sample Product 1', 'Sample Product 2', 'Sample Product 3', 'Sample Product 4', 'Sample Product 5'],
        'product_class': ['Sample Class'] * 5,
        'average_rating': [4.5] * 5
    })

# COMMAND ----------

# DBTITLE 1,Register Fine-Tuned Model to MLflow
# For logging to unity catalog
mlflow.set_registry_uri('databricks-uc')

# Register model - ensure we're not in an existing run context
try:
    mlflow.end_run()  # End any existing run
except:
    pass

with mlflow.start_run(run_name="wands_finetuned_search_small_model") as run:
    # Infer signature
    signature = infer_signature(sample_input, sample_output)

    model_info = mlflow.pyfunc.log_model(
        python_model=FineTunedVectorSearchWrapper(
            vs_endpoint_name=vs_endpoint_name,
            vs_index_name=vs_index_name,
            model_path=volume_model_path
        ),
        artifact_path="wands_finetuned_search_small_model",
        registered_model_name=MODEL_NAME,
        pip_requirements=[
            'databricks-sdk==0.57.0',
            'databricks-vectorsearch==0.56', 
            'mlflow==3.1.0',
            'tenacity==9.1.2',
            'pandas==1.5.3', 
            'numpy==1.26.4',
            'sentence-transformers==4.1.0',
            'torch==2.6.0',
            'transformers==4.52.4',
            'huggingface_hub==0.33.0'
        ],
        input_example=sample_input,
        signature=signature,
        resources=[
            DatabricksVectorSearchIndex(index_name=vs_index_name),
        ]
    )

    # Log performance metrics
    mlflow.log_metrics({
        'original_cos_sim_score': original_cos_sim_score,
        'tuned_cos_sim_score': tuned_cos_sim_score,
        'original_corr_coef_score': original_corr_coef_score,
        'tuned_corr_coef_score': tuned_corr_coef_score,
        'cos_sim_improvement': tuned_cos_sim_score - original_cos_sim_score,
        'corr_coef_improvement': tuned_corr_coef_score - original_corr_coef_score
    })
    
    # Log parameters
    mlflow.log_params({
        'base_model': 'all-MiniLM-L12-v2',
        'epochs': 1,
        'batch_size': 16,
        'warmup_steps': 100,
        'loss_function': 'CosineSimilarityLoss',
        'embedding_dimension': 384
    })
    
    # Add tags
    mlflow.set_tags({
        'vs_endpoint': vs_endpoint_name,
        'vs_index': vs_index_name,
        'model_type': 'FineTunedVectorSearchWrapper',
        'dataset': 'WANDS',
        'search_type': 'ANN_FINETUNED',
        'base_model': 'all-MiniLM-L12-v2'
    })

    print(f"✅ Model registered: {MODEL_NAME}")
    
    # Create @newest alias using the model_info from registration
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()
    
    # Use the registered_model_version from model_info directly
    version_number = model_info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "newest", version_number)
    print(f"✅ Created @newest alias pointing to version {version_number}")

# COMMAND ----------

# DBTITLE 1,Test the Persisted Model
# Load model from registry
from utils import get_latest_model_version

model_version_info = get_latest_model_version(MODEL_NAME)
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version_info.version}")

# Test the model
search = pd.DataFrame({'query':['farmhouse dining room table']})
print("Testing registered model:")
display(loaded_model.predict(search))


# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
# MAGIC
# MAGIC
