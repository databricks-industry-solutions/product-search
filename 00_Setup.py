# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration
# MAGIC
# MAGIC The following parameters are used throughout the notebooks to control the resources being used. If you modify these variables, please note that markdown in the notebooks may refer to the original values associated with these:

# COMMAND ----------

# DBTITLE 1,Initialize Config Variables
if 'config' not in locals().keys():
    config = {}

# COMMAND ----------

# DBTITLE 1,🔧 REQUIRED SETTINGS - Modify These for Your Environment
# =============================================================================
# Unity Catalog Settings (Required)
# - Change 'main' to your catalog name
# - Change 'wands' to your preferred schema name
# =============================================================================
config['catalog_name'] = 'main'  # ← CHANGE THIS to your catalog name
config['schema_name'] = 'wands'   # ← CHANGE THIS to your schema name

# =============================================================================
# Vector Search Endpoint (Required)
# - Change 'one-env-shared-endpoint-11' to your vector search endpoint
# =============================================================================
config['vs_endpoint_name'] = "one-env-shared-endpoint-11"  # ← CHANGE THIS to your endpoint

# COMMAND ----------

# DBTITLE 1,Unity Catalog Configuration
# Derived settings based on your required settings above
config['volume_name'] = f"{config['schema_name']}_volume"

# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config['catalog_name']}.{config['schema_name']}")
spark.sql(f"USE {config['catalog_name']}.{config['schema_name']}")

print(f"✅ Unity Catalog configured: {config['catalog_name']}.{config['schema_name']}")

# COMMAND ----------

# DBTITLE 1,Storage Paths
config['volume_path'] = f"/Volumes/{config['catalog_name']}/{config['schema_name']}/{config['volume_name']}"
config['data_path'] = f"{config['volume_path']}/data"
config['models_path'] = f"{config['volume_path']}/models"
config['downloads_path'] = f"{config['volume_path']}/downloads"

# Create volume and directories
spark.sql(f"""
CREATE VOLUME IF NOT EXISTS {config['catalog_name']}.{config['schema_name']}.{config['volume_name']}
COMMENT "Volume for WANDS product search data and models"
""")

# Create subdirectories
for path in [config['data_path'], config['models_path'], config['downloads_path']]:
    dbutils.fs.mkdirs(path)

print(f"✅ Storage configured: {config['volume_path']}")

# COMMAND ----------

# DBTITLE 1,Table Names
config['products_table'] = f"{config['catalog_name']}.{config['schema_name']}.products"
config['queries_table'] = f"{config['catalog_name']}.{config['schema_name']}.queries"
config['labels_table'] = f"{config['catalog_name']}.{config['schema_name']}.labels"

# Enhanced tables for fine-tuning
config['products_with_embeddings_table'] = f"{config['catalog_name']}.{config['schema_name']}.products_with_finetuned_embeddings_minilm"

print(f"✅ Tables configured")

# COMMAND ----------

# DBTITLE 1,Models
config['basic_model_name'] = f"{config['catalog_name']}.{config['schema_name']}.wands_basic_search"
config['tuned_model_name'] = f"{config['catalog_name']}.{config['schema_name']}.wands_finetuned_minilm"
config['hybrid_reranker_model_name'] = f"{config['catalog_name']}.{config['schema_name']}.wands_hybrid_search_reranker"
config['hybrid_model_name'] = f"{config['catalog_name']}.{config['schema_name']}.wands_hybrid_search"

print(f"✅ Models configured")

# COMMAND ----------

# DBTITLE 1,Vector Search Configuration
# vs_endpoint_name is configured in the REQUIRED SETTINGS section above
config['vs_index_basic'] = f"{config['catalog_name']}.{config['schema_name']}.wands_product_embeddings"
config['vs_index_finetuned'] = f"{config['catalog_name']}.{config['schema_name']}.wands_product_embeddings_finetuned_minilm"

# Embedding configuration
config['embedding_endpoint'] = "databricks-bge-large-en"
config['embedding_column'] = "embedding_column"
config['finetuned_embedding_column'] = "finetuned_embedding_column"

print(f"✅ Vector Search configured")

# COMMAND ----------

# DBTITLE 1,Model Serving Endpoints
config['basic_endpoint_name'] = 'wands_basic_search_endpoint'
config['hybrid_reranker_endpoint_name'] = 'wands_hybrid_search_reranker_endpoint'
config['hybrid_endpoint_name'] = 'wands_hybrid_search_endpoint'
config['finetuned_endpoint_name'] = 'wands_finetuned_minilm_endpoint'

print(f"✅ Serving endpoints configured")

# COMMAND ----------

# DBTITLE 1,Authentication Configuration
import os
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
config['databricks_token'] = ctx.apiToken().getOrElse(None)
config['databricks_url'] = ctx.apiUrl().getOrElse(None)

os.environ['DATABRICKS_TOKEN'] = config["databricks_token"]
os.environ['DATABRICKS_URL'] = config["databricks_url"]

print(f"✅ Authentication configured")

# COMMAND ----------

# DBTITLE 1,MLflow Configuration
import mlflow
mlflow.set_registry_uri("databricks-uc")

print(f"✅ MLflow configured")

# COMMAND ----------

# DBTITLE 1,Fine-tuning Configuration
config['base_model'] = "all-MiniLM-L12-v2"
config['embedding_dimension'] = 384
config['epochs'] = 1
config['batch_size'] = 16
config['learning_rate'] = 2e-5
config['warmup_steps'] = 100

print(f"✅ Fine-tuning configured")

# COMMAND ----------

# DBTITLE 1,Search Parameters
config['num_results'] = 5
config['columns_to_return'] = [
    "product_id",
    "product_name", 
    "product_description",
    "product_class",
    "category_hierarchy",
    "product_features",
    "average_rating"
]

print(f"✅ Search parameters configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Summary

# COMMAND ----------

# DBTITLE 1,Display Configuration Summary
print("🎯 WANDS Product Search Configuration Summary")
print("=" * 60)
print(f"📊 Data:")
print(f"   Catalog.Schema: {config['catalog_name']}.{config['schema_name']}")
print(f"   Volume: {config['volume_name']}")
print(f"   Storage: {config['volume_path']}")
print()
print(f"🔍 Models:")
print(f"   Basic (ANN): {config['basic_model_name']}")
print(f"   Hybrid Search: {config['hybrid_model_name']}")
print(f"   Hybrid + Reranker: {config['hybrid_reranker_model_name']}")
print(f"   Fine-tuned: {config['tuned_model_name']}")
print()
print(f"🚀 Vector Search:")
print(f"   Endpoint: {config['vs_endpoint_name']}")
print(f"   Basic Index: {config['vs_index_basic']}")
print(f"   Fine-tuned Index: {config['vs_index_finetuned']}")
print()
print(f"⚙️ Fine-tuning:")
print(f"   Base Model: {config['base_model']}")
print(f"   Dimensions: {config['embedding_dimension']}")
print(f"   Epochs: {config['epochs']}")
print()
print("✅ Setup complete! Use `config` dictionary in other notebooks.")
print("💡 Run: %run ./00_Setup to load this configuration in other notebooks")

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ | 
