# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC The purpose of this solution accelerator is to show how large language models (LLMs) and their smaller brethren can be used to enable product search.  Unlike product search used in most sites today that rely upon keyword matches, LLMs enable what is commonly referred to as a semantic search where the *conceptual similarities* in words come into play.
# MAGIC
# MAGIC A model's knowledge of the *conceptual similarity* between words comes from being exposed to a wide range of documents and from those documents learning that certain words tend to have close relationships to one another.  For example, one document may discuss the importance of play for *children* and use the term *child* teaching the model that *children* and *child* have some kind of relationship.  Other documents may use these terms in similar proximity and other documents discussing the same topics may introduce the term *kid* or *kids*.  It's possible that in some documents all four terms pop-up but even if that never happens, there may be enough overlap in the words surrounding these terms that the model comes to recognize a close association between all these terms.
# MAGIC
# MAGIC Many of the LLMs available from the open source community come available  as pre-trained models where these word associations have already been learned from a wide range of publicly available  information. With the knowledge these models have already accumulated, they can be used to search the descriptive text for products in a product catalog for items that seem aligned with a search term or phrase supplied by a user. Where the products featured on a site tend to use a more specific set of terms that have their own patterns of association reflecting the tone and style of the retailer or the suppliers they feature, these models can be exposed to additional data specific to the site to shape its understanding of the language being used.  This *fine-tuning* exercise can be used to tailor an off-the-shelf model to the nuances of a specific product catalog, enabling even more effective search results.
# MAGIC
# MAGIC In this solution accelerator, we will show both versions of this pattern using an off-the-shelf model and one tuned to a specific body of product text. We'll then tackle the issues related to model deployment so that users can see how a semantic search capability can easily be deployed through their Databricks environment.
# MAGIC </p>
# MAGIC
# MAGIC <img src='images/inference.png' width=800>

# COMMAND ----------

# MAGIC %md ## Dataset Overview
# MAGIC
# MAGIC ### **WANDS Dataset Foundation**
# MAGIC
# MAGIC We use the [Wayfair Annotation Dataset (WANDS)](https://www.aboutwayfair.com/careers/tech-blog/wayfair-releases-wands-the-largest-and-richest-publicly-available-dataset-for-e-commerce-product-search-relevance), made available by [Wayfair](https://www.wayfair.com/) under an MIT License. This comprehensive dataset provides:
# MAGIC
# MAGIC **Dataset Components:**
# MAGIC * **Products**: 42,000+ furniture and home goods from Wayfair
# MAGIC * **Queries**: 480 real customer search queries across different product categories  
# MAGIC * **Labels**: 233,000+ query-product pairs with human-annotated relevance scores
# MAGIC
# MAGIC **Relevance Scoring System:**
# MAGIC * **Exact Match (1.0)**: Product fully matches the search query intent
# MAGIC * **Partial Match (0.75)**: Product partially relevant but not a perfect match
# MAGIC * **Irrelevant (0.0)**: Product not relevant to the query
# MAGIC
# MAGIC This scoring system, based on informed human judgment from Wayfair's [Annotation Guidelines](https://github.com/wayfair/WANDS/blob/main/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf), provides the ground truth for evaluating and comparing our three search approaches.
# MAGIC
# MAGIC
# MAGIC The solution demonstrates enterprise-grade implementation using:
# MAGIC - **Unity Catalog** for data governance and lineage
# MAGIC - **Delta Lake** with Change Data Feed for real-time vector search
# MAGIC - **MLflow** for model lifecycle management and serving
# MAGIC - **Databricks Vector Search** for scalable similarity search
# MAGIC - **Service Principal Authentication** for production security
# MAGIC
# MAGIC ### **Solution Flow**
# MAGIC
# MAGIC ```
# MAGIC WANDS Raw Data → Unity Catalog Tables → Vector Search Index → MLflow Models → Serving Endpoints (optional)
# MAGIC      │                    │                     │                │              │
# MAGIC   CSV Files         products/queries/      BGE Embeddings    Registered     REST APIs
# MAGIC                        labels                                 Models
# MAGIC ```
# MAGIC
# MAGIC **This notebook (01_Data_Prep.py) handles the first step**: downloading WANDS data and creating the foundational Unity Catalog tables that power all three search approaches.You may find this notebook on https://github.com/databricks-industry-solutions/product-search.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Recommended Compute Configuration**
# MAGIC
# MAGIC For optimal performance with this solution accelerator, we recommend using a **GPU-enabled single node cluster**:
# MAGIC
# MAGIC **Cluster Configuration:**
# MAGIC - **Node Type**: `g5.4xlarge` (AWS) / `Standard_NC6s_v3` (Azure) / `n1-standard-4` with NVIDIA T4 (GCP)
# MAGIC - **Runtime**: `16.4 LTS ML` or higher with GPU support
# MAGIC - **Workers**: 0 (single node configuration)
# MAGIC - **GPU**: NVIDIA A10G (AWS) / NVIDIA V100 (Azure) / NVIDIA T4 (GCP)
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from delta.tables import *
import requests

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run ./00_Setup

# COMMAND ----------

print(f"✅ Configuration loaded from 00_Setup")
print(f"Using catalog: {config['catalog_name']}, schema: {config['schema_name']}")
print(f"Storage path: {config['volume_path']}")

# COMMAND ----------

# MAGIC %md ##Step 1: Download Dataset Files
# MAGIC
# MAGIC In this step, we will download the dataset files to a directory accessible within the Databricks workspace:

# COMMAND ----------

# DBTITLE 1,Download Dataset Files
# Define files to download
files_to_download = [
    {"url": "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv", "output": "labels.csv"},
    {"url": "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv", "output": "products.csv"},
    {"url": "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv", "output": "queries.csv"}
]

# Download files directly to the volume using dbutils
for file in files_to_download:
    # Get the full path in the download directory
    output_path = f"{config['downloads_path']}/{file['output']}"
    
    print(f"Downloading {file['url']} to {output_path}")
    
    # Download the file content using requests
    response = requests.get(file['url'])
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Write the content to the file using dbutils, which works with Volumes paths directly
    dbutils.fs.put(output_path, response.text, overwrite=True)
    
    print(f"✅ Successfully downloaded {file['output']}")

# List the downloaded files
files = dbutils.fs.ls(config['downloads_path'])
print("\nDownloaded files:")
for file_info in files:
    print(f"- {file_info.name} ({file_info.size} bytes)")

# COMMAND ----------

# MAGIC %md ##Step 2: Write Data to Tables
# MAGIC
# MAGIC In this step, we will read data from each of the previously downloaded files and write the data to tables that will make subsequent access easier and faster:

# COMMAND ----------

# DBTITLE 1,Process Products
# Define schema for products including embedding_column
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('product_class', StringType()),
  StructField('category_hierarchy', StringType()),
  StructField('product_description', StringType()),
  StructField('product_features', StringType()),
  StructField('rating_count', FloatType()),
  StructField('average_rating', FloatType()),
  StructField('review_count', FloatType())
])

# Read products CSV and create embedding column in one step
products = (
    spark.read.csv(
        config['downloads_path'] + '/products.csv',
        sep='\t',
        header=True,
        schema=products_schema
    )
    .selectExpr(
        "*",
        # Use CASE to handle both NULL and empty string values
        """CASE 
            WHEN product_description IS NULL OR trim(product_description) = '' 
            THEN product_name 
            ELSE product_description 
        END as embedding_column"""
    )
)

# Write to Delta table with Unity Catalog
products_table_name = config['products_table']
(products
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema', 'true')
  .saveAsTable(products_table_name)
)

spark.sql(f"ALTER TABLE {products_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"✅ Enabled Change Data Feed on {products_table_name}")

# Display sample data to verify embedding column
print("Sample data showing embedding_column logic:")
display(
    spark.table(products_table_name)
    .select("product_id", "product_name", "product_description", "embedding_column")
    .limit(10)
)


# COMMAND ----------

# DBTITLE 1,Process Queries
# Define schema for queries
queries_schema = StructType([
  StructField('query_id', IntegerType()),
  StructField('query', StringType()),
  StructField('query_class', StringType())
])

# Read queries CSV
queries = (
  spark
    .read
    .csv(
      config['downloads_path'] + '/queries.csv',
      sep='\t',
      header=True,
      schema=queries_schema
    )
)

# Write to Delta table with Unity Catalog
queries_table_name = config['queries_table']
(queries
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema', 'true')
  .saveAsTable(queries_table_name)
)

# Display the table
display(spark.table(queries_table_name))

# COMMAND ----------

# DBTITLE 1,Process Labels
# Define schema for labels
labels_schema = StructType([
  StructField('id', IntegerType()),
  StructField('query_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('label', StringType())
])

# Read labels CSV
labels = (
  spark
    .read
    .csv(
      config['downloads_path'] + '/labels.csv',
      sep='\t',
      header=True,
      schema=labels_schema
    )
)

# Write to Delta table with Unity Catalog
labels_table_name = config['labels_table']
(labels
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema', 'true')
  .saveAsTable(labels_table_name)
)

# Display the table
display(spark.table(labels_table_name))

# COMMAND ----------

# MAGIC %md ##Step 3: Assign Label Scores
# MAGIC
# MAGIC To prepare the text-based labels assigned to products returned by a query for use in our algorithm, we'll convert the labels to numerical scores based our judgement of how these labels should be weighted:
# MAGIC
# MAGIC **NOTE** [This article](https://medium.com/@nikhilbd/how-to-measure-the-relevance-of-search-engines-18862479ebc) provides a nice discussion of how to approach the scoring of search results for relevance should you wish to explore alternative scoring patterns. 

# COMMAND ----------

# DBTITLE 1,Add Label Score Column to Labels Table
labels_table_name = config['labels_table']
if 'label_score' not in spark.table(labels_table_name).columns:
  _ = spark.sql(f'ALTER TABLE {labels_table_name} ADD COLUMN label_score FLOAT')

# COMMAND ----------

# DBTITLE 1,Assign Label Scores
# Using spark.sql() instead of %sql magic command for VSCode compatibility
labels_table_name = config['labels_table']
spark.sql(f"""
UPDATE {labels_table_name}
SET label_score = 
  CASE lower(label)
    WHEN 'exact' THEN 1.0
    WHEN 'partial' THEN 0.75
    WHEN 'irrelevant' THEN 0.0
    ELSE NULL
    END
""")

# Display the updated table with scores
display(spark.sql(f"SELECT * FROM {labels_table_name}"))

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
# MAGIC
# MAGIC
