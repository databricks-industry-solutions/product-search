# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data for the Product Search solution accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will access the [Wayfair Annotation Dataset (WANDS)](https://www.aboutwayfair.com/careers/tech-blog/wayfair-releases-wands-the-largest-and-richest-publicly-available-dataset-for-e-commerce-product-search-relevance), made accessible by [Wayfair](https://www.wayfair.com/) under an MIT License.
# MAGIC
# MAGIC The dataset consists of three file types:
# MAGIC </p>
# MAGIC
# MAGIC * Product - 42,000+ products features on the Wayfair website
# MAGIC * Query - 480 customer queries used for product searches
# MAGIC * Label - 233,000+ product results for the provided queries labeled for relevance
# MAGIC
# MAGIC In the [Annotations Guidelines document](https://github.com/wayfair/WANDS/blob/main/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf) that accompanies the dataset, Wayfair addresses the methods by which queries were labeled.  The three labels assigned to any query result are:
# MAGIC </p>
# MAGIC
# MAGIC * Exact match - this label represents the surfaced product fully matches the search query
# MAGIC * Partial match - this label represents the surfaced product does not fully match the search query
# MAGIC * Irrelevant - this label indicates the product is not relevant to the query
# MAGIC
# MAGIC As explained in the document, there is a bit of subjectivity in assigning these labels but the goal here is not to capture ground truth but instead to capture informed human judgement.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn

import os

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Download Dataset Files
# MAGIC
# MAGIC In this step, we will download the dataset files to a directory accessible within the Databricks workspace:

# COMMAND ----------

# DBTITLE 1,Set Path Variable for Script
os.environ['WANDS_DOWNLOADS_PATH'] = '/dbfs'+ config['dbfs_path'] + '/downloads' 

# COMMAND ----------

# DBTITLE 1,Download Dataset Files
# MAGIC %sh 
# MAGIC
# MAGIC # delete any old copies of temp data
# MAGIC rm -rf $WANDS_DOWNLOADS_PATH
# MAGIC
# MAGIC # make directory for temp tiles
# MAGIC mkdir -p $WANDS_DOWNLOADS_PATH
# MAGIC
# MAGIC # move to temp directory
# MAGIC cd $WANDS_DOWNLOADS_PATH
# MAGIC
# MAGIC # download datasets
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv
# MAGIC
# MAGIC # show folder contents
# MAGIC pwd
# MAGIC ls -l

# COMMAND ----------

# MAGIC %md ##Step 2: Write Data to Tables
# MAGIC
# MAGIC In this step, we will read data from each of the previously downloaded files and write the data to tables that will make subsequent access easier and faster:

# COMMAND ----------

# DBTITLE 1,Process Products
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

_ = (
  spark
    .read
      .csv(
        path='dbfs:/wands/downloads/product.csv',
        sep='\t',
        header=True,
        schema=products_schema
        )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('products')
  )

display(
  spark.table('products')
  )

# COMMAND ----------

# DBTITLE 1,Process Queries
queries_schema = StructType([
  StructField('query_id', IntegerType()),
  StructField('query', StringType()),
  StructField('query_class', StringType())
  ])

_ = (
  spark
    .read
    .csv(
      path='dbfs:/wands/downloads/query.csv',
      sep='\t',
      header=True,
      schema=queries_schema
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('queries')
  )

display(
  spark.table('queries')
  )

# COMMAND ----------

# DBTITLE 1,Process Labels
labels_schema = StructType([
  StructField('id', IntegerType()),
  StructField('query_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('label', StringType())
  ])

_ = (
  spark
    .read
    .csv(
      path='dbfs:/wands/downloads/label.csv',
      sep='\t',
      header=True,
      schema=labels_schema
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('labels')
  )

display(spark.table('labels'))

# COMMAND ----------

# MAGIC %md ##Step 3: Assign Label Scores
# MAGIC
# MAGIC To prepare the text-based labels assigned to products returned by a query for use in our algorithm, we'll convert the labels to numerical scores based our judgement of how these labels should be weighted:
# MAGIC
# MAGIC **NOTE** [This article](https://medium.com/@nikhilbd/how-to-measure-the-relevance-of-search-engines-18862479ebc) provides a nice discussion of how to approach the scoring of search results for relevance should you wish to explore alternative scoring patterns. 

# COMMAND ----------

# DBTITLE 1,Add Label Score Column to Labels Table
if 'label_score' not in spark.table('labels').columns:
  _ = spark.sql('ALTER TABLE labels ADD COLUMN label_score FLOAT')

# COMMAND ----------

# DBTITLE 1,Assign Label Scores
# MAGIC %sql
# MAGIC
# MAGIC UPDATE labels
# MAGIC SET label_score = 
# MAGIC   CASE lower(label)
# MAGIC     WHEN 'exact' THEN 1.0
# MAGIC     WHEN 'partial' THEN 0.75
# MAGIC     WHEN 'irrelevant' THEN 0.0
# MAGIC     ELSE NULL
# MAGIC     END;

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
