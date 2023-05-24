# Databricks notebook source
# MAGIC %md The purpose of this notebook is to provide access to configuration data used by the Product Search solution accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

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
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_simple_architecture.png' width=800>
# MAGIC
# MAGIC As we explore this, it is important to recognize that you will need to be running in Databricks workspace that supports GPU-based clusters and the Databricks model serving feature.  The availabilty of GPU clusters is dependent upon your cloud provider and quotas assigned to your cloud subscription by that provider.  The avaialblity of Databricks model serving is currently limited to the following [AWS](https://docs.databricks.com/machine-learning/model-serving/index.html#limitations) and [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/#limitations) regions.
# MAGIC
# MAGIC **NOTE** Please note that for this solution, you can make use of a single-node cluster.  Be sure to select a GPU-enabled cluster (and a corresponding Databricks ML runtime).  Larger node sizes should give better performance for some of the more intensive steps.

# COMMAND ----------

# MAGIC %md ##Configuration
# MAGIC
# MAGIC The following parameters are used throughout the notebooks to control the resources being used.  If you modify these variables, please note that markdown in the notebooks may refer to the original values associated with these:

# COMMAND ----------

# DBTITLE 1,Initialize Config Variables
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
config['database'] = 'wands'

# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# DBTITLE 1,Storage
config['dbfs_path'] = '/wands'

# COMMAND ----------

# DBTITLE 1,Models
config['basic_model_name'] = 'wands_basic_search'
config['tuned_model_name'] = 'wands_tuned_search'

# COMMAND ----------

# DBTITLE 1,Databricks url and token
import os
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
config['databricks token'] = ctx.apiToken().getOrElse(None)
config['databricks url'] = ctx.apiUrl().getOrElse(None)
os.environ['DATABRICKS_TOKEN'] = config["databricks token"]
os.environ['DATABRICKS_URL'] = config["databricks url"]

# COMMAND ----------

# DBTITLE 1,mlflow experiment
import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/prod_search'.format(username))

# COMMAND ----------

# DBTITLE 1,Model serving endpoint
config['serving_endpoint_name'] = 'wands_search'

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
