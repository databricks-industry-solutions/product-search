# Databricks notebook source
# MAGIC %md The purpose of this notebook is to fine-tune our model for use in the Product Search accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC Having demonstrated the basics of assembling a model and supporting data to enable a semantic search, we will now focus on fine-tuning the model.  During fine-tuning, the model is fit against a set of data specific to a particular domain, such as our product catalog.  The original knowledge accumulated by our model from its pre-training remains intact but is supplemented with information gleaned from the additional data provided.  Once the model has been tuned to our satisfaction, it is packaged and persisted just like as before.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install sentence-transformers==2.2.2 langchain==0.0.179 chromadb==0.3.25 typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation
import torch
from torch.utils.data import DataLoader

from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import numpy as np
import pandas as pd

import mlflow

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Estimate Baseline Model Performance
# MAGIC
# MAGIC In this first step, we'll retrieve the queries and the products returned with each from the WANDS dataset.  For each query-product combination, a numerical score assigned to each combination based on the perceived alignment of the product with the query is retrieved as well:

# COMMAND ----------

# DBTITLE 1,Get Search Results
# assemble product text relevant to search
search_pd = (
  spark   
    .table('products')
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
    .join(
      spark
        .table('labels'),
        on='product_id'
      )
    .join(
      spark
        .table('queries'),
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

# MAGIC %md We can then calculate the cosine similarity between the queries and products associated with them.  While we talk about similarity between embeddings as having to do with the distance between two vectors, cosine similarity refers to the angle separating to rays extending from the center of a space to the point identified by the vector (as if it were a coordinate). In a normalized vector space, this angle also captures the degree of similarity between to points:

# COMMAND ----------

# DBTITLE 1,Calculate Cosine Similarity Between Queries and Products
# determine cosine similarity for each query-product pair
original_cos_sim_scores = (
  util.pairwise_cos_sim(
    query_embeddings, 
    product_embeddings
    )
  )

# COMMAND ----------

# MAGIC %md Averaging these values gives us a sense of how close the queries are to the products in the original embedding space.  Please note that cosine similarity ranges from 0.0 to 1.0 with values improving as they approach 1.0:

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

# MAGIC %md ##Step 2: Fine-Tune the Model
# MAGIC
# MAGIC With a baseline measurement of the original model's performance in-hand, we can now fine-tune it using our annotated search result data.  We will start by restructuring our query results into a list of inputs as required by the model:

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
# MAGIC **NOTE** This step will run faster by scaling up the server used for your single-node cluster.

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

# MAGIC %md During model fitting, you will notice we are setting the model to perform just one pass (epoch) over the data.  We will actually see pretty sizeable improvements from this process, but we may wish to increase that value to get multiple passes if we want to explore getting more.  The setting for *warmup_steps* is just a common one used in this space.  Feel free to experiment with other values or take the default.

# COMMAND ----------

# MAGIC %md ##Step 3: Estimate Fine-Tuned Model Performance
# MAGIC
# MAGIC With our model tuned, we can assess it's performance just like we did before:

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

# MAGIC %md We can see from these results that with just a single pass over the data, we've brought the queries closer together with our products, tuning the model to the particulars of our data.  

# COMMAND ----------

# MAGIC %md ##Step 4: Persist Model for Deployment
# MAGIC
# MAGIC Just like before, we can package our tuned model with our data to enable its persistence (and eventual deployment).  The following steps are presented just as they are in the previous notebook with minor adjustments to separate our original assets from the tuned assets:

# COMMAND ----------

# DBTITLE 1,Get Product Text to Search
# assemble product text relevant to search
product_text_pd = (
  spark
    .table('products')
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
  ).toPandas()

# COMMAND ----------

# DBTITLE 1,Load Product Info for Use with Encoder
# assemble product documents in required format (id, text)
documents = (
  DataFrameLoader(
    product_text_pd,
    page_content_column='product_text'
    )
    .load()
  )

# COMMAND ----------

# DBTITLE 1,Load Model as HuggingFaceEmbeddings Object
# encoder path
embedding_model_path = f"/dbfs{config['dbfs_path']}/tuned_model"

# make sure path is clear
dbutils.fs.rm(embedding_model_path.replace('/dbfs','dbfs:'), recurse=True)

# reload model using langchain wrapper
tuned_model.save(embedding_model_path)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# COMMAND ----------

# DBTITLE 1,Generate Embeddings from Product Info
# chromadb path
chromadb_path = f"/dbfs{config['dbfs_path']}/tuned_chromadb"

# make sure chromadb path is clear
dbutils.fs.rm(chromadb_path.replace('/dbfs','dbfs:'), recurse=True)

# generate embeddings
vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path
  )

# persist vector db
vectordb.persist()

# COMMAND ----------

# DBTITLE 1,Define Wrapper Class for Model
class ProductSearchWrapper(mlflow.pyfunc.PythonModel):


  # define steps to initialize model
  def load_context(self, context):

    # import required libraries
    import pandas as pd
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma

    # retrieve embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=context.artifacts['embedding_model'])

    # retrieve vectordb contents
    self._vectordb = Chroma(
      persist_directory=context.artifacts['chromadb'],
      embedding_function=embedding_model
      )

    # set number of results to return
    self._max_results = 5


  # define steps to generate results
  # note: query_df expects only one query
  def predict(self, context, query_df):


    # import required libraries
    import pandas as pd

    # perform search on embeddings
    raw_results = self._vectordb.similarity_search_with_score(
      query_df['query'].values[0], # only expecting one value at a time 
      k=self._max_results
      )

    # get lists of of scores, descriptions and ids from raw results
    scores, descriptions, names, ids = zip(
      *[(r[1], r[0].page_content, r[0].metadata['product_name'], r[0].metadata['product_id']) for r in raw_results]
      )

    # reorganized results as a pandas df, sorted on score
    results_pd = pd.DataFrame({
      'product_id':ids,
      'product_name':names,
      'product_description':descriptions,
      'score':scores
      }).sort_values(axis=0, by='score', ascending=True)
    
    # set return value
    return results_pd

# COMMAND ----------

# DBTITLE 1,Identify Model Artifacts
artifacts = {
  'embedding_model': embedding_model_path.replace('/dbfs','dbfs:'), 
  'chromadb': chromadb_path.replace('/dbfs','dbfs:')
  }

print(
  artifacts
  )

# COMMAND ----------

# DBTITLE 1,Define Environment Requirements
import pandas
import langchain
import chromadb
import sentence_transformers

# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()

# define packages required by model
packages = [
  f'pandas=={pandas.__version__}',
  f'langchain=={langchain.__version__}',
  f'chromadb=={chromadb.__version__}',
  f'sentence_transformers=={sentence_transformers.__version__}'
  ]

# add required packages to environment configuration
conda_env['dependencies'][-1]['pip'] += packages

print(
  conda_env
  )

# COMMAND ----------

# DBTITLE 1,Persist Model
with mlflow.start_run() as run:

    mlflow.pyfunc.log_model(
        artifact_path='model', 
        python_model=ProductSearchWrapper(),
        conda_env=conda_env,
        artifacts=artifacts, # items at artifact path will be loaded into mlflow repository
        registered_model_name=config['tuned_model_name']
    )

# COMMAND ----------

# DBTITLE 1,Elevate to Production
client = mlflow.MlflowClient()

latest_version = client.get_latest_versions(config['tuned_model_name'], stages=['None'])[0].version

client.transition_model_version_stage(
    name=config['tuned_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# DBTITLE 1,Retrieve model from registry
model = mlflow.pyfunc.load_model(f"models:/{config['tuned_model_name']}/Production")

# COMMAND ----------

# DBTITLE 1,Test the Persisted Model
# construct search
search = pd.DataFrame({'query':['farmhouse dining room table']})

# call model
display(model.predict(search))

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
