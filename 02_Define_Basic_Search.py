# Databricks notebook source
# MAGIC %md The purpose of this notebook is to transform product information for use in the Product Search accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our data in place, we will now take an off-the-shelf model and apply it to perform product search. A key part of this work is the introduction of a vector database that our model will use during inference to rapidly search the product catalog.
# MAGIC
# MAGIC To understand the vector database, you first need to understand *embeddings*. An embedding is an array of numbers that indicate the degree to which a unit of text aligns with clusters of words frequently found together in a set of documents. The exact details as to how these numbers are estimated isn't terribly important here.  What is important is to understand that the mathematical distance between two embeddings generated through the same model tells us something about the similarity of two documents.  When we perform a search, the user's search phrase is used to generate an embedding and it's compared to the pre-existing embeddings associated with the products in our catalog to determine which ones the search is closest to.  Those closest become the results of our search.
# MAGIC
# MAGIC To facilitate the fast retrieval of items using embedding similarities, we need a specialized database capable of not only storing embeddings but enabling a rapid search against numerical arrays. The class of data stores that addresses these needs are called vector stores, and one of the most popular of these is a lightweight, file-system based, open source store called [Chroma](https://www.trychroma.com/).  
# MAGIC
# MAGIC In this notebook, we will download a pre-trained model, convert our product text to embeddings using this model, store our embeddings in a Chroma database, and then package the model and the database for later deployment behind a REST API.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install sentence-transformers==2.2.2 langchain==0.0.179 chromadb==0.3.25 typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sentence_transformers import SentenceTransformer

from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import mlflow

import pandas as pd

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Assemble Product Info
# MAGIC
# MAGIC In this first step, we need to assemble the product text data against which we intend to search.  We will use our product description as that text unless there is no description in which case we will use the product name.  
# MAGIC
# MAGIC In addition to the searchable text, we will provide product metadata, such as product ids and names, that will be returned with our search results:

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

display(product_text_pd)

# COMMAND ----------

# MAGIC %md ##Step 2: Convert Product Info into Embeddings
# MAGIC
# MAGIC We will now convert our product text into embeddings.  The instructions for converting text into an embedding is captured in a language model.  The [*all-MiniLM-L12-v2* model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) is a *mini language model* (in contrast to a large language model) which has been trained on a large, well-rounded corpus of input text for good, balanced performance in a variety of document search scenarios.  The benefit of the *mini* language model as compared to a *large* language is that the *mini* model generates a more succinct embedding structure that facilitates faster search and lower overall resource utilization.  Given the limited breadth of the content in a product catalog, this is the best option of our needs:

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
# download embeddings model
original_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# MAGIC %md To use our model with our vector store, we need to wrap it as a LangChain HuggingFaceEmbeddings object.  We could have had that object download the model for us, skipping the previous step, but if we had done that, future references to the model would trigger additional downloads.  By downloading it, saving it to a local path, and then having the LangChain object read it from that path, we are bypassing unnecessary future downloads:

# COMMAND ----------

# DBTITLE 1,Load Model as HuggingFaceEmbeddings Object
# encoder path
embedding_model_path = f"/dbfs{config['dbfs_path']}/embedding_model"

# make sure path is clear
dbutils.fs.rm(embedding_model_path.replace('/dbfs','dbfs:'), recurse=True)

# reload model using langchain wrapper
original_model.save(embedding_model_path)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# COMMAND ----------

# MAGIC %md Using our newly downloaded model, we can now generate embeddings.  We'll persist these to the Chroma vector database, a database that will allow us to retrieve vector data efficiently in future steps:

# COMMAND ----------

# DBTITLE 1,Reset Chroma File Store
# chromadb path
chromadb_path = f"/dbfs{config['dbfs_path']}/chromadb"

# make sure chromadb path is clear
dbutils.fs.rm(chromadb_path.replace('/dbfs','dbfs:'), recurse=True)

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

# DBTITLE 1,Generate Embeddings from Product Info
# define logic for embeddings storage
vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path
  )

# persist vector db to storage
vectordb.persist()

# COMMAND ----------

# MAGIC %md From a count of the vector database collection, we can see that every product entry from our original dataframe has been loaded:

# COMMAND ----------

# DBTITLE 1,Count Items in Vector DB
vectordb._collection.count()

# COMMAND ----------

# MAGIC %md We can also take a peek at one of the records in the database to see how our data has been transformed.  Details about our product id and product name, basically all the fields in the original dataframe not identified as the *document* are stored in the *Metadatas* field.  The text we identified as our *document* is visible in its original form through the *Documents* field and the embedding created from this is available through the *embeddings* field:

# COMMAND ----------

# DBTITLE 1,Examine a Vector DB record
rec= vectordb._collection.peek(1)

print('Metadatas:  ', rec['metadatas'])
print('Documents:  ', rec['documents'])
print('ids:        ', rec['ids'])
print('embeddings: ', rec['embeddings'])

# COMMAND ----------

# MAGIC %md ##Step 3: Demonstrate Basic Search Capability
# MAGIC
# MAGIC To get a sense of how our search will work, we can perform a similarity search on our vector database:

# COMMAND ----------

# DBTITLE 1,Perform Simple Search
vectordb.similarity_search_with_score("kid-proof rug")

# COMMAND ----------

# MAGIC %md Notice that while some of the results reflect key terms, such as *kid*, some do not.  This form of search is leveraging embeddings which understand that terms like *child*, *children*, *kid* and *kids* often are associated with each other. And while the exact term *kid* doesn't appear in every result, the presence of *children* indicates that at least one of the results is close in concept to what we are searching for.

# COMMAND ----------

# MAGIC %md ##Step 4: Persist Model for Deployment
# MAGIC
# MAGIC At this point, we have all the elements in place to build a deployable model.  In the Databricks environment, deployment typically takes place using [MLflow](https://www.databricks.com/product/managed-mlflow), which has the ability to build a containerized service from our model as one of its deployment patterns.  Generic Python models deployed with MLflow typically support a standard API with a *predict* method that's called for inference.  We will need to write a custom wrapper to map a standard interface to our model as follows:

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

# MAGIC %md The *load_context* of the previously defined wrapper class addresses the steps that need to take place at model initialization. Two of those steps make reference to artifacts within the model's context.  
# MAGIC
# MAGIC Artifacts are assets stored with the model as it is logged with MLflow.  Using keys assigned to these artifacts, those assets can be retrieved for utilization at various points in the model's logic. 
# MAGIC
# MAGIC The two artifacts needed for our model are the path to the saved model and the Chroma database, both of which were persisted to storage in previous steps.  Please note that these objects were saved to the *Databricks Filesystem* which MLflow understands how to reference.  As a result, we need to alter the paths to these items by replacing the local */dbfs* to *dbfs:*: 

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

# MAGIC %md Next, we need to identify the packages we need to insure are installed as our model is deployed.  These are:

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

# MAGIC %md Now we can persist our model to MLflow.  Notice that in this scenario, our embedding model and Chroma database are being loaded as artifacts and that our *python_model* is just the class definition that provides the logic for hydrating a model from those artifacts:

# COMMAND ----------

# DBTITLE 1,Persist Model to MLflow
with mlflow.start_run() as run:

    mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=ProductSearchWrapper(),
        conda_env=conda_env,
        artifacts=artifacts, # items at artifact path will be loaded into mlflow repository
        registered_model_name=config['basic_model_name']
    )

# COMMAND ----------

# MAGIC %md If we use the experiments UI (accessible by clicking the flask icon in the right-hand navigation of your workspace), we can access the details surrounding the model we just logged.  By expanding the folder structure behind the model, we can see the model and vector store assets loaded into MLflow:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_mlflow_artifacts.PNG'>

# COMMAND ----------

# MAGIC %md We can now elevate our model to production status.  This would typically be done through a careful process of testing and evaluation but for the purposes of this demo, we'll just programmatically push it forward:

# COMMAND ----------

# DBTITLE 1,Elevate to Production
client = mlflow.MlflowClient()

latest_version = client.get_latest_versions(config['basic_model_name'], stages=['None'])[0].version

client.transition_model_version_stage(
    name=config['basic_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md Loading our model, we can perform a simple test to see results from a sample search.  

# COMMAND ----------

# DBTITLE 1,Retrieve model from registry
model = mlflow.pyfunc.load_model(f"models:/{config['basic_model_name']}/Production")

# COMMAND ----------

# MAGIC %md If you are curious why we are constructing a pandas dataframe for our search term, please understand that this aligns with how data will eventually passed to our model when we host it in model serving.  The logic in our *predict* function anticipates this as well.
# MAGIC
# MAGIC Inferencing a single record can take approximately 50-300 ms, allowing the model to be served and used by a user-facing webapp. 

# COMMAND ----------

# DBTITLE 1,Test Persisted Model with Sample Search
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
