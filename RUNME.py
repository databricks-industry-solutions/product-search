# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow** to see how this solution accelerator executes: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `resources.yml` definition still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from dbacademy.dbgems import get_cloud, get_notebook_dir, get_browser_host_name, get_notebooks_api_endpoint, get_notebooks_api_token
import json
import hashlib
import os

# # For upgrading from json configurations, use this line below for translating json to yaml
# import yaml
# import sys
# ymal_string=yaml.dump(job_json, sys.stdout, default_flow_style=False, indent=2, sort_keys=False)

# COMMAND ----------

# DBTITLE 1,Gather environmental information
# get node type
node_dict = {"AWS": "g5.8xlarge", "MSA": "Standard_NC12s_v3"}
cloud = get_cloud()
node_type = node_dict[cloud]

# get solacc path
solacc_path = get_notebook_dir()

# get job name
solution_code_name = solacc_path.split('/')[-1]
hash_code = hashlib.sha256(solacc_path.encode()).hexdigest()
job_name = f"[RUNNER] {solution_code_name} | {hash_code}" # use hash to differentiate solutions deployed to different paths

# COMMAND ----------

# DBTITLE 1,Set up env vars
# Set up vars for the bundle

os.environ['BUNDLE_VAR_node_type'] = node_type
os.environ['BUNDLE_VAR_solacc_path'] = solacc_path
os.environ['BUNDLE_VAR_job_name'] = job_name

# Set up authentication for bricks CLI

os.environ['DATABRICKS_HOST'] = get_notebooks_api_endpoint()
os.environ['DATABRICKS_TOKEN'] = get_notebooks_api_token()

# COMMAND ----------

# DBTITLE 1,Setting up bricks CLI
# MAGIC %sh -e
# MAGIC cd /databricks/driver
# MAGIC wget https://github.com/databricks/cli/releases/download/v0.201.0/databricks_cli_0.201.0_linux_amd64.zip
# MAGIC unzip -o databricks_cli_0.201.0_linux_amd64.zip
# MAGIC ls
# MAGIC /databricks/driver/databricks version
# MAGIC /databricks/driver/databricks api get /api/2.0/clusters/spark-versions

# COMMAND ----------

# MAGIC %sh /databricks/driver/databricks bundle validate

# COMMAND ----------

# MAGIC %sh /databricks/driver/databricks bundle deploy

# COMMAND ----------


