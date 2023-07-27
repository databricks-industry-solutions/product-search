# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG",
            "accelerator": "prod_search"
        },
        "tasks": [
            {
                "job_cluster_key": "prod_search_cluster",
                "notebook_task": {
                    "notebook_path": f"00_Intro_and_Config"
                },
                "task_key": "prod_search_00"
            },
            {
                "job_cluster_key": "prod_search_cluster",
                "notebook_task": {
                    "notebook_path": f"01_Data_Prep"
                },
                "task_key": "prod_search_01",
                "depends_on": [
                    {
                        "task_key": "prod_search_00"
                    }
                ]
            },
            {
                "job_cluster_key": "prod_search_cluster",
                "notebook_task": {
                    "notebook_path": f"02_Define_Basic_Search"
                },
                "task_key": "prod_search_02",
                "depends_on": [
                    {
                        "task_key": "prod_search_01"
                    }
                ]
            },
            {
                "job_cluster_key": "prod_search_cluster",
                "notebook_task": {
                    "notebook_path": f"03_Fine_Tune_Model"
                },
                "task_key": "prod_search_03",
                "depends_on": [
                    {
                        "task_key": "prod_search_02"
                    }
                ]
            },
            {
                "job_cluster_key": "prod_search_cluster",
                "notebook_task": {
                    "notebook_path": f"04_Deploy_Model"
                },
                "task_key": "prod_search_04",
                "depends_on": [
                    {
                        "task_key": "prod_search_03"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "prod_search_cluster",
                "new_cluster": {
                    "spark_version": "12.2.x-gpu-ml-scala2.12",
                "spark_conf": {
                    "spark.master": "local[*, 4]",
                    "spark.databricks.cluster.profile": "singleNode",
                    "spark.databricks.delta.preview.enabled": "true"
                    },
                    "num_workers": 0,
                    "node_type_id": {"AWS": "g5.8xlarge", "MSA": "Standard_NC12s_v3"}, # this accelerator does not support GCP
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

# dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
# run_job = dbutils.widgets.get("run_job") == "True"
# nsc = NotebookSolutionCompanion()
# nsc.deploy_compute(job_json, run_job=run_job)

# COMMAND ----------

# MAGIC %md 
# MAGIC ============

# COMMAND ----------

from dbacademy.dbgems import get_cloud, get_notebook_dir, get_browser_host_name, get_notebooks_api_endpoint, get_notebooks_api_token

# COMMAND ----------

import json
import yaml
import sys
import hashlib
import os
import shutil
import subprocess
# ymal_string=yaml.dump(job_json, sys.stdout, default_flow_style=False, indent=2, sort_keys=False)

# COMMAND ----------

node_dict = {"AWS": "g5.8xlarge", "MSA": "Standard_NC12s_v3"}
node_type = node_dict[get_cloud()]
solacc_path = get_notebook_dir()
solution_code_name = solacc_path.split('/')[-1]
hash_code = hashlib.sha256(solacc_path.encode()).hexdigest()
job_name = f"[RUNNER] {solution_code_name} | {hash_code}" # use hash to differentiate solutions deployed to different paths


# COMMAND ----------

solacc_path

# COMMAND ----------

dbutils.fs.mkdirs('dbfs:/databricks/scripts')

dbutils.fs.put(
  '/databricks/scripts/product_search/bundle.yml',
  f'''
bundle:
  name: product_search_accelerator

environments:
  development:
    default: true
    workspace:
      host: {get_notebooks_api_endpoint()}
''', 
  True
  )

resources_string = f'''
resources:
  jobs:
    product_search:
      name: "{job_name}"
      timeout_seconds: 28800
      max_concurrent_runs: 1
      tags:
        usage: solacc_testing
        group: RCG
        accelerator: prod_search
      tasks:
      - job_cluster_key: prod_search_cluster
        notebook_task:
          notebook_path: {solacc_path}/00_Intro_and_Config
        task_key: prod_search_00
      - job_cluster_key: prod_search_cluster
        notebook_task:
          notebook_path: {solacc_path}/01_Data_Prep
        task_key: prod_search_01
        depends_on:
        - task_key: prod_search_00
      - job_cluster_key: prod_search_cluster
        notebook_task:
          notebook_path: {solacc_path}/02_Define_Basic_Search
        task_key: prod_search_02
        depends_on:
        - task_key: prod_search_01
      - job_cluster_key: prod_search_cluster
        notebook_task:
          notebook_path: {solacc_path}/03_Fine_Tune_Model
        task_key: prod_search_03
        depends_on:
        - task_key: prod_search_02
      - job_cluster_key: prod_search_cluster
        notebook_task:
          notebook_path: {solacc_path}/04_Deploy_Model
        task_key: prod_search_04
        depends_on:
        - task_key: prod_search_03
      job_clusters:
      - job_cluster_key: prod_search_cluster
        new_cluster:
          spark_version: 12.2.x-gpu-ml-scala2.12
          spark_conf:
            spark.master: local[*, 4]
            spark.databricks.cluster.profile: singleNode
            spark.databricks.delta.preview.enabled: 'true'
          num_workers: 0
          node_type_id: {node_type}
          custom_tags:
            usage: solacc_testing

environments:
  development:
    resources:
      jobs:
        product_search:
          schedule:
            quartz_cron_expression: 14 8 14 * * ?
            timezone_id: UTC
'''

# if subprocess.check_output(['pwd']) == b'/databricks/driver\n': 
#   dbutils.fs.put(
#   '/databricks/scripts/product_search/resources.yml',
#   resources_string,
#   True
#   )

# else:
  
#   dbutils.fs.put(
#     '/databricks/scripts/product_search/resources.yml',
#     resources_string.replace(".py", ""),
#     True
#   )

dbutils.fs.put(
'/databricks/scripts/product_search/resources.yml',
resources_string,
True
)

# COMMAND ----------

# DBTITLE 1,Setting up bricks CLI
# MAGIC %sh -e
# MAGIC cd /databricks/driver
# MAGIC wget https://github.com/databricks/cli/releases/download/v0.201.0/databricks_cli_0.201.0_linux_amd64.zip
# MAGIC unzip -o databricks_cli_0.201.0_linux_amd64.zip
# MAGIC ls

# COMMAND ----------

# DBTITLE 1,Verify that the bricks CLI works
# MAGIC %sh -e 
# MAGIC /databricks/driver/databricks version

# COMMAND ----------

# DBTITLE 1,Set up authentication for bricks CLI
os.environ['DATABRICKS_HOST'] = get_notebooks_api_endpoint()
os.environ['DATABRICKS_TOKEN'] = get_notebooks_api_token()

# COMMAND ----------

# DBTITLE 1,Verify that the bricks CLI authentication works
# MAGIC %sh -e 
# MAGIC /databricks/driver/databricks api get /api/2.0/clusters/spark-versions

# COMMAND ----------

shutil.copy2("/dbfs/databricks/scripts/product_search/bundle.yml", ".")

shutil.copy2("/dbfs/databricks/scripts/product_search/resources.yml", ".")
# assert os.system("""
# /databricks/driver/databricks bundle deploy
# """) == 0

# COMMAND ----------

# MAGIC %sh pwd

# COMMAND ----------

# MAGIC %sh ls

# COMMAND ----------

# MAGIC %sh cat resources.yml

# COMMAND ----------

# MAGIC %sh /databricks/driver/databricks bundle deploy

# COMMAND ----------

# MAGIC %sh /databricks/driver/databricks bundle validate

# COMMAND ----------

# if subprocess.check_output(['pwd']) == b'/databricks/driver\n': 
#   # if the notebook is in a repo but the cluster used is below the DBR threshold for `files in repos` in the workspace, e.g. 11+ for e2-demo-field-eng, `pwd` returns "/databricks/driver" - this threshold is set at a workspace level 
#   # we can write into the repo from this notebook if files in repo works and the cluster is 11.2+
#   # we cannot write into this repo and will need to set up a separate repo folder if `pwd` returns '/databricks/driver'
#   assert os.system("""
# cd /databricks/driver
# rm -rf bundle_deployment
# mkdir -p bundle_deployment
# cd bundle_deployment
# git clone https://github.com/databricks-industry-solutions/product-search.git
# cd product-search
# cp /dbfs/databricks/scripts/product_search/resources.yml .
# cp /dbfs/databricks/scripts/product_search/bundle.yml .
# /databricks/driver/bricks bundle deploy
# """) == 0

# else: # the pwd is your current wsfs location, e.g., `/Repos/..../<repo-name>/`, it means you can access the notebooks as files and run the bundle directly within this directory
#   # this currently does not work because DAB does not recognize notebooks in Databricks Repos as notebooks. The Repo feature strips the .py suffix of notebooks and DAB does not recognize them as notebooks

#   shutil.copy2("/dbfs/databricks/scripts/product_search/bundle.yml", ".")

#   shutil.copy2("/dbfs/databricks/scripts/product_search/resources.yml", ".")
#   assert os.system("""
#   /databricks/driver/bricks bundle deploy
#   """) == 0
    

# COMMAND ----------

# %sh -e
# cd /databricks/driver
# rm -rf bundle_deployment
# mkdir -p bundle_deployment
# cd bundle_deployment
# git clone https://github.com/databricks-industry-solutions/product-search.git
# cd product-search
# cp /dbfs/databricks/scripts/product_search/resources.yml .
# cp /dbfs/databricks/scripts/product_search/bundle.yml .
# /databricks/driver/bricks bundle deploy

# COMMAND ----------


