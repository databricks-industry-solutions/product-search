![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## LLM Product Search Accelerator

The purpose of this solution accelerator is to show how large language models (LLMs) and their smaller brethren can be used to enable product search.  Unlike product search used in most sites today that rely upon keyword matches, LLMs enable what is commonly referred to as a semantic search where the *conceptual similarities* in words come into play.

A model's knowledge of the *conceptual similarity* between words comes from being exposed to a wide range of documents and from those documents learning that certain words tend to have close relationships to one another.  For example, one document may discuss the importance of play for *children* and use the term *child* teaching the model that *children* and *child* have some kind of relationship.  Other documents may use these terms in similar proximity and other documents discussing the same topics may introduce the term *kid* or *kids*.  It's possible that in some documents all four terms pop-up but even if that never happens, there may be enough overlap in the words surrounding these terms that the model comes to recognize a close association between all these terms.

Many of the LLMs available from the open source community come available  as pre-trained models where these word associations have already been learned from a wide range of publicly available  information. With the knowledge these models have already accumulated, they can be used to search the descriptive text for products in a product catalog for items that seem aligned with a search term or phrase supplied by a user. Where the products featured on a site tend to use a more specific set of terms that have their own patterns of association reflecting the tone and style of the retailer or the suppliers they feature, these models can be exposed to additional data specific to the site to shape its understanding of the language being used.  This *fine-tuning* exercise can be used to tailor an off-the-shelf model to the nuances of a specific product catalog, enabling even more effective search results.

In this solution accelerator, we will show both versions of this pattern using an off-the-shelf model and one tuned to a specific body of product text. We'll then tackle the issues related to model deployment so that users can see how a semantic search capability can easily be deployed through their Databricks environment.

___
<tim.lortz@databricks.com> 
<mustafaali.sezer@databricks.com> 
<peyman@databricks.com>
<bryan.smith@databricks.com>
___


<img src='https://github.com/databricks-industry-solutions/product-search/raw/main/images/inference.png' width=800>

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
|  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
| langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
| chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
| sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
