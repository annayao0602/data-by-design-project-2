# DS 4320 Project 2: How Well Do LLMS "Understand?"

This repository contains a data pipeline jupyter notebook and markdown file, which can be run with mongoDB credentials to load the dataset. Additionally, there contains a press release with findings from the analysis pipeline code, a log file, and the press release visualization. To execute, install requirements in requirements.txt and connect to the mongoDB cluster to run pipeline code.

#### Name: Anna Yao
#### Computing ID: zzz2bx
#### DOI  [![DOI](https://zenodo.org/badge/1215393099.svg)](https://doi.org/10.5281/zenodo.19802758)
#### Press Release: [link to press release](https://github.com/annayao0602/data-by-design-project-2/blob/main/press_release.md)
#### Pipeline [link to pipeline files](https://github.com/annayao0602/data-by-design-project-2/blob/main/pipeline.ipynb)
#### License [MIT License](https://github.com/annayao0602/data-by-design-project-2/blob/main/LICENSE)

## Problem Definition
#### Initial general statement: 
Do LLMs understand intent?
#### Specific problem: 
Specific Question: Given a specific prompt, how well can a machine learning model classify a user's underlying intent?

#### Rationale for Refinement
The initial problem is too broad because "AI" encompasses many different algorithms, and "text" can range from tweets to novels. Attempting to build a universal detector is currently unfeasible due to the rapid evolution of generative models. By refining the scope to focus specifically on academic text and anchoring the analysis on how transformer models calculate token probabilities (which results in lower perplexity and burstiness compared to human writers), the problem becomes a quantifiable, supervised classification task. This allows for targeted feature engineering rather than relying on a black-box approach.

#### Project Motivation
The general problem is too broad and can cover a multitude of different model targets and types. Refining this to "Intent Classification" transforms a vague NLP challenge into a structured, quantifiable multi-class machine learning problem. This specific refinement allows for a database where messy chat logs are transformed into nested JSON structures containing specific taxonomy arrays, and a specified target of understanding user intention in ai prompt engineering.


#### Unboxing the Black Box - Do LLMs Understand Intent?
[link to press release](https://github.com/annayao0602/data-by-design-project-2/blob/main/press_release.md)

## Domain Exposition

#### Terminology

|Term|Definition|
|---|---|
|Intent Classification|A natural language processing task that categorizes text based on the underlying goal or purpose of the user (e.g., seeking facts vs. seeking advice).|
|Artificial Intelligence |A domain of technology that examines the mimicry of human intelligence in technology|
|NLP|Natural Language Processing, a type of AI Model usage in which a model can learn to take in language in the form of vectors|
|LLM|Large Language Model, a type of AI model that is built on vast amounts of data created to generate and take in text|
|AI Alignment|bridging the gap between how AI models are currently tested and what real-world users actually need|
|Confidence Score|The probability output by the ML classifier for its chosen prediction. The inverse of this score serves as the quantified metric of mathematical uncertainty.|

#### Domain of Project
This project exists at the intersection of Natural Language Processing (NLP) and  Artificial Intelligence detection. Large Language Models (LLMs) increasingly become primary engines for public information retrieval, understanding how humans interact with these systems is paramount. This domain specifically focuses on "AI Alignment"—bridging the gap between how AI models are currently tested (benchmarked on rigid, academic atmospheric science) and what real-world users actually need (actionable advice, policy reasoning, and operational support). By analyzing real human-to-AI interaction logs and classifying the psychological intent behind them, this field seeks to build more responsive, user-aligned AI systems.

#### Background reading 
|Title|Description|Link|
|---|---|---|
|LLM Benchmark-User Need Misalignment for Climate Change|The foundational paper for this project. It introduces the WildChat dataset and proves the misalignment between AI climate benchmarks and real-world user intent.|https://myuva-my.sharepoint.com/:b:/g/personal/zzz2bx_virginia_edu/IQDYixJlXrf9Qo2yGdf85TRBAdAo58ZlnphOmaUFqY4NnE0?e=duz4cf|
|TEXT CLASSIFICATION: A PERSPECTIVE OF DEEP LEARNING METHODS|A comprehensive breakdown of how models conduct NLP|https://myuva-my.sharepoint.com/:b:/g/personal/zzz2bx_virginia_edu/IQAGYRfaYfywRprZYR28jO8HAWnp7UPQyqbtUa2IFz7l06I?e=fsKfb7|
|User Intent Recognition and Satisfaction with Large Language Models: A User Study with ChatGPT|A study on user intent and how it applies to large language models|hw9-background-readings/2402.02136v2.pdf|
|Beyond Context: Large Language Models Failure to Grasp Users Intent|position paper on the failure of AI models to capture user intent|https://myuva-my.sharepoint.com/:b:/g/personal/zzz2bx_virginia_edu/IQDFfdI9OAAVSqzNxcMhu_A5AZnR-6RV1tm_B_I5UIHDv14?e=CQvE2N|
|Towards Intent-based User Interfaces: Charting the Design Space of Intent-AI Interactions Across Task Types|Paper examining different fixed scope tasks such as news header generation and how to bridge the gap between Ai and human intent|https://myuva-my.sharepoint.com/:b:/g/personal/zzz2bx_virginia_edu/IQA8Fh-hgZp-Q4M20BbjcV2qAellh2IgVfb5YZP53MJEsgE?e=bhTHB5|

## Data Creation 
The raw data for this project was acquired from the open-source platform Hugging Face, specifically the Westing/LLM-Misalign-Climate-Change dataset. This dataset is a curated aggregation of climate-related inquiries and knowledge provision, capturing both human-to-AI interactions (extracted from sources like WildChat and LMSYS-Chat-1M) and human-to-human interactions (sourced from climate-related Reddit communities).

To acquire and process this data, a programmatic pipeline was established using Python. The Hugging Face datasets library was utilized to download the raw dataset into memory. Because the dataset contains 14 different languages and complex nested structures (such as arrays for topics and dictionaries for question types), it was natively formatted as JSON-like documents. The data was processed dynamically in Python to isolate English-only text and then ingested directly into a MongoDB Atlas cluster, taking advantage of MongoDB's document model to preserve the nested schema without requiring relational flattening.

#### Code
|Code|Description|Link|
|---|---|---|
|data_acquisition.ipynb|Extracted from hugging face, isolated only English text data|https://github.com/annayao0602/data-by-design-project-2/blob/main/pipeline.ipynb|

#### Bias Identification
Source/Selection Bias: The original data is sourced from LLM prompt logs (like WildChat) and Reddit. These platforms heavily skew toward younger, tech-savvy, and predominantly Western demographics. Rural populations or those without internet access—who are often the most impacted by climate change—are entirely unrepresented.

Language Bias: During the data acquisition phase, a deliberate programmatic filter was applied to drop all non-English text. Because climate change is a global crisis, removing 13 other languages completely erases the perspectives, priorities, and localized concerns of non-English-speaking populations for the sake of model interpretability.

#### Bias Mitigation
While the inherent demographic bias of Reddit and LLM users cannot be retroactively fixed, it can be accounted for by explicitly defining the scope of the analysis (e.g., stating the analysis reflects Western internet-user sentiment rather than global human sentiment). To quantify and handle the language bias introduced during processing, one mitigation strategy is to run a comparative query prior to dropping the non-English data. By comparing the Final_Topics distribution of the English subset against the dropped multilingual subset, I can statistically quantify exactly which climate topics (e.g., "A4. Extreme Weather Events" vs "E1. Climate Policy") are being underrepresented in the final English-only database.

#### Rationale
Filtering for English Only
A critical judgement call was made to restrict the database to English text. The rationale was to standardize the dataset for downstream Natural Language Processing (NLP) and sentiment analysis, which typically require language-specific models. However, this introduces significant uncertainty regarding the global validity of any conclusions drawn from the data, as international climate concerns are excluded.

Choosing a climate based dataset
Since this is a niche field, the output of results could have some uncertainty in terms of generalizability. However, this also allows the model to give more consistent predictions surrounding the same topic.

## Metadata 
#### Implicit Schema
Required Fields: Every document must contain an id (String) and text (String). Documents missing these fields should be rejected or flagged during insertion.

Array Handling: The Final_Topics field must always be treated as an array of strings. If a document only has one topic, it must still be enclosed in an array (e.g., ["A1. Atmospheric Science"]). Queries should utilize array-specific operators like $in or $all.

Nested Document Parsing: The Question Type field is an embedded document containing Intent and Form sub-fields. Application code must use dot notation (e.g., "Question Type.Intent") to query these deeply, and should implement null-checks (e.g., .get('Intent', [])) as some documents may lack complete classification.

#### Data Summary
|Attribute|Detail|
|---|---|
|Database name|climate_database|
|Collection name|misaligned_queries|
|Primary domain|Climate change inquiries, user intents, requested answer formats|
|Data structure|Nested JSON documents|
|Total documents|40,000+|

#### Data Dictionary
### Item 3. Data Dictionary

| Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | String | A unique alphanumeric identifier for the specific text snippet. | `"WildChat_bc25ae4c..."` |
| `text` | String | The raw text of the user query or discussion post regarding climate change. | `"satellite monitoring for air quality"` |
| `Final_Topics` | Array (String) | A standardized list of climate categories assigned to the text. | `["A6. Environmental Monitoring"]` |
| `Question Type` | Object | An embedded document containing the structural and intentional classification of the text. | `{"Intent": ["..."], "Form": ["..."]}` |
| `Question Type.Intent` | Array (String) | The classified purpose of the user's prompt (e.g., Fact Lookup, Reasoning). | `["INTENT_1b. Concept Definition"]` |
| `Question Type.Form` | Array (String) | The requested output format of the response. | `["FORM_2a. Concise Paragraph"]` |

#### Quantifying Uncertainty
The raw dataset consists entirely of textual and categorical data; there are no native numerical features.