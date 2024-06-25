# Kogitune 🦊 Distributed Dataset for LLM Development

Kogitune is a distributed dataset platform for building machine learning pipelines over wide area networks by separating dataset preprocessing and LLM pre-training.

## Overview

The performance of large language models (LLMs) relies on massive, high-quality, preprocessed datasets, often over hundreds of gigabytes. Maintaining such large datasets by a single organization is challenging. Therefore, a framework that promotes division of labor and collaboration among multiple organizations is necessary. We propose a new distributed dataset platform that separates dataset preprocessing and LLM pre-training, building ML pipelines over the Internet.

![](images/kogitune_concept-fs8.png)

Kogitune is a prototype implementation of this proposed distributed dataset platform. Its main features include the following, with some still in the conceptual stage:

- Various filters
- Data tensor creation (differential privacy)
- Fault tolerance & asynchronous downloads
- Dataset recipes
- Distributed datasets (compatible with PyTorch Dataset)

## Installation

To install Kogitune, use the following command:

```bash
pip3 install -U git+https://github.com/kuramitsulab/kogitune.git
```

## Data Tensors (Provider Side)

In Kogitune, preprocessed and tokenized datasets are called __data tensors__. By placing data tensors in an accessible location (local storage or a web server), they become available for GPU training.

### Creating Data Tensors

First, prepare cleaned and formatted text data in JSONL format. For this example, let's use `wiki2023_tiny.jsonl` with the following content:

```json
{"text": "OTRS stands for Open-source Ticket Request System. It is used by companies, organizations, and groups to handle individual inquiries and their responses..."}
```

The tokenizer is important for creating data tensors because different tokenizers will produce different data tensors from the same text data. Here, we'll use the tokenizer from the llm-jp project.

Run the following Python script to create the data tensor (this is equivalent to the `kogitune store` command):

```python
from kogitune.cli import store_cli
store_cli(files=['wiki2023_tiny.jsonl'], tokenizer='llm-jp/llm-jp-1.3b-v1.0')
```

Creating data tensors can take time depending on the amount of text. This time investment reduces the cost during GPU training. Once the data tensor is successfully created, it will be stored in a directory like:

`'llm-jp-915a/wiki2023_tiny'`

To make the data tensor accessible over the network, place this directory on your web server. If you don't want to make it public, restrict access to those who know the link.

## Dataset Recipes (Trainer Side)

Kogitune allows for the separation of data preprocessing and training. This means you can use data tensors created by someone else to form your final training dataset. Here, we'll continue with the example of the data tensor `'llm-jp-915a/wiki2023_tiny'` created on local storage.

Add the path of the data tensor to a list:

```python
url_list = [
    'llm-jp-915a/wiki2023_tiny',
    ...
]
```

Note: You can include multiple data tensors in the list, but they must all be preprocessed with the same tokenizer.

Use the `DatasetRecipe` class from the `kogitune.trainers` module to create the training dataset:

```python
from kogitune.trainers import DatasetRecipe

recipe = DatasetRecipe(url_list)
train_dataset = recipe.get_train_dataset()
```

The resulting training dataset is compatible with PyTorch's `Dataset` class. 
You can now integrate it into your existing training script. For example, to train with HuggingFace Trainer:

```python
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(...),
    args=...
)
```

Now, you just have to wait for the training to finish!

