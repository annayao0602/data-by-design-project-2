# Pipeline

## Data Acquisition

```python
import os
import sys
import re
import logging
import certifi
import pymongo
from langdetect import detect
from dotenv import load_dotenv
from datasets import load_dataset
from collections import Counter

#logging
logging.basicConfig(
    filename='pipeline.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log, log_err = logging.info, logging.error


#use environment variable for access to mongodb via vscode, secrets via colab
try:
    if 'google.colab' in sys.modules:
        from google.colab import userdata
        db_password = userdata.get('MONGO_PASSWORD')
    else:
        load_dotenv()
        db_password = os.getenv("MONGO_PASSWORD")

    if not db_password:
        raise ValueError("MONGO_PASSWORD not found.")

    client = pymongo.MongoClient(
        f"mongodb+srv://annayao0602:{db_password}@cluster0.qayglik.mongodb.net/",
        tlsCAFile=certifi.where()
    )
    collection = client.climate_database.misaligned_queries
except Exception as e:
    log_err(f"Initialization Failed: {e}")
    sys.exit(1)

#cleaning steps: filtering for only english data, and cleaning intent labels
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def clean_intent(intent_string):
    cleaned = re.split(r' — | - | \(', intent_string).strip()
    return cleaned.replace(' / Empathy', '')

def process_row(row):
    """Handles both intent cleaning and extraction in one pass."""
    raw_intents = row.get("Question Type", {}).get("Intent", [])
    row["intents"] = [clean_intent(i) for i in raw_intents]
    return row

try:
    log("Downloading and filtering dataset...")
    ds = load_dataset("Westing/LLM-Misalign-Climate-Change", split="train")

    ds = ds.filter(lambda x: is_english(x["text"]))

    ds = ds.map(process_row)

    all_intents = [intent for sublist in ds["intents"] for intent in sublist]
    counts = Counter(all_intents)
    valid_labels = {label for label, count in counts.items() if count > 50}

    log(f"Labels kept: {len(valid_labels)} | Dropped: {len(counts) - len(valid_labels)}")

  #filter rare and empty labels
    def filter_rare_and_empty(row):
        row["intents"] = [i for i in row["intents"] if i in valid_labels]
        return len(row["intents"]) > 0

    ds = ds.filter(filter_rare_and_empty)

    log(f"Syncing {len(ds)} documents to MongoDB...")
    collection.delete_many({})
    collection.insert_many(list(ds))
    log("Success: Database synced and cleaned.")

except Exception as e:
    log_err(f"Pipeline Error: {e}")

sample = collection.find_one()
if sample:
    log(f"Sample Verified - ID: {sample.get('id')} | Intents: {sample.get('intents')}")
else:
    log_err("Verification failed: Collection is empty.")


from collections import Counter

#purging any remaining labels that may not have enough data to prevent class imbalance
try:
    log("Starting final database purge of rare labels...")

    raw_docs = list(collection.find({}))
    all_intents = [i for doc in raw_docs for i in doc.get('intents', [])]
    counts = Counter(all_intents)

    valid_labels = {label for label, count in counts.items() if count > 50}
    log(f"Valid labels (>50): {len(valid_labels)}")

    cleaned_docs = []
    for doc in raw_docs:
        doc['intents'] = [i for i in doc.get('intents', []) if i in valid_labels]

        if doc['intents']:
            cleaned_docs.append(doc)

    collection.delete_many({})
    collection.insert_many(cleaned_docs)

    log(f"Purge complete. Documents retained: {len(cleaned_docs)}")

except Exception as e:
    log_err(f"Database clean-up failed: {e}")

```
## Check final label count

```python
import pandas as pd

try:
    pipeline = [
        {"$unwind": "$intents"},

        {"$group": {
            "_id": "$intents",
            "count": {"$sum": 1}
        }},

        {"$sort": {"count": -1}}
    ]

    results = list(collection.aggregate(pipeline))

    if not results:
        log("No data found in collection to count.")
    else:
        log("\n--- Current Label Counts ---")
        for entry in results:
            log(f"{entry['_id']}: {entry['count']}")

        counts_df = pd.DataFrame(results).rename(columns={"_id": "Intent", "count": "Frequency"})
        print(counts_df.to_string(index=False))

except Exception as e:
    log_err(f"Failed to retrieve label counts: {e}")

                                Intent  Frequency
                      INTENT_9z. Others      18950
      INTENT_4e. Information Extraction      12229
 INTENT_2a. Reasoning / Causal Analysis       6015
                 INTENT_1a. Fact Lookup       3447
INTENT_1c. Clarification / Verification       2108
          INTENT_1b. Concept Definition       1533
              INTENT_3a. General Advice       1166
         INTENT_3c. Planning / Strategy        631
         INTENT_2c. Evaluation / Review        429
         INTENT_6a. Operational Writing        236
 INTENT_2b. Data Analysis / Calculation        114
       INTENT_5c. Hypothetical Scenario        100
                INTENT_5a. General Text         98
                     INTENT_4b. Rewrite         75
   INTENT_3d. Teaching / Skill Building         72
               INTENT_4c. Summarisation         65
```
## Model Creation
```python
import os
import sys
import pymongo
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset

# code chunk to start model creation, using multilabelbinarizer on intents
try:
    if 'google.colab' in sys.modules:
        from google.colab import userdata
        db_password = userdata.get('MONGO_PASSWORD')
    else:
        from dotenv import load_dotenv
        load_dotenv()
        db_password = os.getenv("MONGO_PASSWORD")

    if not db_password:
        raise ValueError("MongoDB password not found in environment/secrets.")
        log_err("MongoDB password not found in environment/secrets")

    uri = f"mongodb+srv://annayao0602:{db_password}@cluster0.qayglik.mongodb.net/"
    client = pymongo.MongoClient(uri, tlsCAFile=certifi.where())
    collection = client.climate_database.misaligned_queries

    pipeline = [
        {"$match": {"intents": {"$exists": True, "$ne": []}}},

        {"$project": {
            "_id": 0,
            "text": 1,
            "intents": 1
        }}
    ]

    raw_data = list(collection.aggregate(pipeline))

    if not raw_data:
        log_err("No valid data found in MongoDB")
        raise ValueError("No valid data found in MongoDB matching the query.")

    df = pd.DataFrame(raw_data)

    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(df['intents'])

    label_names = mlb.classes_
    df['labels'] = binary_matrix.tolist()
    df = df.drop(columns=['intents'])

    hf_dataset = Dataset.from_pandas(df)

    log(f"Successfully formatted {len(hf_dataset)} documents.")
    log(f"Discovered {len(label_names)} labels: {label_names}")
    print(label_names)
    print(len(label_names))

except pymongo.errors.PyMongoError as db_err:
    log_err(f"Database connection or query failed: {db_err}")
except KeyError as key_err:
    log_err(f"Data structure error - missing expected field: {key_err}")
except Exception as e:
    log_err(f"An unexpected error occurred: {e}")

    ['INTENT_1a. Fact Lookup' 'INTENT_1b. Concept Definition'
    'INTENT_1c. Clarification / Verification'
    'INTENT_2a. Reasoning / Causal Analysis'
    'INTENT_2b. Data Analysis / Calculation' 'INTENT_2c. Evaluation / Review'
    'INTENT_3a. General Advice' 'INTENT_3c. Planning / Strategy'
    'INTENT_3d. Teaching / Skill Building' 'INTENT_4b. Rewrite'
    'INTENT_4c. Summarisation' 'INTENT_4e. Information Extraction'
    'INTENT_5a. General Text' 'INTENT_5c. Hypothetical Scenario'
    'INTENT_6a. Operational Writing' 'INTENT_9z. Others']
    16
```
### Analysis Rationale
I decided to use the RoBERTa model to train intent classification due to its ability to excel in text classification and optimization off of the original BERT model.

```python
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Sequence, Value


log("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

#tokenize text
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    if "labels" in examples:
        outputs["labels"] = [
            [float(v) for v in label] for label in examples["labels"]
        ]

    return outputs

log("Tokenizing the dataset...")
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.cast_column(
    "labels", Sequence(Value("float32"))
)
log("Splitting data into train and test sets...")
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

train_labels_array = np.array(train_dataset['labels'])

total_samples = len(train_labels_array)
positive_counts = np.sum(train_labels_array, axis=0)
negative_counts = total_samples - positive_counts

pos_weights = negative_counts / (positive_counts + 1e-5)
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)

log(f"Calculated pos_weights shape: {pos_weights_tensor.shape}")
print(f"Calculated pos_weights shape: {pos_weights_tensor.shape}")

log(f"Training on {len(train_dataset)} examples, Evaluating on {len(eval_dataset)} examples.")
print(f"Training on {len(train_dataset)} examples, Evaluating on {len(eval_dataset)} examples.")

#compute probabilities and metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities > 0.5).astype(int)

    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {'f1': f1, 'accuracy': acc}

log("Loading the base RoBERTa model...")
num_labels = len(label_names)

from torch import nn

#create custom trainer that weights based on classes
class WeightedTrainer(Trainer):
    def __init__(self, pos_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        self.pos_weights = self.pos_weights.to(logits.device)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

#Set up training arguments
try:
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    log("Setting up training configurations...")
    training_args = TrainingArguments(
      output_dir="./climate_intent_model",
      eval_strategy="epoch",
      save_strategy="epoch",
      learning_rate=2e-5,
      per_device_train_batch_size=32,
      per_device_eval_batch_size=64,
      num_train_epochs=3,
      weight_decay=0.01,
      load_best_model_at_end=True,
      logging_steps=50,
      fp16=True,
      dataloader_num_workers=4,
      dataloader_pin_memory=True,
      group_by_length=True,
  )

    from transformers import DataCollatorWithPadding

    class MultiLabelCollator(DataCollatorWithPadding):
        def __call__(self, features):
            batch = super().__call__(features)
            batch["labels"] = batch["labels"].float()
            return batch

    trainer = WeightedTrainer(
        pos_weights=pos_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=MultiLabelCollator(tokenizer),
    )

    log("Starting model training! This will take some time depending on your GPU...")
    trainer.train()

    log("Training complete. Saving the final model...")
    trainer.save_model("./climate_intent_model_final")
    tokenizer.save_pretrained("./climate_intent_model_final")

    log("Pipeline finished successfully! Your model is ready.")

except Exception as e:
    log_err(f"A critical error occurred during model training: {e}")
for label, score in sorted(zip(label_names, per_label_f1), key=lambda x: -x):
    print(f"{score:.4f}  {label}")

0.9389  INTENT_9z. Others
0.9025  INTENT_4e. Information Extraction
0.7324  INTENT_2a. Reasoning / Causal Analysis
0.4975  INTENT_1a. Fact Lookup
0.4665  INTENT_1c. Clarification / Verification
0.3701  INTENT_1b. Concept Definition
0.1964  INTENT_3a. General Advice
0.1633  INTENT_5a. General Text
0.1501  INTENT_3c. Planning / Strategy
0.1370  INTENT_6a. Operational Writing
0.1316  INTENT_4b. Rewrite
0.1188  INTENT_2c. Evaluation / Review
0.0571  INTENT_3d. Teaching / Skill Building
0.0563  INTENT_4c. Summarisation
0.0403  INTENT_2b. Data Analysis / Calculation
0.0118  INTENT_5c. Hypothetical Scenario
```
### Visualization Rationale
I chose to use a bar chart showcasing the top 5 F1 score labels for its simplicity and publication like potential. Since F1 is a good metric of model performance, it is easy for users to understand by equating it to to how well the model is doing.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score


def short_label(name):
    clean_name = name.replace('INTENT_', '')
    if '. ' in clean_name:
        return clean_name.split('. ', 1)[-1]
    return clean_name

ranked = [(f, l) for f, l in sorted(zip(per_label_f1, label_names), reverse=True)
          if 'Others' not in l]
top5 = ranked[:5]
bot5 = ranked[-5:]

def plot_subset(data, title, subtitle, color, filename):
    f1s    = [f for f, _ in data]
    labels = [short_label(l) for _, l in data]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, f1s, color=color, height=0.6)

    for bar, val in zip(bars, f1s):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=11)

    ax.text(0.5, 1.10, title, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='center')
    ax.text(0.5, 1.00, subtitle, transform=ax.transAxes,
            fontsize=12, color='dimgray', ha='center')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold', labelpad=10)
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=11)

    # Save the figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

plot_subset(
    data=top5,
    title='Model Performance (By Label)',
    subtitle='Performance of the top 5 user intent categories',
    color='#2D6A4F',
    filename='top5_f1_publication.png'
)
```
![image](f1_image.png)