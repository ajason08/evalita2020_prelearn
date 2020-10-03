
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification

import torch
import numpy as np


from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from datasets import load_dataset
#train = pd.read_csv('pre_learn/train.tsv',sep='\t')
#dev = pd.read_csv('pre_learn/dev.tsv', sep='\t')
#
#train_conc = train.concept.values
#train_label = train.label.values
#
#dev_conc = dev.concept.values
#dev_label = dev.label.values





tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-uncased")
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-italian-xxl-uncased")# AutoModel

def encode(examples):
    return tokenizer(examples['concept'], examples['preconcept'], truncation=True, padding='max_length')
    #return tokenizer(examples['text'], examples['text (#1)'], truncation=True, padding='max_length')




def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision,recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

dataset = load_dataset('csv', data_files={'train':'pre_learn/train.tsv', 'validation':'pre_learn/dev.tsv'}, delimiter='\t')
dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
dataset = dataset.map(encode, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir='./pre_learn/results/10_epoch_model',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=200,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./pre_learn/results/10_epoch_model/logs',
    save_steps=200,
    logging_steps=200
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)
trainer.train()

trainer.evaluate()

#trainer.save_model()
model.save_pretrained('./pre_learn/results/10_epoch_model')
