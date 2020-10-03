
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




bert_base_small_data = "dbmdz/bert-base-italian-uncased" #13G
#bert_base_large_data = "dbmdz/bert-base-italian-xxl-uncased" #81G
tokenizer = AutoTokenizer.from_pretrained(bert_base_small_data)
model = AutoModelForSequenceClassification.from_pretrained(bert_base_small_data)# AutoModel

#for param in model.base_model.parameters():
#    param.requires_grad = False

def encode(examples):
    #return tokenizer(examples['concept'], examples['preconcept'], truncation=True, padding='max_length')
    #return tokenizer(examples['text'], examples['text (#1)'], truncation=True, padding='max_length')
    return tokenizer(examples['concept_text'], examples['preconcept_text'], truncation=True, padding='max_length')



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

dataset = load_dataset('csv', data_files={'train':'pre_learn/crossdomain_splits/geometry_out.tsv', 'validation':'pre_learn/crossdomain_splits/geometry_only.tsv'}, delimiter='\t')
dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
dataset = dataset.map(lambda examples: {'concept_text': [x+' '+y for x,y in zip(examples['concept'],examples['text'])]}, batched=True)
dataset = dataset.map(lambda examples: {'preconcept_text': [x+' '+y for x,y in zip(examples['preconcept'],examples['text (#1)'])]}, batched=True)
dataset = dataset.map(encode, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

print(len(dataset['train']),'examples in training set and', len(dataset['validation']), 'examples in dev set')

training_args = TrainingArguments(
    output_dir='./pre_learn/cross_domain/geometry/results_with_concepts_paragraph_bert_small_data_5ep',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./pre_learn/cross_domain/geometry/results_with_concepts_paragraph_bert_small_data_5ep/logs/',
    save_steps=200,
    logging_steps=200,
    eval_steps=200
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
model.save_pretrained('./pre_learn/cross_domain/geometry/results_with_concepts_paragraph_bert_small_data_5ep/')
