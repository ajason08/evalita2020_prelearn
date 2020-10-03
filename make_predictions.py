
#use argparse to get path of dataset
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='in_domain or cross_domain')
parser.add_argument('test_domain', help='The target domain')


#load model tokenizer

#load data

#fn to predict on data and return dict
#{'pred':pred}


#map with datasets lib


#write predictions to file use set_format

if __name__ == '__main__':

    args = parser.parse_args()
    if args.mode == 'in_domain':
        from config_indomain import *
    elif args.mode == 'cross_domain':
        from config import *
    else:
        print('No valid config exists for',args.test_domain)
        sys.exit()
        
    if args.test_domain == 'datamining':
        config = datamining_config
    elif args.test_domain == 'geometry':
        config = geometry_config
    elif args.test_domain == 'physics':
        config = physics_config
    elif args.test_domain == 'precalculus':
        config = precalculus_config
    else:
        print('No valid config exists for',args.test_domain)
        sys.exit()

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(config.best_model)# AutoModel
    model.eval()
    
    #for param in model.base_model.parameters():
    #    param.requires_grad = False
#    def predict(examples):
#        set['train']['concept_text'][8],dataset['train']['preconcept_text'][8],truncation=True, padding='max_leng
#            ...: th',return_tensors='pt'))[0].argmax()
#        pred = model(examples[])
#
#        return

    def encode(examples):
        #return tokenizer(examples['concept'], examples['preconcept'], truncation=True, padding='max_length')
        #return tokenizer(examples['text'], examples['text (#1)'], truncation=True, padding='max_length')
        return tokenizer(examples['concept_text'], examples['preconcept_text'], truncation=True, padding='max_length')


    dataset = load_dataset('csv', data_files={'test':config.test}, delimiter='\t')
#    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset = dataset.map(lambda examples: {'concept_text': [x+' '+y for x,y in zip(examples['concept'],examples['text'])]}, 
        batched=True
    )
    dataset = dataset.map(lambda examples: {'preconcept_text': [x+' '+y for x,y in zip(examples['preconcept'],examples['text (#1)'])]},         batched=True
    )
    print(len(dataset['test']))
#    dataset = dataset.map(encode, batched=True)
    
#    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    print(dataset['test'].features)
    preds = []
    for i in range(len(dataset['test'])):
        pred = model(**tokenizer(dataset['test']['concept_text'][i],dataset['test']['preconcept_text'][i],truncation=True, padding='max_length',return_tensors='pt'))[0].argmax().item()
        #pred =  model(example['input_ids'], example['attentio_mask'])[0].argmax().item()
        preds.append(pred)

    assert len(dataset['test']) == len(preds)
#    print(len(dataset))
 #   print(len(preds))
    import pandas as pd
    df = pd.DataFrame(preds)
    df.to_csv(config.submission_file,header=None, index=False)
    #dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
