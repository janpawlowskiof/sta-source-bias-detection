from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
from transformers.pipelines import pipeline
from typing import List, Dict


class AttributionModel:

    def __init__(self, model_path: str = os.path.join('../../models/attribution_clf')):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.nlp = pipeline(task='text-classification', model=self.model, tokenizer=self.tokenizer)

    def process_dataset(self, data: List[Dict]) -> List[Dict]:
        inputs = [f"{row['entity']} [SEP] {row['context']}" for row in data]
        res = [out['label'] for out in tqdm(self.nlp(inputs), total=len(data))]
        data = [{**row, 'attribution_label': pred} for row, pred in zip(data, res)]
        return self.process_data_to_text_entity_format(data)

    def process_data_to_text_entity_format(self, data: List[Dict]) -> List[Dict]:
        texts = defaultdict(lambda: {"entities": []})

        for row in data:
            if row['attribution'] == 'LABEL_1':
                texts[row['article_id']]['text'] = row['text']
                texts[row['article_id']]['entities'].append(
                    {
                        "entity_value": row['entity'],
                        "start": row['entity_start'],
                        "end": row['entity_end'],
                        "ner_label": row['label'],
                    }
                )

        return [{**article_data, 'article_id': article_id} for article_id, article_data in texts.items()]
