from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
from tqdm import tqdm
import os
from transformers.pipelines import pipeline


class AttributionModel:

    def __init__(self, model_path: str = os.path.join(os.path.dirname(__file__),
                                                      'saved_model/results_b48/checkpoint-370')):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.nlp = pipeline(task='text-classification', model=self.model, tokenizer=self.tokenizer, device=0)

    def process_dataset(self, in_file_path: str, out_file_path: str, out_file_path_combined: str, batch_size: int = 32):
        data = pd.read_json(in_file_path, lines=True)
        data = data.to_dict(orient='records')
        inputs = [f"{row['entity']} [SEP] {row['context']}" for row in data]
        res = []

        for out in tqdm(self.nlp(inputs, batch_size=batch_size), total=len(data)):
            res.append(out['label'])

        with open(out_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join([json.dumps({**row, 'attribution': pred}) for row, pred in zip(data, res)]))

        texts_with_entities = self.combine_results(data, res)

        with open(out_file_path_combined, 'w', encoding='utf-8') as f:
            f.write("\n".join([json.dumps({article: list(entities)})
                               for article, entities in texts_with_entities.items()]))

    def combine_results(self, data, predictions):
        texts = defaultdict(lambda: set())

        for entity, pred in zip(data, predictions):
            if pred == 'LABEL_1':
                texts[entity['article_id']].add(entity['entity'])

        return texts


if __name__ == '__main__':
    model = AttributionModel()
    model.process_dataset(r"C:\Users\macie\Desktop\sem3\sta\bias-detection\data\ner_results_1_5k.jsonl",
                          r"C:\Users\macie\Desktop\sem3\sta\bias-detection\data\ner_results_1_5k_labelled_attribution.jsonl",
                          r"C:\Users\macie\Desktop\sem3\sta\bias-detection\data\ner_results_1_5k_labelled_attribution_texts.jsonl")


