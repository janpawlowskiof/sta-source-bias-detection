from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def find_entities_neighbourhood(
        keywords: list,
        text: str,
        input_character_window_size: int = 150,
        output_word_window_size: int = 6,
) -> list:
    results = []
    for word in keywords:
        if int(word['start']) < input_character_window_size:
            start = text[:int(word['start'])].split()[output_word_window_size:]
        else:
            start = text[int(word['start']) - input_character_window_size:int(word['start'])] \
                        .split()[-output_word_window_size:]
        if int(word['end']) + output_word_window_size > len(text):
            end = text[int(word['end']):].split()[:output_word_window_size]
        else:
            end = text[int(word['end']):int(word['end']) + input_character_window_size].split()[
                  :output_word_window_size]
        result = start.copy()
        result.append(word['word'])
        result.extend(end)
        results.append(' '.join(result))
    return results


class NER:
    def __init__(self,
                 model_name_or_path="./models/sloner"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy='simple')

    def find_entities(self,
                      text: str,
                      ):
        # ADD Functionality for texts longer than 512 tokens
        ner_results = self.nlp(text)

        # connect all entities where start of the first one is the end of the previous one and they are the same type
        lst = []
        for i in range(len(ner_results)):
            if ner_results[i]['entity_group'] == 'per' or ner_results[i]['entity_group'] == 'org':
                if i == 0:
                    lst.append(
                        {'word': ner_results[i]['word'], 'start': ner_results[i]['start'], 'end': ner_results[i]['end'],
                         'entity_group': ner_results[i]['entity_group']})
                else:
                    if ner_results[i]['start'] == ner_results[i - 1]['end'] and ner_results[i]['entity_group'] == \
                            ner_results[i - 1]['entity_group']:
                        lst[-1] = {'word': lst[-1]['word'] + ner_results[i]['word'], 'start': lst[-1]['start'],
                                   'end': ner_results[i]['end'], 'entity_group': lst[-1]['entity_group']}
                    else:
                        lst.append({'word': ner_results[i]['word'], 'start': ner_results[i]['start'],
                                    'end': ner_results[i]['end'], 'entity_group': ner_results[i]['entity_group']})
        return lst
