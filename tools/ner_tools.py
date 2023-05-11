from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import xx_ent_wiki_sm


def find_entities_neighbourhood(
        keywords: list,
        text: str,
        input_character_window_size: int = 150,
        output_word_window_size: int = 10,
) -> list:
    results = []


    for word in keywords:
        if int(word['start']) < input_character_window_size:
            start = text[:int(word['start'])].split()[-output_word_window_size:]
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
        context = ' '.join(result)
        context = context.replace('  ', ' ')
        results.append((word['word'],word['entity_group'],context))
    return results


class NER:
    def __init__(self,
                 model_name_or_path="./models/sloner"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy='simple')
        self.nlp.tokenizer.model_max_length = 512
        self.nlp.tokenizer.truncation = True
        self.sentencizer = xx_ent_wiki_sm.load()
        self.sentencizer.add_pipe('sentencizer')

    def find_entities(self,
                      text: str,
                      ):
        """TO DO - add description"""
        # split text into sentences
        sentences = [str(sent) for sent in self.sentencizer(text).sents]
        # split sentences into packs of 4 sentences
        texts = [' '.join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]
        results = []
        for smaller_text in texts:
            smaller_entities = self.__find_entities_single_text__(smaller_text)

            results.append(smaller_entities)

        out = []
        for result, smaller_text in zip(results, texts):
            out.extend(find_entities_neighbourhood(keywords=result, text=smaller_text))

        return out

    def __find_entities_single_text__(self,
                                      text: str,
                                      ):
        ner_results = self.nlp(text)
        lst = []
        for i in range(len(ner_results)):
            if ner_results[i]['entity_group'] == 'per' or ner_results[i]['entity_group'] == 'org':
                if i == 0:
                    lst.append(
                        {'word': text[ner_results[i]['start']:ner_results[i]['end']],
                         'start': ner_results[i]['start'], 'end': ner_results[i]['end'],
                         'entity_group': ner_results[i]['entity_group']})
                else:
                    if ner_results[i]['start'] == ner_results[i - 1]['end'] and ner_results[i]['entity_group'] == \
                            ner_results[i - 1]['entity_group']:
                        lst[-1] = {'word': lst[-1]['word'] + text[ner_results[i]['start']:ner_results[i]['end']],
                                   'start': lst[-1]['start'],
                                   'end': ner_results[i]['end'], 'entity_group': lst[-1]['entity_group']}
                    else:
                        lst.append({'word': text[ner_results[i]['start']:ner_results[i]['end']],
                                    'start': ner_results[i]['start'],
                                    'end': ner_results[i]['end'], 'entity_group': ner_results[i]['entity_group']})
        return lst