from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import xx_ent_wiki_sm
import pandas as pd


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


def reformat_df(
        base: pd.DataFrame,
):
    dfs = []
    for i in range(len(base)):
        for j in range(len(base['entities'].iloc[i])):
            dfs.append(pd.DataFrame({'entity': base['entities'].iloc[i][j][0],
                                     'context': base['entities'].iloc[i][j][2],
                                     'label': base['entities'].iloc[i][j][1],
                                     'article_id': base['id'].iloc[i],
                                     'text': base['text']}, index=[0]))
    return pd.concat(dfs, ignore_index=True)


def __find_context_in_text__(context, text):
    start = text.find(context)
    end = start + len(context)
    return start, end


def postprocess_df(
        df_to_process: pd.DataFrame,
):
    df_to_process['context'] = df_to_process.apply(lambda x: x['context'].replace(' .', '.').replace(' ,', ','), axis=1)
    df_to_process['text'] = df_to_process.apply(lambda x: x['text'].replace('\n\n', ' '), axis=1)
    df_to_process['context_start'] = df_to_process.apply(lambda x: __find_context_in_text__(x['context'], x['text'])[0], axis=1)
    df_to_process['context_end'] = df_to_process.apply(lambda x: __find_context_in_text__(x['context'], x['text'])[1], axis=1)

    df_to_process['entity_start'] = df_to_process.apply(lambda x: __find_context_in_text__(x['entity'], x['context'])[0] + x['context_start'], axis=1)

    df_to_process['entity_end'] = df_to_process.apply(lambda x: __find_context_in_text__(x['entity'], x['context'])[1] + x['context_start'], axis=1)
    df_to_process['found_entity'] = df_to_process.apply(lambda x: x['text'][x['entity_start']:x['entity_end']], axis=1)

    df_to_process = df_to_process.drop_duplicates(subset=['article_id', 'entity_end'], keep='first')
    df_to_process = df_to_process.drop_duplicates(subset=['article_id', 'entity_start'], keep='first')

    res = df_to_process[df_to_process['entity'] == df_to_process['found_entity']]
    res = res[res['entity'].apply(lambda x: len(x) > 1)]

    return res

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