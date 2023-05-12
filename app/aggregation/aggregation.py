from typing import List, Dict, Tuple
from app.lemma.stanza_lemma import StanzaLemma
from collections import defaultdict


class DataAggregation:

    def __init__(self):
        self.lemma = StanzaLemma()

    def aggregate_results(self, data: List[Dict]):
        entities_info = {}

        for text in data:
            entities = [(entity['entity_value'], entity['ner_label']) for entity in text['entities']]
            lemmas_to_occurrences = self._disambiguate_entities(entities)

            for lemma, mentions in lemmas_to_occurrences.items():
                if lemma not in entities_info:
                    entities_info[lemma] = {'count': 1, 'mentions': defaultdict(lambda: 0)}
                else:
                    entities_info[lemma]['count'] += 1

                for mention in mentions:
                    entities_info[lemma]['mentions'][mention] += 1

        self._get_top_mention_form(entities_info)
        return entities_info

    def _disambiguate_entities(self, entities: List[Tuple[str, str]]):
        lemma_to_occurrence = defaultdict(lambda: [])

        for entity, ner_label in entities:
            lemma = self.lemma.lemma_into_text(entity, lower=False)['lemma_text'].upper()

            lemma_split = lemma.split()
            if ner_label == 'org' and len(lemma_split) >= 3:
                lemma = ''.join(word[0] for word in lemma)
            elif ner_label == 'per':
                lemma = lemma_split[-1]

            lemma_to_occurrence[lemma].append(entity.upper())

        return lemma_to_occurrence

    def _get_top_mention_form(self, entities_info):
        for entity_info in entities_info.values():
            top_mention = max((mention for mention in entity_info['mentions'].items()), key=lambda x: x[1])[0]
            entity_info['top_mention'] = top_mention

