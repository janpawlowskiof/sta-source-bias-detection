import json5
from typing import Dict, List

from tqdm import tqdm

from app.lemma.stanza_lemma import StanzaLemma
from app.utils import remove_multiple_spaces


class GPT35EntitiesCleaner:
    def __init__(self) -> None:
        self.lemma = StanzaLemma()

    def cleanup_responses(self, rows: List[Dict]) -> List[Dict]:
        responses_with_metadata = [
            {"raw_response": row["response"], "model_input": row["model_input"]}
            for row in rows
        ]
        responses_with_metadata = filter(self._is_parsable_with_correct_keys, tqdm(responses_with_metadata, "filtering parsable responses"))
        responses_with_metadata = list(responses_with_metadata)
        entities_with_metadata = sum(map(self._extract_entities_with_metadata, responses_with_metadata), [])
        entities_with_metadata = filter(self._is_context_an_exact_match, tqdm(entities_with_metadata, "filtering exact context matching"))
        entities_with_metadata = filter(self._is_entity_in_context, tqdm(entities_with_metadata, "filtering entity in lemma context"))

        for entity in entities_with_metadata:
            entity["is_valid"] = True
            del entity["model_input"]

        for response_with_metadata in responses_with_metadata:
            response_with_metadata["all_entities"] = response_with_metadata["response"]["media outlets"] + response_with_metadata["response"]["institutions"]
            del response_with_metadata["raw_response"]
            del response_with_metadata["response"]

        responses_with_metadata = list(filter(self._has_only_valid_entites, tqdm(responses_with_metadata, "filtering responses with only valid entites")))
        responses_with_metadata = list(map(self._mark_entites_in_text, tqdm(responses_with_metadata, "marking entites in text")))
        responses_with_metadata = list(filter(bool, responses_with_metadata))
        responses_with_metadata = list(map(self._convert_to_final_format, tqdm(responses_with_metadata, "converting to final format")))

        print(f"finished with {len(responses_with_metadata)} articles")
        return list(responses_with_metadata)

    def _is_parsable_with_correct_keys(self, response_with_metadata: Dict[str, str]) -> bool:
        try:
            parsed_response = json5.loads(response_with_metadata["raw_response"])
            assert("media outlets" in parsed_response)
            assert("institutions" in parsed_response)

            assert(all("text" in x for x in parsed_response["media outlets"]))
            assert(all("context" in x for x in parsed_response["media outlets"]))
            assert(all("text" in x for x in parsed_response["institutions"]))
            assert(all("context" in x for x in parsed_response["institutions"]))

            assert(all(x["text"] for x in parsed_response["media outlets"]))
            assert(all(x["context"] for x in parsed_response["media outlets"]))
            assert(all(x["text"] for x in parsed_response["institutions"]))
            assert(all(x["context"] for x in parsed_response["institutions"]))
            response_with_metadata["response"] = parsed_response
            return True
        except AssertionError:
            return False
        except ValueError:
            return False
        
    def _has_only_valid_entites(self, response_with_metadata: Dict[str, str]) -> bool:
        if len(response_with_metadata["all_entities"]) <= 0:
            return False
        return all(entity["is_valid"] for entity in response_with_metadata["all_entities"])
        
    def _mark_entites_in_text(self, response_with_metadata: Dict[str, str]) -> Dict[str, str]:
        for entity in response_with_metadata["all_entities"]:
            context_words = entity["context_lemma"]["words"]
            for first_matching_context_word_index in range(len(context_words)):
                is_good_first_context_word = True
                for entity_word_index, entity_word in enumerate(entity["entity_lemma"]["words"]):
                    corresponding_context_word = entity["context_lemma"]["words"][first_matching_context_word_index + entity_word_index]
                    if entity_word["lemma"].lower() != corresponding_context_word["lemma"].lower():
                        is_good_first_context_word = False
                        break
                if is_good_first_context_word:
                    break
            else:
                print(f"Did not find matching words in context for entity {entity['text']}")
                return None

            num_entity_words = len(entity["entity_lemma"]["words"])
            entity_words_in_context = entity["context_lemma"]["words"][first_matching_context_word_index: first_matching_context_word_index + num_entity_words]
            context_start_index = response_with_metadata["model_input"].index(entity["context"])
            local_start_index = entity_words_in_context[0]["start_char"]
            local_end_index = entity_words_in_context[-1]["end_char"]

            entity["start_index"] = context_start_index + local_start_index
            entity["end_index"] = context_start_index + local_end_index
            entity["text"] = entity["context"][local_start_index:local_end_index]
            assert entity["text"] == response_with_metadata["model_input"][entity["start_index"]:entity["end_index"]]
        return response_with_metadata

    def _convert_to_final_format(self, response_with_metadata: Dict[str, str]) -> Dict[str, str]:
        return {
            "text": response_with_metadata["model_input"],
            "entities": [
                {
                    "entity_value": entity["text"],
                    "start": entity["start_index"],
                    "end": entity["end_index"],
                    "label": entity["type"]
                }
                for entity in response_with_metadata["all_entities"]
            ]
        }

    def _extract_entities_with_metadata(self, response_with_metadata: Dict) -> List[Dict]:
        response = response_with_metadata["response"]
        model_input = response_with_metadata["model_input"]

        for entity in response["media outlets"]:
            entity["type"] = "media"
        for entity in response["institutions"]:
            entity["type"] = "institution"

        all_entities = response["media outlets"] + response["institutions"]
        for entity in all_entities:
            entity["model_input"] = model_input
            # flag to be set to true is entity passes all checks
            entity["is_valid"] = False
        # ensure no duplicates
        entities_texts = {}
        for entity in all_entities:
            entities_texts[entity["text"]] = entity
        all_entities = list(entities_texts.values())
        return all_entities

    def _is_context_an_exact_match(self, entity_with_metadata: Dict) -> bool:
        return entity_with_metadata["context"] in entity_with_metadata["model_input"]

    def _is_entity_in_context(self, entity_with_metadata: Dict) -> bool:
        entity_with_metadata["entity_lemma"] = self.lemma.lemma_into_text(entity_with_metadata["text"])
        entity_with_metadata["context_lemma"] = self.lemma.lemma_into_text(entity_with_metadata["context"])
        # This is to exclude cases, where lemma of a word is a part of a longer word
        return self._surround_with_spaces(entity_with_metadata["entity_lemma"]["lemma_text"]) in self._surround_with_spaces(entity_with_metadata["context_lemma"]["lemma_text"])

    def _surround_with_spaces(self, text: str) -> str:
        return f" {text} "