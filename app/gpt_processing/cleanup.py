import json5
from typing import Dict, List

from app.lemma.stanza_lemma import StanzaLemma


class GPT35EntitiesCleaner:
    def __init__(self) -> None:
        self.lemma = StanzaLemma()

    def cleanup_responses(self, rows: List[Dict]) -> List[Dict]:
        responses_with_metadata = [
            {"raw_response": row["response"], "model_input": row["model_input"]}
            for row in rows
        ]
        print("filtering parsable..")
        responses_with_metadata = filter(self._is_parsable_with_correct_keys, responses_with_metadata)
        entities_with_metadata = sum(map(self._extract_entities_with_metadata, responses_with_metadata), [])
        print("filtering exact context matching...")
        entities_with_metadata = filter(self._is_context_an_exact_match, entities_with_metadata)
        print("filtering entity in lemma context...")
        entities_with_metadata = filter(self._is_entity_in_context, entities_with_metadata)
        print("finished filtering")
        return list(entities_with_metadata)

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
        
    def _extract_entities_with_metadata(self, response_with_metadata: Dict) -> List[Dict]:
        response = response_with_metadata["response"]
        model_input = response_with_metadata["model_input"]

        for entity in response["media outlets"]:
            entity["type"] = "media"
            entity["model_input"] = model_input
        for entity in response["institutions"]:
            entity["type"] = "institution"
            entity["model_input"] = model_input

        return response["media outlets"] + response["institutions"]

    def _is_context_an_exact_match(self, entity_with_metadata: Dict) -> bool:
        return entity_with_metadata["context"] in entity_with_metadata["model_input"]

    def _is_entity_in_context(self, entity_with_metadata: Dict) -> bool:
        entity_with_metadata["entity_lemma"] = self.lemma.lemma_into_text(entity_with_metadata["text"])
        entity_with_metadata["context_lemma"] = self.lemma.lemma_into_text(entity_with_metadata["context"])
        # This is to exclude cases, where lemma of a word is a part of a longer word
        return self._surround_with_spaces(entity_with_metadata["entity_lemma"]) in self._surround_with_spaces(entity_with_metadata["context_lemma"])

    def _surround_with_spaces(self, text: str) -> str:
        return f" {text} "