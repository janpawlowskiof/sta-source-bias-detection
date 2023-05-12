from collections import defaultdict
import pandas as pd
from typing import Dict, List, Optional


def generate_summary_for_agency(predictions: List[List[Dict[str, str]]], return_top_k: Optional[int] = 20) -> pd.DataFrame:
    counter = defaultdict(int)
    for prediction in predictions:
        # only one citation of entity per article
        article_entity_names = set()
        for entity in prediction["entities"]:
            if entity["label"] == "SourcePer":
                entity_name = "PublicPerson"
            else:
                entity_name = entity["entity_value"]
            article_entity_names.add(entity_name)
        for entity_name in article_entity_names:
            counter[entity_name] += 1
    counter = dict(counter)
    rows = [
        {"entity_name": entity_name, "number_of_occurences": number_of_occurences}
        for entity_name, number_of_occurences
        in counter.items()
    ]

    df = pd.DataFrame.from_records(rows).sort_values(by="number_of_occurences", ascending=False)
    if not return_top_k:
        return df
    top_df = df.iloc[:return_top_k]
    bottom_df = df.iloc[return_top_k:]
    top_df = top_df.append({"entity_name": "OtherEntities", "number_of_occurences": bottom_df["number_of_occurences"].sum()}, ignore_index=True)
    return top_df
