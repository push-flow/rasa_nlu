from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import range
import io
import os
import warnings
import six

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class EntitySynonymMapper(EntityExtractor):
    name = "ner_synonyms"

    provides = ["entities"]

    def __init__(self, synonyms=None):
        # type: (Optional[Dict[Text, Text]]) -> None
        self.synonyms = synonyms if synonyms else {}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get("entities", []):
                entity_val = example.text[entity["start"]:entity["end"]]
                self.add_entities_if_synonyms(entity_val, entity.get("value"))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated_entities = message.get("entities", [])[:]
        self.replace_synonyms(updated_entities)
        message.set("entities", updated_entities, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        if self.synonyms:
            return {"entity_synonyms": self.synonyms}

    @classmethod
    def load(cls, model_dir, model_metadata, cached_component, **kwargs):
        # type: (Text, Metadata, Optional[EntitySynonymMapper], **Any) -> EntitySynonymMapper

        if model_dir and model_metadata.get("entity_synonyms"):
            synonyms = model_metadata.get("entity_synonyms")
            return EntitySynonymMapper(synonyms)

        return EntitySynonymMapper()

    def replace_synonyms(self, entities):
        for entity in entities:
            entity_value = entity["value"]
            if entity_value.lower() in self.synonyms:
                entity["value"] = self.synonyms[entity_value.lower()]
                self.add_processor_name(entity)

    def add_entities_if_synonyms(self, entity_a, entity_b):
        if entity_b is not None:
            original = entity_a if isinstance(entity_a, six.text_type) else six.text_type(entity_a)
            original = original.lower()
            replacement = entity_b if isinstance(entity_b, six.text_type) else six.text_type(entity_b)

            if original != replacement:
                self.synonyms[original] = replacement
