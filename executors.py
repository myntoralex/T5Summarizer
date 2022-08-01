from jina import Executor, requests
from docarray import DocumentArray, Document
from typing import Dict
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class T5Summarizer(Executor):
  """T5 Transformer executor class for summarization"""

  def __init__(
        self,
        pretrained_model_name_or_path: str = 't5-small',
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

  @requests
  def encode(self, docs: 'DocumentArray', **kwargs):
    print(docs["@cc"].texts)
    prepared_text = "summarize: " + " ".join(docs["@cc"].texts)
    batch = self.tokenizer(prepared_text, truncation=True, padding='longest', return_tensors="pt")

    translated = self.model.generate(**batch, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=200, early_stopping=True)
    tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(tgt_text[0])
    return DocumentArray(Document(text=tgt_text[0]))

