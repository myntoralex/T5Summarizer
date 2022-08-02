from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from jina import Document, DocumentArray, Executor
from executor import T5Summarizer


@pytest.fixture(scope='session')
def basic_summarizer() -> T5Summarizer:
    return T5Summarizer()

def test_summarize(basic_summarizer: T5Summarizer):
    sentences = DocumentArray([Document(text="The term rabbit is typically used for all Leporidae species excluding the genus Lepus. Members of that genus are instead known as hares or jackrabbits.")])
    basic_summarizer.encode(sentences)

