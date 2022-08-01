from jina import Executor, DocumentArray, requests


class T5Summarizer(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
