.PHONY: test test-verbose test-dress test-diagnosis

test:
	pytest tests/test_all.py

test-verbose:
	pytest -v -s tests/test_all.py

test-dress:
	pytest -v -s tests/test_all.py::test_dress_diagnosis

test-diagnosis:
	pytest -v -s tests/test_all.py::test_dress_diagnosis tests/test_all.py::test_embeddings_diagnosis
