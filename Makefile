init:
	pip install -r requirements.txt

test:
	python -m pytest nn_functor ${TEST_OPT}

watch_test:
	watchmedo shell-command -p '*.py;' -R -W -D -c "make test"
