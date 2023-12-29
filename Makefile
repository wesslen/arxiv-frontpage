install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-build.txt
	python -m pip install -e .
	python -m spacy download en_core_web_sm
	python -m pip install prodigy -f https://$PRODIGY_KEY@download.prodi.gy
