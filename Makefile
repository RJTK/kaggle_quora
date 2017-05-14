.PHONY: clean data lint sync_data_to_s3 sync_data_from_s3 #requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = rjtk/quora_question_pairs
PROJECT_NAME = quora_question_pairs
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
#requirements: test_environment
#	pip install -r requirements.txt

.PHONY: show-help get_raw_data tokenize_data data
TEST?=False
I_MAX?=0

## Get the raw data
get_raw_data:
	$(PYTHON_INTERPRETER) src/data/download_raw_data.py $(TEST)

## Clean the data
clean_data:
	$(PYTHON_INTERPRETER) src/data/clean_data.py $(TEST) $(I_MAX)

## Tokenize everything
tokenize_data:
	$(PYTHON_INTERPRETER) src/data/tokenize_data.py $(TEST) $(I_MAX)

## Take the position of words
pos_tag:
	$(PYTHON_INTERPRETER) src/data/pos_tag.py $(TEST) $(I_MAX)

## Lemmatizes the data
lemmatize:
	$(PYTHON_INTERPRETER) src/data/lemmatize_data.py $(TEST) $(I_MAX)

## Calculate distances
distances:
	$(PYTHON_INTERPRETER) src/data/distance_calculations.py $(TEST) $(I_MAX)

## Extract the type of question
extract_question:
	$(PYTHON_INTERPRETER) src/data/extract_question.py $(TEST) $(I_MAX)

## ALL DATA
all_data: clean_data tokenize_data pos_tag lemmatize distances extract_question
#all_data: clean_data tokenize_data distances extract_question
#all_data: pos_tag lemmatize distances

## PRODUCTION PIPELINE
production:
	cp data/interim/test.csv data/processed/test.csv
	cp data/interim/train.csv data/processed/train.csv
	aws s3 sync data/processed/ s3://$(BUCKET)/data/processed/

## Basic Model
basic_model:
	python src/models/train_model.py
	python src/models/predict_model.py
	aws s3 sync reports/submissions/ s3://$(BUCKET)/reports/submissions/
	sudo shutdown now

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Upload Data to S3
sync_data_to_s3:
	aws s3 sync data/processed/ s3://$(BUCKET)/data/processed/

## Download Data from S3
sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.5
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
