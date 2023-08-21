SHELL=/bin/bash
LINT_PATHS=multiarmedbandits/ tests/ 

pytest:
	$(SHELL) ./scripts/run_tests.sh

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint


# Test PyPi package release
# adapt to poetry
#TODO: add release
test-release:
	python -m build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: lint format check-codestyle commit-checks
