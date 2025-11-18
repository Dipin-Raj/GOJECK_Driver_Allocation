.PHONY: setup_env remove_env data features train predict run clean test
PROJECT_NAME=work-at-gojek

ifeq ($(OS),Windows_NT)
    HAS_PYENV=False
    CONDA_ROOT=$(shell conda info --root)
    BINARIES = $(CONDA_ROOT)\\envs\\$(PROJECT_NAME)
    PYTHON_EXEC = $(BINARIES)\\python.exe
    PYTEST_EXEC = $(BINARIES)\\Scripts\\pytest.exe
    ACTIVATE_ENV = call $(CONDA_ROOT)\\Scripts\\activate $(PROJECT_NAME)
else
    ifeq (,$(shell which pyenv))
        HAS_PYENV=False
        CONDA_ROOT=$(shell conda info --root)
        BINARIES = ${CONDA_ROOT}/envs/${PROJECT_NAME}/bin
    else
        HAS_PYENV=True
        CONDA_VERSION=$(shell echo $(shell pyenv version | awk '{print $$1;}') | awk -F "/" '{print $$1}')
        BINARIES = $(HOME)/.pyenv/versions/${CONDA_VERSION}/envs/${PROJECT_NAME}/bin
    endif
    PYTHON_EXEC = $(BINARIES)/python
    PYTEST_EXEC = $(BINARIES)/pytest
    ACTIVATE_ENV = source $${CONDA_ROOT}/bin/activate $(PROJECT_NAME)
endif

setup_env:
ifeq (True,$(HAS_PYENV))
	@echo ">>> Detected pyenv, setting pyenv version to ${CONDA_VERSION}"
	pyenv local ${CONDA_VERSION}
	conda env create --name $(PROJECT_NAME) -f environment.yaml
	pyenv local ${CONDA_VERSION}/envs/${PROJECT_NAME}
else
	@echo ">>> Creating conda environment."
	conda env create --name $(PROJECT_NAME) -f environment.yaml
	@echo ">>> Activating new conda environment"
	$(ACTIVATE_ENV)
endif

#remove_env:
#ifeq (True,$(HAS_PYENV))
#	@echo ">>> Detected pyenv, removing pyenv version."
#	pyenv local ${CONDA_VERSION} && rm -rf ~/.pyenv/versions/${CONDA_VERSION}/envs/$(PROJECT_NAME)
#else
#	@echo ">>> Removing conda environemnt"
#	conda remove -n $(PROJECT_NAME) --all
#endif

data:
	@echo "Creating dataset from booking_log and participant_log.."
	$(PYTHON_EXEC) -m src.data.make_dataset

features:
	@echo "Running feature engineering on dataset.."
	$(PYTHON_EXEC) -m src.features.build_features

train:
	@echo "Training classification model for allocation task.."
	$(PYTHON_EXEC) -m src.models.train_model

predict:
	@echo "Performing model inference to identify best drivers.."
	$(PYTHON_EXEC) -m src.models.predict_model

test:
	@echo "Running all unit tests.."
	$(PYTEST_EXEC)

run: clean data features train predict test

clean:
	@find . -name "*.pyc" -exec rm {} \;
	@rm -f data/processed/* models/* submission/*;
