MANAGER = poetry run
DEVICE = 'cuda:0'
MLFLOW_PORT = 8080
MLFLOW_HOST = 127.0.0.1
install:
	${MANAGER} poetry install --with dev,gpu,conversion,lint,test
download-dataset:
	${MANAGER} python butterfly_classification/download_dataset.py
format:
	${MANAGER} isort butterfly_classification
	${MANAGER} black butterfly_classification
run-mlflow-server:
	${MANAGER} mlflow server --default-artifact-root ./mlruns --host ${MLFLOW_HOST} --port ${MLFLOW_PORT}
run-train:
	${MANAGER} python butterfly_classification/train_pipeline/train.py
run-inference-pipeline:
	${MANAGER} python butterfly_classification/inference/predict.py
pre-commit-install:
	${MANAGER} pre-commit install
run-conversion-pipeline:
	${MANAGER} python butterfly_classification/model_conversion_pipeline/pipeline.py
