from azure.ai.ml import MLClient, dsl
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import CommandComponent

# -------------------------------
# Azure ML workspace details
# -------------------------------
subscription_id = "a00dcbea-fd05-4973-82dc-120208b60116"
resource_group = "rg-60104758"
workspace_name = "lab5-60104758"

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)


bronze_component = CommandComponent(
    name="bronze_ingest_mri_temp",
    command="python create_parquet.py --output ${{outputs.raw_data}}",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38:1",
    inputs={},
    outputs={"raw_data": {"type": "uri_folder"}}
)

silver_component = CommandComponent(
    name="silver_feature_extraction_temp",
    command="python extract_features_component/extract_features.py --input ${{inputs.raw_data}} --output ${{outputs.features}}",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38:1",
    inputs={"raw_data": {"type": "uri_folder"}},
    outputs={"features": {"type": "uri_folder"}}
)

gold_fs_component = CommandComponent(
    name="gold_feature_selection_temp",
    command="python 'Gold Layer/feature_selection.py' --input ${{inputs.features}} --output ${{outputs.selected_features}}",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38:1",
    inputs={"features": {"type": "uri_folder"}},
    outputs={"selected_features": {"type": "uri_folder"}}
)

train_model_component = CommandComponent(
    name="train_model_temp",
    command="python 'Gold Layer/train_model.py' --input ${{inputs.selected_features}} --output ${{outputs.model}}",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38:1",
    inputs={"selected_features": {"type": "uri_folder"}},
    outputs={"model": {"type": "mlflow_model"}}
)

# -------------------------------
# Build the pipeline
# -------------------------------
@dsl.pipeline(description="Tumor image ML pipeline")
def tumor_pipeline():
    bronze_step = bronze_component()
    silver_step = silver_component(raw_data=bronze_step.outputs.raw_data)
    gold_fs_step = gold_fs_component(features=silver_step.outputs.features)
    train_step = train_model_component(selected_features=gold_fs_step.outputs.selected_features)

    return {"model": train_step.outputs.model}

pipeline_job = tumor_pipeline()
ml_client.jobs.create_or_update(pipeline_job)
print("Pipeline submitted successfully!")
