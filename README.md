# lab5-tumor-images
**Half of the lab was done using CLI and other half was done in Azure ML**
**Storage name: tumorimages6010475, Azure ML workspace name: lab5-60104758**
All layer codes for this project are organised inside the src/ folder
pipeline code in file `pipeline_job.py`

**Phase 1: Bronze Layer ->  Data Ingestion**

Created containers and respective class folders in Azure, then programmatically ingested the image data into the correct folders using a Python script. Azure was accessed via CLI through the VS Code terminal to upload the images. The dataset was then registered in the Azure ML Workspace as a data asset called `tumor_images_raw`.

**Phase 2: Silver Layer -> Data saved as Parquet**

Registered the environment and component using the .yml files through the VS Code terminal connected to Azure. The extract MRI images component was created inside the components folder.
Then, in the Azure ML notebook, the feature extraction code was run to generate the parquet file. This parquet file contains all extracted features and labels, preparing the data for the next modelling steps.

**Phase 3: Gold Layer -> Data ready for modelling**

The Silver parquet files were first split into train and test sets using the feature retrieval code. The resulting files, train.parquet and test.parquet, were saved in the gold/data_splitting folder. Next, the feature selection code was run, producing and saving ga_metrics.json, baseline_metrics.json, and selected_features.json in the gold/feature_selection folder. Finally, the Random Forest model was trained using the selected features, metrics were saved, and the trained model was registered in the Azure ML Workspace named `tumor_rf_model`.

## **How to Run:**
All layer codes for this project are organised inside the src/ folder, and the pipeline is in pipeline_job.py. To run the project, first ensure your Azure ML workspace is set up and your dataset is uploaded to ADLS Gen2. Start by running the Bronze-layer ingestion script to load the MRI images and register the dataset. Next, run the Silver-layer feature extraction code to generate Parquet files containing all image features. Then, execute the Gold-layer scripts: split the data using feature retrieval, perform feature selection (GA approach), and train the Random Forest model. The resulting metrics and model are saved in the Gold layer, and the trained model can be registered in Azure ML Workspace as tumor_rf_model. The scripts can be run sequentially in VS Code terminal or an Azure ML notebook, following the order: bronze_ingest.py → extract_features.py → feature_retrieval.py → feature_selection.py → train_model.py.

## **Pipeline Overview:**
The Azure ML pipeline (pipeline_job.py) automates the workflow from Bronze to Gold layers. It links the ingestion, feature extraction, feature selection, and model training steps, passing outputs between components so the entire process runs end-to-end without manual intervention. While the feature store integration was skipped, the pipeline captures GA metrics versus baseline metrics and saves train/test Parquet files, selected features, and the trained model. This pipeline enables reproducible experimentation, reduces manual errors, and can be extended to deploy an online endpoint for single-image tumor predictions, demonstrating a full MLOps workflow for MRI tumor detection.
