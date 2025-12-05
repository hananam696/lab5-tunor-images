# lab5-tunor-images

All layer codes for this project are organised inside the src/ folder
**Phase 1: Bronze Layer ->  Data Ingestion**

Created containers and respective class folders in Azure, then programmatically ingested the image data into the correct folders using a Python script. Azure was accessed via CLI through the VS Code terminal to upload the images. The dataset was then registered in the Azure ML Workspace as a data asset.

**Phase 2: Silver Layer -> Data saved as Parquet**

Registered the environment and component using the .yml files through the VS Code terminal connected to Azure. The extract MRI images component was created inside the components folder.
Then, in the Azure ML notebook, the feature extraction code was run to generate the parquet file. This parquet file contains all extracted features and labels, preparing the data for the next modelling steps.

**Phase 3: Gold Layer -> Data ready for modelling**

The parquet files were then splited and saved into data splititng feature etrival and selection ws done then did training model 
