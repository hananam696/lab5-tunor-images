from azure.storage.filedatalake import DataLakeServiceClient
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

account_name = "tumorimages60104758"
account_key = "<account key replaced for publishing>"
service_client = DataLakeServiceClient(
    account_url=f"https://{account_name}.dfs.core.windows.net",
    credential=account_key
)
raw_container = "raw"
raw_folder = "tumor_images/"

file_system_client = service_client.get_file_system_client(raw_container)
paths = file_system_client.get_paths(path=raw_folder)

data_list = []
for path in paths:
    if path.name.endswith((".jpeg", ".jpg")):

        file_client = file_system_client.get_file_client(path.name)
        download = file_client.download_file()
        file_bytes = download.readall()

        data_list.append({"filename": path.name, "feature1": len(file_bytes)})


df = pd.DataFrame(data_list)

silver_container = "silver"
output_path = "features_v1/features.parquet"

silver_client = service_client.get_file_system_client(silver_container)
file_client = silver_client.get_file_client(output_path)

table = pa.Table.from_pandas(df)
pq_bytes = BytesIO()
pq.write_table(table, pq_bytes)
pq_bytes.seek(0)

file_client.upload_data(pq_bytes.read(), overwrite=True)

print("Parquet successfully created in silver/features_v1!")
