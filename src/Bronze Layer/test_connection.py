from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

ACCOUNT_NAME = "tumorimages60104758"
CONTAINER_NAME = "raw"

credential = DefaultAzureCredential()
blob_service = BlobServiceClient(account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net", credential=credential)
container_client = blob_service.get_container_client(CONTAINER_NAME)

# List blobs
for blob in container_client.list_blobs():
    print(blob.name)
