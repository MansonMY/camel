#%%
from typing import Any, List, Optional, Tuple
from uuid import uuid4
from camel.storages.vectordb_storages import (
    BaseVectorStorage,
    VectorDBQuery,
    VectorDBQueryResult,
    VectorDBStatus,
    VectorRecord,
)
### Create a MilvusStorage module with function of add, quary, search, delet and clear 

class MilvusStorage(BaseVectorStorage):
    """
    This module is to integration of Milvus, a vector database, into Camel agent.
    The information of Milvus can be accessed to Milvus websites at: https://milvus.io/docs

    Args:
        vector_dim (int): The dimenstion of storing vectors.
        collection_name (Optional[str], optional): Name for the collection in
            the Milvus. If not provided, set it to the current time with iso
            format. (default: :obj:`None`)
        url_and_api (Optional[Tuple[str, str]], optional): Tuple containing
            the URL and API key for connecting to a remote Milvus instance.
            (default: :obj:`None`)
        path (Optional[str], optional): Path to a directory for initializing a
            local Milvus client. (default: :obj:`None`)
        distance (VectorDistance, optional): The distance metric for vector
            comparison (default: :obj:`VectorDistance.COSINE`)
        **kwargs (Any): Additional keyword arguments for initializing
            `MilvusClient`.
    """

    # Initialize the MilvusStorage class with required parameters
    def __init__(self, vector_dim: int, url_and_api: Tuple[str, str], collection_name:Optional[str]=None, **kwargs: Any) -> None:
        from pymilvus import MilvusClient
        self._client: MilvusClient
        self.vector_dim = vector_dim
        self._create_client(url_and_api, **kwargs)   
        self.collection_name = collection_name or self._generate_collection_name()
        if not self._check_collection_exists(self.collection_name):
            self._create_collection(self.collection_name)

    # Check and if the collection name has been create from Milvus database
    def _check_collection_exists(self, collection_name: str) -> bool:
        for c_n in self._client.list_collections():
            if collection_name == c_n:
                return True
        return False    

    # Create a Milvus client using the provided URL and API key
    def _create_client(self, url_and_api_key: Tuple[str, str], **kwargs: Any) -> None:
        from pymilvus import MilvusClient
        self._client = MilvusClient(uri=url_and_api_key[0], token=url_and_api_key[1], **kwargs)

    # Create a new collection in Milvus with a unique name and vector dimensions
    def _create_collection(self,collection_name, **kwargs: Any) -> None:    
        from pymilvus import CollectionSchema, FieldSchema, DataType
        from pymilvus.milvus_client.index import IndexParams
        fields = [
            FieldSchema(name = "id", dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema(name = "vector", dtype=DataType.FLOAT_VECTOR, is_primary=False, dim=self.vector_dim),
            FieldSchema(name = "payload", dtype=DataType.VARCHAR, is_primary=False, max_length=65535)
            ] #the params of dtype, is_primary, and max_length can check from https://milvus.io/docs/create_collection.md

        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)  

        index_params = IndexParams()
        index_params.add_index("vector", params={"metric_type": "COSINE", "params": {}})
        self._client._create_collection_with_schema(collection_name=collection_name, schema=schema, index_params=index_params, **kwargs)    
    
    # Generate a unique collection name using UUID
    def _generate_collection_name(self) -> str:
        uuid_id = str(uuid4())
        valid_name = "test_" + uuid_id.replace("-", "_")
        return valid_name 
    
    # Delete a specified collection from Milvus 
    def _delete_collection(self, collection_name: str) -> None:
        self._client.drop_collection(collection_name=collection_name)

    # Add vector records to the collection    
    def add(self, records: List[VectorRecord], **kwargs) -> None:
        add_data = []
        for record in records:
            payload_str = str(record.payload) if record.payload is not None else ""
            milvus_record = {
                "id": record.id,
                "vector": record.vector,
                "payload": payload_str  
            }
            add_data.append(milvus_record)
        
        self._client.insert(collection_name=self.collection_name, data=add_data, **kwargs)
        self._client.load_collection(collection_name=self.collection_name, data=add_data, **kwargs)

    # Delete specified vector records from the collection using their IDs
    def delete(self, ids: List[str], **kwargs: Any) -> None:
        self._client.delete(collection_name=self.collection_name, ids=ids, **kwargs)

    # Retrieve the current status of the collection, including vector dimensions and count
    def status(self) -> VectorDBStatus:
        collection_description = self._client.describe_collection(self.collection_name)
        collection_id = collection_description['collection_id']
        for field in collection_description['fields']:
            if field['name'] == 'vector':
                vector_dim = field['params']['dim']
                break 
        stats = self._client.get_collection_stats(collection_name=self.collection_name)
        total_vectors = stats.get("row_count", 0)
        return {
        "collection_id":collection_id,
        "vector_dim": vector_dim,
        "total_vectors": total_vectors
        }

    # Query the collection for similar vector records based on a given vector
    def query(self, query: VectorDBQuery, **kwargs: Any) -> List[VectorDBQueryResult]:
        search_results = self._client.search(collection_name=self.collection_name, data=[query.query_vector], limit=query.top_k,output_fields=['vector', 'payload'], **kwargs)
        query_results = [
            VectorDBQueryResult.construct(
                id=str(point['id']),
                similarity=(1 - point['distance']),
                payload={'message': point['entity'].get('payload')},
                vector=point['entity'].get('vector')
            ) for hits in search_results for point in hits]       
        query_results = sorted(query_results, key=lambda x: x.similarity, reverse=True)        
        return query_results

    # Clear the collection by deleting and recreating it
    def clear(self) -> None:
        self._delete_collection(self.collection_name)
        self.collection_name = self._create_collection(collection_name=self.collection_name)

    # Provide access to the private Milvus client instance
    @property
    def client(self) -> Any:
        return self._client

### Test the MilvusStorage module and its function of Add, Quary, Search, Deletion and Clear
    
def test_milvus_storage():

    # Initialise MilvusStorage 
    vector_dim = 4  # The dimension of vectors to be stored
    url = "https://in03-56cdc4f4c89fcd6.api.gcp-us-west1.zillizcloud.com" # Replace with the URL of your Milvus server (or try my account)
    api ="c9e014b9e7593495ac2d69acb5e4fa3760024774142376df113a044fee3e61bd136b642e74578ed176e3c6a90ca9d52f1f211c3b" # Replace with your API key for the Milvus server.
    milvus_storage = MilvusStorage(vector_dim, (url, api)) # Creating an instance of MilvusStorage

    # Print the name of the collection used for testing
    print("Collection Name:", milvus_storage.collection_name)

    # Create some test VectorRecord dataset
    import random
    num_test_vector = 4
    test_records = []
    for _ in range(num_test_vector):
        vector = [random.random() for _ in range(vector_dim)]
        record = VectorRecord(vector=vector, payload={"info": "test payload"})
        test_records.append(record)

    # Get the test VectorRecord dataset id
    record_ids = [record.id for record in test_records]
    print("Record ids:", record_ids)

    ### Test Add function
        
    # Add dataset in Milvus
    milvus_storage.add(test_records)
    print("Added vectors to the collection: Add successful.")

    ### Test Search function.
     
    # Set search parameters
    search_params = {
        "data": [test_records[0].vector],  # Specifies the data to be searched, here it uses the first vector from the test records
        "anns_field": "vector",  #Specifies the field name in the collection used for ANN search
        "params": {"metric_type": "COSINE", "params": {"nprobe": 2}}, # Specifies parameters for the search algorithm, nprobe is the number of cluster centers
        "limit": 3, # Specifies the maximum number of similar vectors to return
        "output_fields": ["vector", "payload"]} # Given vector and payload imformation
    
    # Search required data in Milvus
    search_results = milvus_storage.client.search(collection_name=milvus_storage.collection_name, **search_params)
    print("Search Results:", search_results)

    ### Test Query function

    #Creat a new data to find the similarity in database 
    query_vector = [0.1 * i for i in range(vector_dim)]

    #Generate a VectorRecord list and setting the similiarties datapoint, here given the 2 most similar data (top_k) 
    query = VectorDBQuery(query_vector=query_vector, top_k=2)

    # Quary required data and find the similarity data in Milvus
    query_results = milvus_storage.query(query)
    print("Query results:")
    for result in query_results:
        print(f"id: {result.record.id}, Similarity: {result.similarity}")
        print(f"Vector: {result.record.vector}")
        if result.record.payload:
            print(f"Payload: {result.record.payload}")

    ### Test Status function
            
    """
    There is a lag on its server and zilliz web page, and the vector cannot be loaded immediately. 
    It usually takes more than 30 minutes to load all of it, but it also depends on the numbers of vector count.
    """  

    import time
    time.sleep(0)
    # Check the current status of milvus database
    status = milvus_storage.status()
    print("Collection Status:", status)

    ### Test Deletion function

    # Perform delete operation
    ids_to_delete = [test_records[0].id]
    print(f"Deleting id: {ids_to_delete[0]}")
    milvus_storage.delete(ids=ids_to_delete)
    print("Delete samlpe to the collection: Delete successful.")

    ### Test Clear function

    # Perform clear operation from milvus and create a new collection
    milvus_storage.clear()
    print("Collection cleared.")


if __name__ == "__main__":
    test_milvus_storage()
    
# %%
