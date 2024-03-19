#%%
from typing import List
from camel.storages.vectordb_storages.milvusDB_v2 import MilvusStorage
import random
import time
from camel.storages.vectordb_storages import (
    BaseVectorStorage,
    VectorDBQuery,
    VectorDBQueryResult,
    VectorDBStatus,
    VectorRecord,
)

vector_dim = 4
milvus_storage = MilvusStorage(vector_dim=vector_dim, url_and_api=(
    "https://in03-56cdc4f4c89fcd6.api.gcp-us-west1.zillizcloud.com", 
    "c9e014b9e7593495ac2d69acb5e4fa3760024774142376df113a044fee3e61bd136b642e74578ed176e3c6a90ca9d52f1f211c3b"),
    )
# Generation of test data
def generate_test_records(num_records=4, vector_dim=vector_dim):
    records = []
    for _ in range(num_records):
        vector = [random.random() for _ in range(vector_dim)]
        record = VectorRecord(vector=vector, payload={"info": "test payload"})
        records.append(record)
    return records

print("Collection Name:", milvus_storage.collection_name)

test_records = generate_test_records()
record_ids = [record.id for record in test_records]
print("Record IDs:", record_ids)

add_data = milvus_storage.add(test_records)
print("____________________________Add successful_______________________________")

#%%
status = milvus_storage.status()
print("Collection Status:", status)
#%%
# Operation Search
search_params = {
    "data": [test_records[0].vector],  
    "anns_field": "vector",  
    "params": {"metric_type": "CONSINE", "params": {"nprobe": 10}},
    "limit": 3,
    "output_fields": ["vector", "payload"]}
search_results = milvus_storage.client.search(collection_name=milvus_storage.collection_name, **search_params)

# Print the results after similarity comparsion
print("Search Results:", search_results)

#%%
query_vector = [0.1 * i for i in range(vector_dim)]
query = VectorDBQuery(query_vector=query_vector, top_k=5)  # 假设我们要找到最相似的前5个向量
query_results = milvus_storage.query(query)

# 打印查询结果
print("Query results:")
for result in query_results:
    print(f"ID: {result.record.id}, Similarity: {result.similarity}")
    print(f"Vector: {result.record.vector}")
    if result.record.payload:
        print(f"Payload: {result.record.payload}")
# Setting search parameters

#%%
status_before_deletion = milvus_storage.status()
print("Status before deletion:")
print(f"Total vectors before deletion: {status_before_deletion['total_vectors']}")

# 执行删除操作
ids_to_delete = [test_records[0].id]
print(f"Deleting ID: {ids_to_delete[0]}")
milvus_storage.delete(ids=ids_to_delete)

# 暂停一段时间，确保删除操作完成
import time
time.sleep(2)  # 根据实际情况调整等待时间

# 显示删除后的集合状态
status_after_deletion = milvus_storage.status()
print("Status after deletion:")
print(f"Total vectors after deletion: {status_after_deletion['total_vectors']}")

#%%
# 显示删除前的状态
status_before_clear = milvus_storage.status()
print("Status before clear operation:")
print(f"Collection ID: {status_before_clear['collection_id']}, Vector Dimension: {status_before_clear['vector_dim']}, Total Vectors: {status_before_clear['total_vectors']}")

# 执行删除操作
milvus_storage.clear()

# 等待一小段时间确保删除操作完成，根据实际情况可能需要调整
import time
time.sleep(2)  # 等待2秒

# 显示删除后的状态
status_after_clear = milvus_storage.status()
print("Status after clear operation:")
print(f"Collection ID: {status_after_clear['collection_id']}, Vector Dimension: {status_after_clear['vector_dim']}, Total Vectors: {status_after_clear['total_vectors']}")



# %%
