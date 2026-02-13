import numpy as np
import faiss
from openai import OpenAI

# Replace with your API key
client = OpenAI(api_key="apikey")

documents = [
    "Fiber cut can cause network outage.",
    "High latency often relates to routing congestion.",
    "Packet loss may indicate hardware failure."
]

# Create embeddings
embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Query
query = "Why is there packet loss?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

query_vector = np.array([query_embedding]).astype("float32")

k = 2
distances, indices = index.search(query_vector, k)

print("Top relevant chunks:")
for idx in indices[0]:
    print("-", documents[idx])
