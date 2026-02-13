# SIMPLE_RAG_TEST

🔍 Let’s Break Down the Code
1️⃣ Imports
import numpy as np
import faiss
from openai import OpenAI

What this does:

numpy → Handles arrays and vector math

faiss → Vector similarity search engine

OpenAI → Client to call embedding API

Think of it as:

OpenAI = vector creator
FAISS = vector search engine
NumPy = data handler

2️⃣ Create Client
client = OpenAI(api_key="YOUR_API_KEY")


This creates a connection to OpenAI servers.

When you call:

client.embeddings.create(...)


It sends a request to OpenAI and gets back a vector.

3️⃣ Your Documents
documents = [
    "Fiber cut can cause network outage.",
    "High latency often relates to routing congestion.",
    "Packet loss may indicate hardware failure."
]


These are your knowledge chunks.

In real life:

These would be PDF chunks

Or telecom logs

Or support tickets

Or troubleshooting docs

Each string will become a vector.

4️⃣ Create Embeddings
embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)


Let’s explain this carefully.

What is happening?

For each document:

Text → API → Vector (1536 numbers)


Example (simplified):

"Packet loss issue"
↓
[0.023, -0.51, 0.91, ... 1536 values]


That vector represents the meaning of the sentence.

Why 1536 numbers?

The embedding model outputs a fixed-size high-dimensional vector.

High dimension = more semantic information captured.

You can check:

print(len(response.data[0].embedding))

5️⃣ Convert to NumPy Array
embeddings = np.array(embeddings).astype("float32")


FAISS requires:

NumPy array

float32 type

Now your shape is:

(3 documents, 1536 dimensions)


Think of it like a matrix:

[
  [vector1],
  [vector2],
  [vector3]
]

🚀 Now We Build the Search Engine
6️⃣ Create FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

What this does:

IndexFlatL2 means:

Use L2 distance (Euclidean distance)

No approximation

Exact nearest neighbor search

You are telling FAISS:

"Store these vectors and allow similarity search."

Now FAISS has your document vectors stored internally.

🔎 Query Time
7️⃣ Convert Query to Embedding
query = "Why is there packet loss?"

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding


Important:

You MUST use the same embedding model for:

Documents

Query

Otherwise vector space will mismatch.

8️⃣ Convert Query to Proper Shape
query_vector = np.array([query_embedding]).astype("float32")


Why [query_embedding]?

Because FAISS expects 2D array:

(number_of_queries, dimension)


So shape becomes:

(1, 1536)

9️⃣ Search Top-K
k = 2
distances, indices = index.search(query_vector, k)


This means:

"Find the 2 closest vectors to this query."

FAISS calculates distance between:

Query vector

Every stored vector

Smaller distance = more similar meaning.

Output Example
indices = [[2, 1]]


That means:

Document 2 is most similar

Document 1 is second most similar

🔟 Retrieve Text
for idx in indices[0]:
    print(documents[idx])


We use the index numbers to get original text.

This is the final semantic result.

🧠 Why This Works (Deep Understanding)

Embeddings convert meaning into geometry.

Similar meaning → closer vectors.

Example:

"Packet loss problem"
"Network packet dropping"


Even though words differ,
their vectors are close.

That’s why this beats keyword search.

🏗️ How This Becomes RAG

Right now you only retrieved documents.

Next step:

context = top_k_chunks


Then send:

Answer using ONLY this context:
{context}
Question: {query}