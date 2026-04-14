from pymongo import MongoClient
from app.core.config import get_settings
s = get_settings()
client = MongoClient(s.gencampus_mongo_uri, serverSelectionTimeoutMS=8000)
db = client['gencampus']

print("=== transcript_segments ===")
doc = db['transcript_segments'].find_one()
if doc:
    for k, v in doc.items():
        print(f"  {k}: {repr(str(v)[:80])} ({type(v).__name__})")
else:
    print("  empty")

print("\n=== transcripts ===")
doc = db['transcripts'].find_one()
if doc:
    for k, v in doc.items():
        print(f"  {k}: {repr(str(v)[:80])} ({type(v).__name__})")
else:
    print("  empty")

print(f"\ntranscript_segments count: {db['transcript_segments'].estimated_document_count()}")
print(f"transcripts count: {db['transcripts'].estimated_document_count()}")
client.close()
