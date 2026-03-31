from pymongo import MongoClient
from app.core.config import get_settings
s = get_settings()
client = MongoClient(s.gencampus_mongo_uri, serverSelectionTimeoutMS=8000)
db = client['gencampus']
for col in ['courseattendees', 'organizationusers', 'activities']:
    doc = db[col].find_one()
    if doc:
        print(f"\n{col} fields: {list(doc.keys())}")
        # Mostrar valores de campos que parecen IDs
        for k, v in doc.items():
            if v is not None:
                print(f"  {k}: {repr(v)[:60]} ({type(v).__name__})")
    else:
        print(f"{col} | empty")
client.close()
