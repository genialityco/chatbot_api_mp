import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def main():
    uri = os.getenv('GENCAMPUS_MONGO_URI')
    db_name = os.getenv('GENCAMPUS_MONGO_DB')
    client = AsyncIOMotorClient(uri)
    db = client[db_name]
    
    meta_phone = "573104365063"
    
    possible_phones = [meta_phone, f"+{meta_phone}"]
    if len(meta_phone) >= 10:
        last_10 = meta_phone[-10:]
        possible_phones.append(last_10)
        try:
            possible_phones.append(float(last_10))
        except ValueError:
            pass
            
    print("Buscando con variaciones:", possible_phones)
    
    docs = await db.organizationusers.find({'properties.phone': {'$in': possible_phones}}).to_list(None)
    for d in docs:
        phone = d.get('properties', {}).get('phone')
        user_id = d.get('user_id')
        print(f"ENCONTRADO: {phone} (Type: {type(phone)}), user_id: {user_id} (Type: {type(user_id)})")
    
    client.close()

asyncio.run(main())
