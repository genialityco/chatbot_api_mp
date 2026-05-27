"""Script para limpiar todas las FAQs candidatas/confirmadas y reiniciar el aprendizaje."""
import asyncio
import shutil
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.models.faq import FAQItem
from app.core.config import get_settings

settings = get_settings()


async def main():
    client = AsyncIOMotorClient(settings.meta_mongodb_uri)
    await init_beanie(database=client[settings.meta_mongodb_db], document_models=[FAQItem])

    result = await FAQItem.find_all().delete()
    print(f"FAQItems eliminados de MongoDB: {result.deleted_count}")
    client.close()

    faq_dir = os.path.join(settings.chroma_persist_dir, "faq")
    if os.path.exists(faq_dir):
        shutil.rmtree(faq_dir)
        print(f"ChromaDB FAQ eliminado: {faq_dir}")
    else:
        print("No habia directorio FAQ en ChromaDB")

    print("Reset completo. Listo para re-aprender.")


if __name__ == "__main__":
    asyncio.run(main())
