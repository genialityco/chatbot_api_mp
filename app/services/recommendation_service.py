from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pymongo import MongoClient

from app.core.config import get_settings
from app.models.platform import Platform
from app.models.recommendation import RecommendationItem
from app.services.history_service import get_user_history

MAX_HISTORY_TURNS = 80
MAX_DOCS_PER_COLLECTION = 30
MAX_RECOMMENDATIONS = 5

_STOPWORDS = {
    "para", "como", "sobre", "entre", "donde", "desde", "hasta", "porque", "cuando",
    "quiero", "puedes", "podrias", "podrías", "favor", "gracias", "hola", "buenas",
    "curso", "cursos", "actividad", "actividades", "evento", "eventos", "memoria", "memorias",
    "tell", "about", "please", "need", "want", "with", "from", "that", "this",
}

_TEXT_FIELDS = ("name", "title", "description", "summary", "text", "content", "topic")

_PLATFORM_COLLECTIONS = {
    "gencampus": ("events", "activities"),
    "acho": ("events", "memories", "highlights"),
}


def _extract_keywords(history: list[dict], max_keywords: int = 8) -> list[str]:
    corpus = " ".join((h.get("user", "") or "") for h in history)
    tokens = re.findall(r"[a-zA-Z0-9áéíóúÁÉÍÓÚñÑ]{4,}", corpus.lower())
    filtered = [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]
    if not filtered:
        return []
    return [w for w, _ in Counter(filtered).most_common(max_keywords)]


def _build_regex_filter(keywords: list[str]) -> dict:
    if not keywords:
        return {}
    regex = "|".join(re.escape(k) for k in keywords)
    return {"$or": [{field: {"$regex": regex, "$options": "i"}} for field in _TEXT_FIELDS]}


def _score_doc(doc: dict[str, Any], keywords: list[str]) -> float:
    text = " ".join(str(doc.get(f, "")) for f in _TEXT_FIELDS).lower()
    if not text:
        return 0.0
    return float(sum(1 for kw in keywords if kw in text))


def _pick_title(doc: dict[str, Any]) -> str:
    for key in ("name", "title", "topic", "_id"):
        value = doc.get(key)
        if value:
            return str(value)
    return "Sin título"


def _pick_summary(doc: dict[str, Any]) -> str:
    for key in ("description", "summary", "text", "content"):
        value = doc.get(key)
        if value:
            s = str(value).strip()
            return s[:220] + ("..." if len(s) > 220 else "")
    return ""


def _build_url(platform_id: str, collection: str, item_id: str, org_id: str | None) -> str | None:
    settings = get_settings()
    if platform_id == "gencampus" and org_id:
        base_url = settings.gencampus_base_url
        if collection == "events":
            return f"{base_url}/organization/{org_id}/course/{item_id}"
        if collection == "activities":
            return f"{base_url}/organization/{org_id}/activitydetail/{item_id}"
    return None


class RecommendationService:
    def __init__(self, platform: Platform, org_id: str | None = None):
        self.platform = platform
        self.platform_id = platform.platform_id
        self.org_id = org_id

    async def recommend_for_user(self, user_id: str, limit: int = MAX_RECOMMENDATIONS) -> tuple[list[RecommendationItem], list[str]]:
        history = await get_user_history(self.platform_id, user_id, limit=MAX_HISTORY_TURNS)
        keywords = _extract_keywords(history)
        if not keywords:
            return [], []

        target_collections = _PLATFORM_COLLECTIONS.get(self.platform_id, ("events",))
        dedupe: set[str] = set()
        items: list[RecommendationItem] = []

        for conn in self.platform.db_connections:
            allowed = set(conn.collections or target_collections)
            collections = [c for c in target_collections if c in allowed]
            if not collections:
                continue

            client = MongoClient(conn.uri, serverSelectionTimeoutMS=8000)
            try:
                db = client[conn.database]
                query_filter = _build_regex_filter(keywords)
                for collection in collections:
                    docs = list(db[collection].find(query_filter, limit=MAX_DOCS_PER_COLLECTION))
                    for doc in docs:
                        item_id = str(doc.get("_id", ""))
                        if not item_id:
                            continue
                        dedupe_key = f"{collection}:{item_id}"
                        if dedupe_key in dedupe:
                            continue
                        dedupe.add(dedupe_key)

                        score = _score_doc(doc, keywords)
                        if score <= 0:
                            continue

                        items.append(
                            RecommendationItem(
                                collection=collection,
                                item_id=item_id,
                                title=_pick_title(doc),
                                summary=_pick_summary(doc),
                                score=score,
                                url=_build_url(self.platform_id, collection, item_id, self.org_id),
                                metadata={"database": conn.database},
                            )
                        )
            except Exception as e:
                print(f"[recommendations] query error ({conn.database}): {e}")
            finally:
                client.close()

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:min(limit, MAX_RECOMMENDATIONS)], keywords
