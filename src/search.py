import chromadb
from sentence_transformers import SentenceTransformer
import json
import re

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "chekhov_texts"
MODEL_NAME = "intfloat/multilingual-e5-small"

def search_text(query: str, top_k: int = 5):
    print(f"\n🔎 Ищем ответ на вопрос: '{query}'...\n")
    
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # 1. Векторный поиск (по смыслу)
    query_embedding = model.encode([f"query: {query}"]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    # 2. Точный поиск по JSON (как Ctrl+F)
    names = re.findall(r'\b[А-ЯЁ][а-яё]+\b', query)
    target_name = names[1] if len(names) > 1 else None
    
    exact_found = []
    if target_name:
        print(f"💡 Выделено имя: '{target_name}'. Ищем точное совпадение в текстах...")
        try:
            with open(f"{CHROMA_DB_PATH}/chunks.json", "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            for chunk in all_chunks:
                if target_name in chunk['text']:
                    exact_found.append(chunk)
        except FileNotFoundError:
            print("⚠️ Файл chunks.json не найден. Запусти embed.py!")
    
    # 3. Вывод
    if exact_found:
        print("\n🎯 ТОЧНОЕ СОВПАДЕНИЕ НАЙДЕНО (Гибридный поиск):")
        for chunk in exact_found[:2]: # Показываем первые 2
            print(f"📖 {chunk['work']}")
            idx = chunk['text'].find(target_name)
            start = max(0, idx - 50)
            end = min(len(chunk['text']), idx + 200)
            print(f"📝 Контекст: ...{chunk['text'][start:end]}...\n")
    else:
        if target_name:
            print(f"❌ Точное совпадение для '{target_name}' не найдено в JSON.")

    print("--- Векторный поиск (для контекста): ---")
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        print(f"[{i+1}] {meta['work']} (Дистанция: {distance:.2f})")

if __name__ == "__main__":
    test_query = "Какое звание у персонажа Родэ?"
    search_text(test_query, top_k=5)