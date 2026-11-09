import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_all, DATA_DIR

# 1. Настройки
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "chekhov_texts"
MODEL_NAME = "intfloat/multilingual-e5-small"

def build_database():
    print("1. Загрузка чанков из файлов...")
    chunks = load_all(DATA_DIR)
    if not chunks:
        print("Ошибка: Чанки не найдены! Проверь папку data/")
        return

    print(f"Загружено чанков: {len(chunks)}")

    print("2. Инициализация модели эмбеддингов...")
    model = SentenceTransformer(MODEL_NAME)

    print("3. Подключение к ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Удаляем старую коллекцию, чтобы не было дублей
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)

    print("4. Генерация эмбеддингов и запись в базу...")
    
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        
        # Для моделей e5 нужен префикс "passage: "
        embeddings = model.encode([f"passage: {t}" for t in texts]).tolist()
        
        metadatas = [
            {
                "work": c.work,
                "author": c.author,
                "chunk_id": c.chunk_id,
                "paragraph_start": c.paragraph_start
            } for c in batch
        ]
        ids = [c.chunk_id for c in batch]

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  Обработано {min(i + batch_size, len(chunks))} из {len(chunks)}")

    # --- СОХРАНЯЕМ В JSON ДЛЯ ТОЧНОГО ПОИСКА ---
    print("5. Сохранение текстов в JSON для текстового поиска...")
    data_to_save = [{"text": c.text, "work": c.work, "chunk_id": c.chunk_id} for c in chunks]
    # Создаем папку, если ее нет
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    with open(f"{CHROMA_DB_PATH}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово! База сохранена в папку '{CHROMA_DB_PATH}'")

if __name__ == "__main__":
    build_database()