import os
# Отключаем проверку обновлений в интернете
os.environ["HF_HUB_OFFLINE"] = "1" 

import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "chekhov_texts"
MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL = "qwen2.5:7b"

def find_context(query: str):
    """Ищет текст в базе (векторный поиск + точный поиск по JSON)"""
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # 1. Векторный поиск (берем только 1 кусок), query - вопрос
    query_embedding = model.encode([f"query: {query}"]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=1) # Только (1) один!
    
    context_texts = []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        context_texts.append(f"Произведение: {meta['work']}\nТекст:\n{text}")

    # 2. Точный поиск по JSON (если в вопросе есть имя собственное)
    names = re.findall(r'\b[А-ЯЁ][а-яё]+\b', query)
    target_name = names[1] if len(names) > 1 else None
    
    if target_name: #JSON!
        try:
            with open(f"{CHROMA_DB_PATH}/chunks.json", "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            for chunk in all_chunks:
                if target_name in chunk['text']:
                    # Ставим точный кусок в самое начало
                    exact_text = f"!!! ВАЖНЫЙ ФРАГМЕНТ !!!\nПроизведение: {chunk['work']}\nТекст:\n{chunk['text']}"
                    context_texts.insert(0, exact_text)
                    break # Остановка
        except FileNotFoundError:
            print("  ⚠️ Файл chunks.json не найден. Запусти embed.py!")
            
    return "\n\n---\n\n".join(context_texts) #Склейка кусков

def ask_llm(query: str):
    print(f"\n🔎 Вопрос: '{query}'")
    
    print("  [1/4] Загрузка модели для поиска (e5) и базы ChromaDB...")
    context = find_context(query)
    print(f"  [2/4] Контекст найден! (Размер: {len(context)} символов).")
    
    # Промпт с примером, чтобы модель не галлюцинировала
    prompt = f"""Ты — литературный помощник. Твоя задача — отвечать на вопросы, строго цитируя текст.
Не придумывай слова, которых нет в тексте. Копируй точные слова из отрывка.

Пример правильного ответа:
ВОПРОС: Какое звание у Ивана?
ТЕКСТ: ...Иван Петрович, поручик.
ОТВЕТ: Поручик.

А теперь твой черед:

ТЕКСТ:
{context}

ВОПРОС: {query}
ОТВЕТ:"""

    print("  [3/4] Отправка запроса в Ollama (модель qwen2.5:7b)...")
    print("        ⏳ Пожалуйста, подождите. Генерация ответа...")
    
    try:
        response = ollama.chat(
            model=LLM_MODEL, 
            messages=[{'role': 'user', 'content': prompt}],
            options={'keep_alive': '5m'}
        )
        print("  [4/4] Ответ получен!\n")
        
        print("💬 ОТВЕТ НЕЙРОСЕТИ:")
        print("-" * 50)
        print(response['message']['content'])
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА при работе с Ollama: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("📚 Литературный RAG-помощник по текстам А. П. Чехова")
    print("=" * 50)
    print("Введите 'выход' или 'exit', чтобы завершить работу.\n")
    
    while True:
        user_query = input("👉 Ваш вопрос: ")
        
        if user_query.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break
            
        if not user_query.strip():
            continue
            
        ask_llm(user_query)
        print("\n" + "-" * 50 + "\n")