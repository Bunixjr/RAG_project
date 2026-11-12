import os
import re
from dataclasses import dataclass, asdict

DATA_DIR = "data" # Где лежат тексты 

MAPPING = {
    "chekhov_chaika": ("Чайка", "А. П. Чехов"),
    "chekhov_tri_sestry": ("Три сестры", "А. П. Чехов"),
    "chekhov_palata_6": ("Палата №6", "А. П. Чехов"),
    "chekhov_vishnevyy_sad": ("Вишнёвый сад", "А. П. Чехов"),
}

@dataclass
class Chunk:
    chunk_id: str
    text: str
    work: str
    author: str
    paragraph_start: int

def split_into_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n") # Windows и linux
    paragraphs = re.split(r"\n\s*\n", text) # Реж там где пробелы
    return [p.strip() for p in paragraphs if p.strip()]

def make_chunks(paragraphs: list[str], max_chars: int = 1500) -> list[tuple[str, int]]:
    chunks = []
    current_chunk = []
    current_length = 0

    for i, para in enumerate(paragraphs):
        if len(para) > max_chars:
            if current_chunk:
                chunks.append(("\n\n".join(current_chunk), i - len(current_chunk)))
                current_chunk = []
                current_length = 0
            
            # Режем длинный абзац по предложениям (или просто по кускам)
            for j in range(0, len(para), max_chars):
                chunks.append((para[j:j+max_chars], i))
            continue

        # Если абзац короткий, добавляем к текущему
        current_chunk.append(para)
        current_length += len(para) + 2 # +2 за перенос строки

        # Если набрали достаточно текста — сохраняем чанк
        if current_length >= max_chars:
            chunks.append(("\n\n".join(current_chunk), i - len(current_chunk) + 1))
            current_chunk = []
            current_length = 0

    # Не забываем сохранить остаток
    if current_chunk:
        chunks.append(("\n\n".join(current_chunk), len(paragraphs) - len(current_chunk)))

    return chunks

def filename_to_meta(filename: str) -> tuple[str, str]:
    key = filename.rsplit(".", 1)[0]
    return MAPPING.get(key, (key, "А. П. Чехов"))

def parse_file(path: str) -> list[Chunk]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="windows-1251") as f:
            raw = f.read()
            
    filename = os.path.basename(path)
    work, author = filename_to_meta(filename)
    
    # Чистим текст от одиночных ремарок
    raw = re.sub(r'\n+(Пауза\.|Входит\.|Уходит\.|Смеется\.|Плачет\.)\n+', '\n\n', raw)
    
    paragraphs = split_into_paragraphs(raw)
    
    # Отсеиваем абзацы короче 30 символов
    paragraphs = [p for p in paragraphs if len(p) > 30]
    # Запуск чанкинга
    chunks = []
    for idx, (chunk_text, para_start) in enumerate(make_chunks(paragraphs)):
        chunks.append(Chunk(
            chunk_id=f"{filename}::{idx}",
            text=chunk_text,
            work=work,
            author=author,
            paragraph_start=para_start,
        ))
    return chunks

def load_all(data_dir: str = DATA_DIR) -> list[Chunk]:
    all_chunks = []
    if not os.path.exists(data_dir):
        return all_chunks
    for name in sorted(os.listdir(data_dir)):
        if name.endswith(".txt"):
            all_chunks.extend(parse_file(os.path.join(data_dir, name)))
    return all_chunks

if __name__ == "__main__":
    chunks = load_all()
    print(f"Загружено чанков: {len(chunks)}")
    for c in chunks[:3]:
        print("-" * 60)
        print(f"ID: {c.chunk_id}")
        print(f"Произведение: {c.work}")
        print(f"Текст: {c.text[:100]}...")