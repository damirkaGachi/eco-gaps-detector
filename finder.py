import re
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from docx import Document

# ========== 1. ЗАГРУЗКА ТЕКСТА ИЗ ФАЙЛОВ ==========
def load_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.docx':
        doc = Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs])
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

def chunk_text(text, chunk_size=500):
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        current_len += len(sent)
        current_chunk.append(sent)
        if current_len >= chunk_size:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks if chunks else [text]

# ========== 2. ЗАГРУЖАЕМ ВСЕ ДОКУМЕНТЫ ==========
folder_path = "./documents"
all_chunks = []
chunk_to_file = []

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Создана папка {folder_path}. Положите туда документы и запустите снова.")
    exit(1)

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.docx', '.txt')):
        text = load_text_from_file(os.path.join(folder_path, filename))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        chunk_to_file.extend([filename] * len(chunks))
        print(f"Загружен {filename}: {len(chunks)} фрагментов")

if len(all_chunks) == 0:
    print("В папке documents нет файлов .txt или .docx")
    exit(1)

print(f"\nВсего фрагментов: {len(all_chunks)}")

# ========== 3. ЭМБЕДДИНГИ И КЛАСТЕРИЗАЦИЯ ==========
print("Загружаем модель rubert-tiny2...")
model = SentenceTransformer('cointegrated/rubert-tiny2')
embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings_norm = normalize(embeddings)

clustering = DBSCAN(eps=0.45, min_samples=2, metric='cosine')
clusters = clustering.fit_predict(embeddings_norm)

# ========== 4. ПОИСК ЧИСЕЛ И ДАТ ==========
def extract_numbers(text):
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?', text) if x]

def extract_dates(text):
    return re.findall(r'\b\d{2}\.\d{2}\.(?:\d{4}|\d{2})\b', text)

# ========== 5. ПОИСК РАЗРЫВОВ ==========
gaps = []
for cluster_id in set(clusters):
    if cluster_id == -1:
        continue
    idxs = np.where(clusters == cluster_id)[0]
    if len(idxs) < 2:
        continue
    cluster_texts = [all_chunks[i] for i in idxs]
    cluster_files = [chunk_to_file[i] for i in idxs]
    numbers_list = [extract_numbers(t) for t in cluster_texts]
    dates_list = [extract_dates(t) for t in cluster_texts]
    
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if numbers_list[i] and numbers_list[j]:
                for ni in numbers_list[i]:
                    for nj in numbers_list[j]:
                        if ni == 0 and nj == 0:
                            continue
                        max_val = max(abs(ni), abs(nj))
                        if max_val > 0 and abs(ni - nj) / max_val > 0.2:
                            gaps.append({
                                'файл_A': cluster_files[i],
                                'файл_B': cluster_files[j],
                                'фрагмент_A': cluster_texts[i][:300],
                                'фрагмент_B': cluster_texts[j][:300],
                                'тип': 'числовой',
                                'значение_A': ni,
                                'значение_B': nj,
                                'кластер': int(cluster_id)
                            })
            if dates_list[i] and dates_list[j]:
                for di in dates_list[i]:
                    for dj in dates_list[j]:
                        if di != dj:
                            gaps.append({
                                'файл_A': cluster_files[i],
                                'файл_B': cluster_files[j],
                                'фрагмент_A': cluster_texts[i][:300],
                                'фрагмент_B': cluster_texts[j][:300],
                                'тип': 'временной',
                                'значение_A': di,
                                'значение_B': dj,
                                'кластер': int(cluster_id)
                            })

df_gaps = pd.DataFrame(gaps) if gaps else pd.DataFrame(columns=['файл_A', 'файл_B', 'фрагмент_A', 'фрагмент_B', 'тип', 'значение_A', 'значение_B', 'кластер'])
if len(df_gaps) > 0:
    df_gaps = df_gaps.drop_duplicates(subset=['файл_A', 'файл_B', 'тип', 'значение_A', 'значение_B'])

df_gaps.to_csv('разрывы.csv', index=False, encoding='utf-8-sig')

print(f"\n✅ Найдено {len(df_gaps)} разрывов. Результат в файле 'разрывы.csv'")
if len(df_gaps) > 0:
    print("\nПримеры:")
    for _, row in df_gaps.head(3).iterrows():
        print(f"- {row['файл_A']} vs {row['файл_B']}: {row['тип']} ({row['значение_A']} ≠ {row['значение_B']})")