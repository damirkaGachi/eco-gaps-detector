import re
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from docx import Document

def extract_numbers(text):
    """Извлекает только числа-нормы (не номера статей и не числа из дат)"""
    # Удаляем даты формата ДД.ММ.ГГГГ или ДД.ММ.ГГ
    text = re.sub(r'\b\d{2}\.\d{2}\.(?:\d{4}|\d{2})\b', '', text)
    
    # Удаляем номера статей и пунктов
    text = re.sub(r'(?:статья|ст\.|пункт|п\.|раздел|глав[аы]|часть|параграф|§)\s*\d+(?:\.\d+)?', '', text, flags=re.IGNORECASE)
    
    # Удаляем числа, которые идут перед словами "март", "февраль", "январь" и т.д.
    text = re.sub(r'\d+\s*(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])', '', text, flags=re.IGNORECASE)
    
    # Из оставшегося извлекаем числа
    numbers = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', text) if x]
    
    return numbers

def extract_dates(text):
    """Извлекает даты в форматах ДД.ММ.ГГГГ или ДД.ММ.ГГ"""
    # Стандартные даты с точками
    dates = re.findall(r'\b\d{2}\.\d{2}\.(?:\d{4}|\d{2})\b', text)
    
    # Даты в формате "1 марта", "15 февраля"
    month_dates = re.findall(r'(\d+)\s*(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)', text, re.IGNORECASE)
    
    # Превращаем "1" в "01", "15" в "15" (храним как строки для единообразия)
    for md in month_dates:
        dates.append(f"{md.zfill(2)}.мм.гг")
    
    return dates

def extract_modality(text):
    """Определяет обязательность действия: должен / может / запрещено"""
    # Обязательные действия
    mandatory = re.findall(r'(?:обязан|должен|необходимо|требуется|надлежит|следует|обязуется)', text, re.IGNORECASE)
    # Разрешительные действия
    permissive = re.findall(r'(?:может|вправе|имеет право|допускается|разрешается)', text, re.IGNORECASE)
    # Запретительные действия
    prohibited = re.findall(r'(?:запрещен|не допускается|не вправе|не может|не должен|нельзя)', text, re.IGNORECASE)
    
    if mandatory:
        return 'обязательно'
    elif prohibited:
        return 'запрещено'
    elif permissive:
        return 'разрешено'
    else:
        return 'не указано'


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


# ========== 2. ЗАГРУЖАЕМ ВСЕ ПАРЫ ДОКУМЕНТОВ ИЗ ПОДПАПОК ==========
folder_path = "./documents"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Создана папка {folder_path}. Создайте внутри подпапки с парами документов.")
    exit(1)

# Получаем список всех подпапок
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

if len(subfolders) == 0:
    print(f"В папке {folder_path} нет подпапок. Создайте подпапки (например, pair1, pair2) и положите в них документы.")
    exit(1)

print(f"Найдено {len(subfolders)} подпапок. Будет обработано {len(subfolders)} пар документов.\n")

# Для каждой подпапки обрабатываем свою пару документов
all_results = []

for subfolder in subfolders:
    pair_name = os.path.basename(subfolder)
    print(f"=" * 50)
    print(f"Обработка пары: {pair_name}")
    print(f"=" * 50)
    
    all_chunks = []
    chunk_to_file = []
    
    # Загружаем все файлы в текущей подпапке
    for filename in os.listdir(subfolder):
        if filename.lower().endswith(('.docx', '.txt')):
            filepath = os.path.join(subfolder, filename)
            text = load_text_from_file(filepath)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            chunk_to_file.extend([filename] * len(chunks))
            print(f"  Загружен {filename}: {len(chunks)} фрагментов")
    
    if len(all_chunks) < 2:
        print(f"  ⚠️ В паре {pair_name} меньше 2 файлов. Пропускаем.\n")
        continue
    
    print(f"  Всего фрагментов: {len(all_chunks)}")
    
    # Эмбеддинги и кластеризация
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    embeddings_norm = normalize(embeddings)
    
    clustering = DBSCAN(eps=0.45, min_samples=2, metric='cosine')
    clusters = clustering.fit_predict(embeddings_norm)
    
    # Поиск разрывов (такой же как был)
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
                                    'пара': pair_name,
                                    'файл_A': cluster_files[i],
                                    'файл_B': cluster_files[j],
                                    'фрагмент_A': cluster_texts[i][:300],
                                    'фрагмент_B': cluster_texts[j][:300],
                                    'тип': 'числовой',
                                    'значение_A': ni,
                                    'значение_B': nj
                                })
                if dates_list[i] and dates_list[j]:
                    for di in dates_list[i]:
                        for dj in dates_list[j]:
                            if di != dj:
                                gaps.append({
                                    'пара': pair_name,
                                    'файл_A': cluster_files[i],
                                    'файл_B': cluster_files[j],
                                    'фрагмент_A': cluster_texts[i][:300],
                                    'фрагмент_B': cluster_texts[j][:300],
                                    'тип': 'временной',
                                    'значение_A': di,
                                    'значение_B': dj
                                })
                
                # Модальность
                mod_i = extract_modality(cluster_texts[i])
                mod_j = extract_modality(cluster_texts[j])
                if mod_i != mod_j and mod_i != 'не указано' and mod_j != 'не указано':
                    gaps.append({
                        'пара': pair_name,
                        'файл_A': cluster_files[i],
                        'файл_B': cluster_files[j],
                        'фрагмент_A': cluster_texts[i][:300],
                        'фрагмент_B': cluster_texts[j][:300],
                        'тип': 'обязательство',
                        'значение_A': mod_i,
                        'значение_B': mod_j
                    })
    
    # Сохраняем результаты для этой пары
    if gaps:
        df = pd.DataFrame(gaps)
        output_file = f'разрывы_{pair_name}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  ✅ Найдено {len(gaps)} разрывов. Результат в '{output_file}'\n")
    else:
        print(f"  ✅ Разрывов не найдено\n")
    
    all_results.extend(gaps)

# Итоговый отчёт
print("=" * 50)
print("ОБЩИЙ ИТОГ")
print("=" * 50)
print(f"Обработано подпапок: {len(subfolders)}")
print(f"Всего найдено разрывов: {len(all_results)}")
