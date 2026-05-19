import re
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from docx import Document

# Функция: извлекает числа-нормы (игнорирует номера статей, даты, части больших чисел)
def extract_numbers(text):
    text = re.sub(r'\b\d{2}\.\d{2}\.(?:\d{4}|\d{2})\b', '', text)
    text = re.sub(r'(?:статья|ст\.|пункт|п\.|раздел|глав[аы]|часть|параграф|§)\s*\d+(?:\.\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])', '', text, flags=re.IGNORECASE)
    text_fixed = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    all_numbers = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', text_fixed) if x]
    important_words = r'штраф|пен[яю]|тонн|лимит|норматив|сброс|объём|количество|руб'
    numbers = []
    for num in all_numbers:
        num_str = str(int(num)) if num.is_integer() else str(num)
        context = text[max(0, text.find(num_str)-50):text.find(num_str)+50] if num_str in text else ''
        if num < 1000 and not re.search(important_words, context, re.IGNORECASE):
            continue
        if any(x for x in all_numbers if x == num * 1000 or x == num * 100):
            continue
        numbers.append(num)
    return numbers

# Функция: извлекает даты (ДД.ММ.ГГГГ или "1 марта")
def extract_dates(text):
    dates = re.findall(r'\b\d{2}\.\d{2}\.(?:\d{4}|\d{2})\b', text)
    month_dates = re.findall(r'(\d+)\s*(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)', text, re.IGNORECASE)
    for md in month_dates:
        dates.append(f"{md.zfill(2)}.мм.гг")
    return dates

# Функция: определяет модальность (обязательно/разрешено/запрещено)
def extract_modality(text):
    mandatory = re.findall(r'(?:обязан|должен|необходимо|требуется|надлежит|следует|обязуется)', text, re.IGNORECASE)
    permissive = re.findall(r'(?:может|вправе|имеет право|допускается|разрешается)', text, re.IGNORECASE)
    prohibited = re.findall(r'(?:запрещен|не допускается|не вправе|не может|не должен|нельзя)', text, re.IGNORECASE)
    if mandatory:
        return 'обязательно'
    elif prohibited:
        return 'запрещено'
    elif permissive:
        return 'разрешено'
    else:
        return 'не указано'

# Функция: извлекает штрафы и меры ответственности
def extract_liability(text):
    results = []
    patterns = [
        (r'штраф\s*(?:в\s*размере\s*)?(\d+(?:\s*\d+)?)\s*(?:тыс\.?|тысяч)?\s*(?:рублей|руб\.)', 'штраф'),
        (r'пен[яю]\s*(?:в\s*размере\s*)?(\d+(?:\s*\d+)?)\s*(?:тыс\.?|тысяч)?\s*(?:рублей|руб\.)', 'пеня'),
        (r'административный\s+штраф\s*(?:в\s*размере\s*)?(\d+(?:\s*\d+)?)\s*(?:тыс\.?|тысяч)?\s*(?:рублей|руб\.)', 'админ_штраф'),
    ]
    for pattern, typ in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            value_str = match.replace(' ', '') if isinstance(match, str) else str(match[0] if isinstance(match, tuple) else match)
            try:
                value = float(value_str)
                if 'тыс' in text[max(0, text.find(str(match))-20):text.find(str(match))+20]:
                    value = value * 1000
                results.append({'тип': typ, 'значение': value, 'единица': 'руб', 'категория': 'денежная', 'исходный': match})
            except:
                pass
    measures = {
        'предупреждение': ('предупреждение', 1),
        'дисквалификация': ('дисквалификация', 2),
        'административный арест': ('адм_арест', 3),
        'приостановление деятельности': ('приостановка', 4),
        'уголовная ответственность': ('уголовная', 5),
    }
    for measure, (code, severity) in measures.items():
        if re.search(measure, text, re.IGNORECASE):
            results.append({'тип': 'мера_ответственности', 'значение': code, 'уровень': severity, 'категория': 'неденежная', 'исходный': measure})
    return results

# Функция: загружает текст из файла (.txt, .docx)
def load_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.docx':
        doc = Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs])
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

# Функция: разбивает текст на фрагменты (чанки) по ~500 символов
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

# ========== ОСНОВНОЙ БЛОК ==========
folder_path = "./documents"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Создана папка {folder_path}. Создайте внутри подпапки с парами документов.")
    exit(1)

subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

if len(subfolders) == 0:
    print(f"В папке {folder_path} нет подпапок. Создайте подпапки (например, pair1, pair2) и положите в них документы.")
    exit(1)

print(f"Найдено {len(subfolders)} подпапок. Будет обработано {len(subfolders)} пар документов.\n")

all_results = []

for subfolder in subfolders:
    pair_name = os.path.basename(subfolder)
    print(f"=" * 50)
    print(f"Обработка пары: {pair_name}")
    print(f"=" * 50)
    
    all_chunks = []
    chunk_to_file = []
    
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
    
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    embeddings_norm = normalize(embeddings)
    
    clustering = DBSCAN(eps=0.45, min_samples=2, metric='cosine')
    clusters = clustering.fit_predict(embeddings_norm)
    
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
                                gaps.append({'пара': pair_name, 'файл_A': cluster_files[i], 'файл_B': cluster_files[j], 'фрагмент_A': cluster_texts[i][:300], 'фрагмент_B': cluster_texts[j][:300], 'тип': 'числовой', 'значение_A': ni, 'значение_B': nj})
                if dates_list[i] and dates_list[j]:
                    for di in dates_list[i]:
                        for dj in dates_list[j]:
                            if di != dj:
                                gaps.append({'пара': pair_name, 'файл_A': cluster_files[i], 'файл_B': cluster_files[j], 'фрагмент_A': cluster_texts[i][:300], 'фрагмент_B': cluster_texts[j][:300], 'тип': 'временной', 'значение_A': di, 'значение_B': dj})
                
                mod_i = extract_modality(cluster_texts[i])
                mod_j = extract_modality(cluster_texts[j])
                if mod_i != mod_j and mod_i != 'не указано' and mod_j != 'не указано':
                    gaps.append({'пара': pair_name, 'файл_A': cluster_files[i], 'файл_B': cluster_files[j], 'фрагмент_A': cluster_texts[i][:300], 'фрагмент_B': cluster_texts[j][:300], 'тип': 'обязательство', 'значение_A': mod_i, 'значение_B': mod_j})
                
                liability_i = extract_liability(cluster_texts[i])
                liability_j = extract_liability(cluster_texts[j])
                if liability_i and liability_j:
                    for li in liability_i:
                        for lj in liability_j:
                            if li['категория'] == lj['категория']:
                                if li['категория'] == 'денежная' and lj['категория'] == 'денежная':
                                    if li['значение'] != lj['значение']:
                                        gaps.append({'пара': pair_name, 'файл_A': cluster_files[i], 'файл_B': cluster_files[j], 'фрагмент_A': cluster_texts[i][:300], 'фрагмент_B': cluster_texts[j][:300], 'тип': f"ответственность_{li['тип']}", 'значение_A': f"{li['значение']} {li['единица']}", 'значение_B': f"{lj['значение']} {lj['единица']}"})
                                elif li['категория'] == 'неденежная' and lj['категория'] == 'неденежная':
                                    if li['уровень'] != lj['уровень']:
                                        gaps.append({'пара': pair_name, 'файл_A': cluster_files[i], 'файл_B': cluster_files[j], 'фрагмент_A': cluster_texts[i][:300], 'фрагмент_B': cluster_texts[j][:300], 'тип': 'ответственность_мера', 'значение_A': li['значение'], 'значение_B': lj['значение']})
    
    if gaps:
        df = pd.DataFrame(gaps)
        output_file = f'разрывы_{pair_name}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  ✅ Найдено {len(gaps)} разрывов. Результат в '{output_file}'\n")
    else:
        print(f"  ✅ Разрывов не найдено\n")
    
    all_results.extend(gaps)

print("=" * 50)
print("ОБЩИЙ ИТОГ")
print("=" * 50)
print(f"Обработано подпапок: {len(subfolders)}")
print(f"Всего найдено разрывов: {len(all_results)}")