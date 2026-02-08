import csv

# 1. Загружаем все CSV файлы
def load_csv(filename):
    """Чтение CSV и возврат списка словарей"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Основные карточки
chronobiotic_data = load_csv('chronobiotic.csv')

# Дополнительные таблицы для ссылок и описаний
targets_data = load_csv('target.csv')
articles_data = load_csv('article.csv')
synonyms_data = load_csv('synonyms.csv')
classf_data = load_csv('class.csv')
chron_class_data = load_csv('chronobiotic_classf.csv')
effect_data = load_csv('effect.csv')
chron_effect_data = load_csv('chronobiotic_effect.csv')
mechanism_data = load_csv('mechanism.csv')
chron_mechanism_data = load_csv('chronobiotic_mechanisms.csv')
chron_articles_data = load_csv('chronobiotic_articles.csv')
chron_targets_data = load_csv('chronobiotic_target.csv')


# 2. Создаём словари для быстрого поиска по ID
targets_dict = {row['id']: row for row in targets_data}
articles_dict = {row['id']: row for row in articles_data}
synonyms_dict = {}
for row in synonyms_data:
    # несколько синонимов на одно вещество
    bid = row['originalbiotic_id']
    synonyms_dict.setdefault(bid, []).append(row['synonymsmname'])
class_dict = {row['id']: row['nameclass'] for row in classf_data}
effect_dict = {row['id']: row['Effectname'] for row in effect_data}
mechanism_dict = {row['id']: row['mechanismname'] for row in mechanism_data}

# 3. Создаём текстовый индекс
with open('chronobiotic_index_full.txt', 'w', encoding='utf-8') as f:
    f.write("=== ИНДЕКС CHRONOBIOTICSDB (полный) ===\n\n")

    for drug in chronobiotic_data:
        drug_id = drug['id']
        name = drug['gname']
        status = drug.get('fdastatus', 'Unknown')

        # Описание (обрезаем если длинное)
        desc = drug.get('description', '')
        if len(desc) > 300:
            desc = desc[:297] + "..."
        desc = ' '.join(desc.split())  # убираем лишние переносы

        # Ссылки из самого файла
        pubchem = drug.get('pubchem', '')
        kegg = drug.get('kegg', '')

        # Добавляем классы (через промежуточную таблицу chronobiotic_classf)
        classes = []
        for row in chron_class_data:
            if row['chronobiotic_id'] == drug_id:
                class_id = row['bioclass_id']
                classes.append(class_dict.get(class_id, 'Unknown'))

        # Добавляем эффекты
        effects = []
        for row in chron_effect_data:
            if row['chronobiotic_id'] == drug_id:
                effect_id = row['effect_id']
                effects.append(effect_dict.get(effect_id, 'Unknown'))

        # Добавляем механизмы
        mechanisms = []
        for row in chron_mechanism_data:
            if row['chronobiotic_id'] == drug_id:
                mechanism_id = row['mechanism_id']
                mechanisms.append(mechanism_dict.get(mechanism_id, 'Unknown'))

        # Добавляем цели (targets)
        targets = []
        for row in chron_targets_data:
            if row['chronobiotic_id'] == drug_id:
                target_id = row['targets_id']
                t = targets_dict.get(target_id)
                if t:
                    targets.append(f"{t['targetsname']} ({t['targeturl']})")


        # Добавляем статьи
        articles = []
        for row in chron_articles_data:
            if row['chronobiotic_id'] == drug_id:
                art_id = row['articles_id']
                a = articles_dict.get(str(art_id))
                if a:
                    articles.append(f"{a['articlename']}: {a['articleurl']}")

        # Добавляем синонимы
        synonyms = synonyms_dict.get(drug_id, [])

        # Пишем карточку в файл
        f.write(f"ID: {drug_id} | {name}\n")
        f.write(f"FDA: {status}\n")
        if desc:
            f.write(f"Описание: {desc}\n")
        if pubchem:
            f.write(f"PubChem: {pubchem}\n")
        if kegg:
            f.write(f"KEGG: {kegg}\n")
        if classes:
            f.write(f"Классы: {', '.join(classes)}\n")
        if effects:
            f.write(f"Эффекты: {', '.join(effects)}\n")
        if mechanisms:
            f.write(f"Механизмы: {', '.join(mechanisms)}\n")
        if targets:
            f.write(f"Цели: {', '.join(targets)}\n")
        if articles:
            f.write(f"Статьи: {', '.join(articles)}\n")
        if synonyms:
            f.write(f"Синонимы: {', '.join(synonyms)}\n")

        f.write("-" * 60 + "\n\n")

print(f"Полный индекс создан: chronobiotic_index_full.txt ({len(chronobiotic_data)} карточек)")
