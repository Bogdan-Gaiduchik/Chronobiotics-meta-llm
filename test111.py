#!/usr/bin/env python3
"""ChronobioticsDB Meta-Prompting Agent - Final Version with Post-Processing"""

import torch
import time
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ============================================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================================

MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
CARDS_FILE = "chronobiotic_index_full.txt"
OUTPUT_LOG = "chronobiotics_final_results.txt"
PROMPT_FILE = "meta_generated_prompt.txt"

print("=" * 80)
print("CHRONOBIOTICSDB FINAL META-PROMPTING AGENT")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# 2. ПАРСЕР КАРТОЧЕК
# ============================================================================

def parse_chronobiotics_cards(file_path: str) -> Tuple[List[Dict], Dict, List[str]]:
    """Парсит файл с карточками ChronobioticsDB"""
    print(f"Parsing ChronobioticsDB cards from {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    raw_cards = content.split('------------------------------------------------------------')
    raw_cards = [card.strip() for card in raw_cards if card.strip()]
    
    cards = []
    cards_by_id = {}
    all_links = []
    
    for card_text in raw_cards:
        id_match = re.search(r'ID:\s*(\d+)\s*\|', card_text)
        if not id_match:
            continue
        
        card_id = int(id_match.group(1))
        
        name_match = re.search(r'ID:\s*\d+\s*\|\s*(.+?)\n', card_text)
        name = name_match.group(1).strip() if name_match else f"Card_{card_id}"
        
        # Извлекаем все ссылки
        card_links = re.findall(r'https?://[^\s<>"\'\)]+', card_text)
        
        card_data = {
            'id': card_id,
            'name': name,
            'full_text': card_text[:2000],
            'summary': f"ID: {card_id} | {name}",
            'links': card_links
        }
        
        cards.append(card_data)
        cards_by_id[card_id] = card_data
        all_links.extend(card_links)
    
    print(f"✓ Parsed {len(cards)} cards with {len(all_links)} links")
    return cards, cards_by_id, all_links

# ============================================================================
# 3. ШАГ 1: МЕТА-ПРОМПТИНГ - ГЕНЕРАЦИЯ ОПТИМАЛЬНОГО ПРОМПТА
# ============================================================================

def generate_meta_prompt_step(cards_sample: str, tokenizer, pipe) -> str:
    """
    ШАГ 1: Используем DeepSeek-R1 для создания оптимального системного промпта
    """
    print("\n" + "="*80)
    print("STEP 1: META-PROMPTING - Generating optimal system prompt")
    print("="*80)
    
    # Усиленный мета-промпт с явными требованиями
    meta_prompt = f"""You are DeepSeek-R1 in META-MODE. Your ONLY task is to write the PERFECT system prompt for answering questions about ChronobioticsDB.

CRITICAL REQUIREMENTS for the system prompt:

1. LANGUAGE POLICY (STRICT):
   - "Output language MUST be English ONLY"
   - "Never translate to Russian or any other language"
   - "If user asks in Russian, answer in English"

2. NO REASONING POLICY (ABSOLUTE):
   - "NEVER output <think> tags or chain-of-thought"
   - "NEVER show internal reasoning or analysis"
   - "Provide only final factual answers"
   - "No thinking aloud, no step-by-step"

3. DATA ADHERENCE (STRICT):
   - "Answer ONLY using information from provided cards"
   - "If data is not in cards: 'Data not found in ChronobioticsDB'"
   - "Always cite Card IDs: [Card ID: X]"

4. RESPONSE FORMAT (FIXED):
   - Start with: "Answer: [factual answer in English]"
   - Then: "Sources: [Card IDs and links used]"
   - Then: "Data Completeness: [What's available/missing]"

5. HANDLING NO DATA:
   - "If no cards match: 'Answer: Data not found in ChronobioticsDB'"
   - "Sources: None"
   - "Data Completeness: No information available"

SAMPLE CARD:
{cards_sample[:1500]}

Write the system prompt that enforces ALL these rules absolutely.
Make it clear, strict, and impossible to misinterpret.
"""
    
    start_time = time.time()
    
    messages = [{"role": "user", "content": meta_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    output = pipe(
        prompt_text,
        max_new_tokens=1500,
        temperature=0.3,
        do_sample=True,
        top_p=0.9
    )
    
    elapsed = time.time() - start_time
    
    generated_text = output[0]['generated_text']
    if "assistant\n" in generated_text:
        system_prompt = generated_text.split("assistant\n")[-1].strip()
    else:
        system_prompt = generated_text[len(prompt_text):].strip()
    
    print(f"✓ Meta-prompt generated in {elapsed:.1f}s")
    print(f"✓ Prompt length: {len(system_prompt)} characters")
    
    # Сохраняем сгенерированный промпт
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(system_prompt)
    
    print(f"✓ System prompt saved to {PROMPT_FILE}")
    
    return system_prompt

# ============================================================================
# 4. ПОСТ-ПРОЦЕССИНГ ОТВЕТОВ
# ============================================================================

def clean_response(response: str) -> str:
    """
    Очищает ответ от reasoning tags и лишнего текста
    Возвращает только финальный ответ в правильном формате
    """
    # Удаляем все <think> секции
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Удаляем любые chain-of-thought маркеры
    response = re.sub(r'(First|Then|Next|Finally|Step \d+|Let me think|Let\'s reason).*?(?=\n\n|\Z)', '', 
                     response, flags=re.DOTALL | re.IGNORECASE)
    
    # Находим начало фактического ответа
    # Ищем "Answer:" или создаем его если нет
    if "Answer:" not in response:
        # Если есть обычный текст, делаем его ответом
        lines = response.strip().split('\n')
        if lines and lines[0].strip():
            response = f"Answer: {lines[0].strip()}\n"
            if len(lines) > 1:
                response += '\n'.join(lines[1:])
        else:
            response = "Answer: Data not found in ChronobioticsDB\n"
    
    # Убедимся что формат правильный
    if not response.startswith("Answer:"):
        # Находим где начинается Answer
        answer_idx = response.find("Answer:")
        if answer_idx != -1:
            response = response[answer_idx:]
        else:
            response = f"Answer: {response}"
    
    # Убираем повторяющиеся пустые строки
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
    
    return response.strip()

def extract_main_answer(response: str) -> str:
    """Извлекает основную часть ответа для проверки качества"""
    # Берем текст между Answer: и следующими заголовками
    answer_match = re.search(r'Answer:\s*(.+?)(?=\n(?:Sources:|Data Completeness:|$))', 
                           response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return response

# ============================================================================
# 5. ШАГ 2: EXECUTOR-РЕЖИМ
# ============================================================================

class ChronobioticsExecutor:
    """
    ШАГ 2: Исполнитель с пост-процессингом
    """
    
    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        
        print("\n" + "="*80)
        print("STEP 2: EXECUTOR MODE - Initializing with strict parameters")
        print("="*80)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        print(f"✓ Executor model loaded on: {self.model.device}")
    
    def retrieve_relevant_cards(self, question: str, cards: List[Dict], limit: int = 6) -> List[Dict]:
        """Находит релевантные карточки"""
        question_lower = question.lower()
        relevant = []
        
        # Точное совпадение по названию
        for card in cards:
            card_name_lower = card['name'].lower()
            # Убираем возможные скобки из названия для сравнения
            clean_name = re.sub(r'\([^)]*\)', '', card_name_lower).strip()
            if clean_name in question_lower or card_name_lower in question_lower:
                relevant.append(card)
        
        # Поиск по ключевым словам
        if len(relevant) < 3:
            word_scores = {}
            for card in cards:
                if card in relevant:
                    continue
                    
                card_text = card['full_text'].lower()
                score = 0
                
                for word in question_lower.split():
                    if len(word) > 4 and word in card_text:
                        score += 1
                        if word in card['name'].lower():
                            score += 2
                
                if score > 0:
                    word_scores[card['id']] = {'card': card, 'score': score}
            
            # Сортируем и добавляем лучшие
            sorted_cards = sorted(word_scores.values(), key=lambda x: x['score'], reverse=True)
            for item in sorted_cards[:limit - len(relevant)]:
                relevant.append(item['card'])
        
        return relevant[:limit]
    
    def ask_question(self, question: str, relevant_cards: List[Dict]) -> Dict:
        """Задает вопрос с пост-процессингом"""
        print(f"\nQ: {question}")
        
        # ОСОБЫЙ СЛУЧАЙ: если нет релевантных карточек
        if not relevant_cards:
            print(f"  No relevant cards found")
            result = {
                "question": question,
                "response": "Answer: Data not found in ChronobioticsDB\nSources: None\nData Completeness: No information available",
                "cards_used": [],
                "card_names": [],
                "tokens_estimated": 0,
                "time": 0,
                "quality_check": {
                    "is_in_english": True,
                    "has_card_references": False,
                    "has_links": False,
                    "no_reasoning_tags": True,
                    "mentions_limitations": True,
                    "proper_format": True
                }
            }
            print(f"  ✓ No-data response generated")
            return result
        
        print(f"  Found {len(relevant_cards)} relevant cards: {[c['name'] for c in relevant_cards]}")
        
        # Формируем контекст
        context = "\n\n".join([card['full_text'] for card in relevant_cards])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CHRONOBIOTICSDB CARDS:\n\n{context}\n\nQUESTION: {question}\n\nProvide answer based ONLY on cards above."}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        start_time = time.time()
        
        try:
            output = self.pipe(
                prompt_text,
                max_new_tokens=6000,
                temperature=0.1,  # Строгий режим
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            elapsed = time.time() - start_time
            
            # Извлекаем сырой ответ
            full_output = output[0]['generated_text']
            if "assistant\n" in full_output:
                raw_response = full_output.split("assistant\n")[-1].strip()
            else:
                raw_response = full_output[len(prompt_text):].strip()
            
            print(f"  Raw response: {len(raw_response)} chars, {elapsed:.1f}s")
            
            # ПОСТ-ПРОЦЕССИНГ: очищаем ответ
            cleaned_response = clean_response(raw_response)
            
            # Убедимся что есть ссылки на карточки
            if cleaned_response != "Answer: Data not found in ChronobioticsDB\n":
                # Добавляем информацию об источниках если ее нет
                if "Sources:" not in cleaned_response:
                    card_ids = [str(card['id']) for card in relevant_cards]
                    sources = f"\nSources: Cards {', '.join(card_ids)}"
                    
                    # Добавляем ссылки если есть
                    all_links = []
                    for card in relevant_cards:
                        if card['links']:
                            all_links.extend(card['links'][:2])  # Берем первые 2 ссылки
                    
                    if all_links:
                        sources += f"\nLinks: {', '.join(all_links[:3])}"
                    
                    cleaned_response += sources
                
                if "Data Completeness:" not in cleaned_response:
                    cleaned_response += "\nData Completeness: Information from relevant cards included"
            
            # Валидация
            quality = self._validate_response(cleaned_response, relevant_cards)
            
            result = {
                "question": question,
                "response": cleaned_response,
                "cards_used": [card['id'] for card in relevant_cards],
                "card_names": [card['name'] for card in relevant_cards],
                "tokens_estimated": len(cleaned_response) // 4,
                "time": round(elapsed, 2),
                "quality_check": quality,
                "raw_response_length": len(raw_response)
            }
            
            print(f"  ✓ Cleaned: {len(cleaned_response)} chars, ~{result['tokens_estimated']} tokens")
            print(f"  Quality: {quality}")
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            print("  ⚠ CUDA OOM - using fallback...")
            return self._ask_with_fallback(question, relevant_cards)
    
    def _ask_with_fallback(self, question: str, relevant_cards: List[Dict]) -> Dict:
        """Фолбэк метод"""
        card_ids = [str(card['id']) for card in relevant_cards]
        card_names = [card['name'] for card in relevant_cards]
        
        response = f"""Answer: Information available from ChronobioticsDB cards: {', '.join(card_names)}
Sources: Cards {', '.join(card_ids)}
Data Completeness: Summary information from relevant cards"""
        
        return {
            "question": question,
            "response": response,
            "cards_used": [card['id'] for card in relevant_cards],
            "card_names": card_names,
            "tokens_estimated": len(response) // 4,
            "time": 0,
            "quality_check": {"fallback_used": True}
        }
    
    def _validate_response(self, response: str, used_cards: List[Dict]) -> Dict:
        """Улучшенная валидация ответа"""
        # Извлекаем только основную часть ответа для проверки
        main_answer = extract_main_answer(response)
        
        validation = {
            "is_in_english": True,
            "has_card_references": False,
            "has_links": False,
            "no_reasoning_tags": True,
            "mentions_limitations": False,
            "proper_format": True,
            "answer_length": len(main_answer)
        }
        
        # 1. Проверка языка (только в основной части)
        russian_chars = re.findall(r'[а-яА-ЯёЁ]', main_answer)
        if russian_chars:
            validation["is_in_english"] = False
        
        # 2. Проверка reasoning tags (во всем ответе)
        if "<think>" in response.lower():
            validation["no_reasoning_tags"] = False
        
        # 3. Проверка ссылок на карточки
        for card in used_cards:
            # Ищем ID карточки в любом формате
            if (f"Card {card['id']}" in response or 
                f"[{card['id']}]" in response or 
                f"Cards {card['id']}" in response or
                f"Card {card['id']}:" in response):
                validation["has_card_references"] = True
        
        # 4. Проверка URL
        if re.search(r'https?://', response):
            validation["has_links"] = True
        
        # 5. Проверка формата
        if not response.startswith("Answer:"):
            validation["proper_format"] = False
        
        # 6. Проверка упоминания ограничений
        limitation_phrases = [
            "not found", "no data", "no information", "not available",
            "limited", "incomplete", "further research"
        ]
        for phrase in limitation_phrases:
            if phrase in response.lower():
                validation["mentions_limitations"] = True
                break
        
        return validation

# ============================================================================
# 6. ОСНОВНОЙ ПРОЦЕСС
# ============================================================================

def main():
    """Основной процесс"""
    
    # Загружаем карточки
    print("Loading ChronobioticsDB data...")
    cards, cards_by_id, all_links = parse_chronobiotics_cards(CARDS_FILE)
    
    if not cards:
        print("✗ No cards found!")
        sys.exit(1)
    
    # ============================================
    # ШАГ 1: META-ПРОМПТИНГ
    # ============================================
    print("\n" + "="*80)
    print("INITIALIZING META-PROMPTING")
    print("="*80)
    
    meta_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    meta_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    meta_pipe = pipeline("text-generation", model=meta_model, tokenizer=meta_tokenizer)
    
    sample_card = cards[0]['full_text'] if cards else "No cards"
    system_prompt = generate_meta_prompt_step(sample_card, meta_tokenizer, meta_pipe)
    
    # Очищаем память
    del meta_model, meta_tokenizer, meta_pipe
    torch.cuda.empty_cache()
    
    # ============================================
    # ШАГ 2: EXECUTOR
    # ============================================
    print("\n" + "="*80)
    print("INITIALIZING EXECUTOR")
    print("="*80)
    
    executor = ChronobioticsExecutor(MODEL_NAME, system_prompt)
    
    # ============================================
    # ТЕСТИРОВАНИЕ
    # ============================================
    
    TEST_QUESTIONS = [
        # Основные вопросы
        "What is the mechanism of action of Resveratrol?",
        "What is the FDA status of Ramelteon?",
        "List the classes of substances to which Rosiglitazone belongs.",
        "What effects does Melatonin have on circadian rhythms?",
        "What targets are associated with Luzindole?",
        "Name at least one article related to Agomelatine.",
        "What is described about the action of 5-MCA-NAT in the database?",
        
        # Вопросы без данных (должны вернуть "Data not found")
        "What is known about the substance Xylonate?",
        "What are the effects of Chronobrainol?",
        "Does the database contain information about Neurophase?",
        
        # Комплексные вопросы
        "Compare the mechanisms of Resveratrol and Rosiglitazone.",
        "List all FDA approved substances in ChronobioticsDB.",
    ]
    
    print(f"\n" + "="*80)
    print(f"TESTING WITH {len(TEST_QUESTIONS)} QUESTIONS")
    print("="*80)
    
    results = []
    
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log_file:
        log_file.write(f"CHRONOBIOTICSDB META-PROMPTING FINAL RESULTS\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model: {MODEL_NAME}\n")
        log_file.write(f"Cards: {len(cards)}, Links: {len(all_links)}\n")
        log_file.write("="*80 + "\n\n")
        
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[{i}/{len(TEST_QUESTIONS)}] Processing...")
            
            try:
                # Поиск карточек
                relevant_cards = executor.retrieve_relevant_cards(question, cards)
                
                # Генерация ответа
                result = executor.ask_question(question, relevant_cards)
                results.append(result)
                
                # Логирование
                log_file.write(f"\n{'='*60}\nQUESTION {i}\n{'='*60}\n")
                log_file.write(f"Q: {question}\n")
                log_file.write(f"Relevant cards: {result['card_names']}\n\n")
                log_file.write(f"RESPONSE ({result['tokens_estimated']} tokens, {result['time']}s):\n")
                log_file.write(f"{result['response']}\n\n")
                log_file.write(f"QUALITY CHECK:\n")
                for key, value in result['quality_check'].items():
                    log_file.write(f"  {key}: {value}\n")
                log_file.write("\n")
                
                # Пауза
                if i < len(TEST_QUESTIONS):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                
                error_result = {
                    "question": question,
                    "response": f"ERROR: {str(e)}",
                    "cards_used": [],
                    "tokens_estimated": 0,
                    "time": 0,
                    "quality_check": {"error": str(e)}
                }
                results.append(error_result)
    
    # ============================================
    # СТАТИСТИКА
    # ============================================
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    successful = sum(1 for r in results if "ERROR" not in r["response"])
    
    if results:
        # Метрики качества
        metrics = {
            "total": len(results),
            "successful": successful,
            "avg_tokens": sum(r["tokens_estimated"] for r in results) / len(results),
            "quality": {
                "english_only": sum(1 for r in results if r.get("quality_check", {}).get("is_in_english", False)),
                "has_refs": sum(1 for r in results if r.get("quality_check", {}).get("has_card_references", False)),
                "no_reasoning": sum(1 for r in results if r.get("quality_check", {}).get("no_reasoning_tags", False)),
                "proper_format": sum(1 for r in results if r.get("quality_check", {}).get("proper_format", False)),
                "mentions_limits": sum(1 for r in results if r.get("quality_check", {}).get("mentions_limitations", False)),
            }
        }
        
        print(f"Questions processed: {metrics['total']}")
        print(f"Successful answers: {metrics['successful']}/{metrics['total']}")
        print(f"Average tokens: {metrics['avg_tokens']:.0f}")
        print(f"\nQUALITY METRICS:")
        print(f"  English only: {metrics['quality']['english_only']}/{metrics['total']}")
        print(f"  Has card references: {metrics['quality']['has_refs']}/{metrics['total']}")
        print(f"  No reasoning tags: {metrics['quality']['no_reasoning']}/{metrics['total']}")
        print(f"  Proper format: {metrics['quality']['proper_format']}/{metrics['total']}")
        print(f"  Mentions limitations: {metrics['quality']['mentions_limits']}/{metrics['total']}")
    
    print(f"\nOUTPUT FILES:")
    print(f"  Results: {OUTPUT_LOG}")
    print(f"  Generated prompt: {PROMPT_FILE}")
    
    # Сохраняем статистику
    stats = {
        "test_info": {
            "date": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "architecture": "two_step_meta_prompting_with_postprocessing"
        },
        "database": {
            "total_cards": len(cards),
            "total_links": len(all_links),
            "sample_card": cards[0]['name'] if cards else None
        },
        "parameters": {
            "meta_temperature": 0.3,
            "executor_temperature": 0.1,
            "max_tokens": 6000,
            "post_processing": "enabled"
        },
        "results": metrics if 'metrics' in locals() else {}
    }
    
    with open("final_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✓ FINAL META-PROMPTING COMPLETED SUCCESSFULLY!")
    print("="*80)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    main()
