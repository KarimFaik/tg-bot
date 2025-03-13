import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")

def postprocess_answer(answer):
    """
    Постобработка ответа: удаление лишних пробелов и символов, добавление точки в конце.
    """
    # Удаление лишних пробелов
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Добавление точки в конце, если её нет
    if not answer.endswith(('.', '!', '?')):
        answer += '.'
    return answer

def get_full_sentence(text, answer, max_length=460):
    """
    Возвращает полное предложение, содержащее ответ.
    Если в тексте есть пустые строки, обрезает ответ до первой пустой строки.
    Также ограничивает длину ответа до max_length символов.
    """
    # Находим позицию ответа в тексте
    start_pos = text.find(answer)
    if start_pos == -1:
        return answer  # Если ответ не найден, возвращаем как есть
    
    # Находим начало предложения (последний символ перед началом ответа)
    sentence_start = text.rfind('\n', 0, start_pos) + 1  # Ищем начало строки
    if sentence_start == 0:  # Если ответ в начале текста
        sentence_start = 0
    
    # Находим конец предложения (первый символ после конца ответа)
    sentence_end = text.find('\n', start_pos + len(answer))
    if sentence_end == -1:  # Если ответ в конце текста
        sentence_end = len(text)
    
    # Извлекаем полное предложение
    full_sentence = text[sentence_start:sentence_end].strip()
    
    # Обрезаем до первой пустой строки (если она есть)
    empty_line_pos = full_sentence.find('\n\n')
    if empty_line_pos != -1:
        full_sentence = full_sentence[:empty_line_pos].strip()
    
    # Ограничиваем длину ответа до max_length символов
    if len(full_sentence) > max_length:
        # Обрезаем до max_length символов, сохраняя целостность слов
        last_space_pos = full_sentence.rfind(' ', 0, max_length)
        if last_space_pos != -1:  # Если найден пробел
            full_sentence = full_sentence[:last_space_pos].strip() + "..."
        else:  # Если пробела нет, просто обрезаем
            full_sentence = full_sentence[:max_length].strip() + "..."
    
    return full_sentence

def question_answer(text, question):
    """
    Основная функция для получения ответа на вопрос.
    """
    # Токенизация вопроса и текста
    tokenized = tokenizer.encode_plus(
        question, text,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])

    # Общая длина каждого блока
    max_chunk_length = 512
    # Длина наложения
    overlapped_length = 30

    # Длина вопроса в токенах
    answer_tokens_length = tokenized.token_type_ids.count(0)
    # Токены вопроса, закодированные числами
    answer_input_ids = tokenized.input_ids[:answer_tokens_length]

    # Длина основного текста первого блока без наложения
    first_context_chunk_length = max_chunk_length - answer_tokens_length
    # Длина основного текста остальных блоков с наложением
    context_chunk_length = max_chunk_length - answer_tokens_length - overlapped_length

    # Токены основного текста
    context_input_ids = tokenized.input_ids[answer_tokens_length:]
    # Основной текст первого блока
    first = context_input_ids[:first_context_chunk_length]
    # Основной текст остальных блоков
    others = context_input_ids[first_context_chunk_length:]

    # Если есть блоки кроме первого
    if len(others) > 0:
        # Кол-во нулевых токенов, для выравнивания последнего блока по длине
        padding_length = context_chunk_length - (len(others) % context_chunk_length)
        others += [0] * padding_length

        # Кол-во блоков и их длина без добавления наложения
        new_size = (
            len(others) // context_chunk_length,
            context_chunk_length
        )

        # Упаковка блоков
        new_context_input_ids = np.reshape(others, new_size)

        # Вычисление наложения
        overlappeds = new_context_input_ids[:, -overlapped_length:]
        # Добавление в наложения частей из первого блока
        overlappeds = np.insert(overlappeds, 0, first[-overlapped_length:], axis=0)
        # Удаление наложение из последнего блока, так как оно не нужно
        overlappeds = overlappeds[:-1]

        # Добавление наложения
        new_context_input_ids = np.c_[overlappeds, new_context_input_ids]
        # Добавление первого блока
        new_context_input_ids = np.insert(new_context_input_ids, 0, first, axis=0)

        # Добавление вопроса в каждый блок
        new_input_ids = np.c_[
            [answer_input_ids] * new_context_input_ids.shape[0],
            new_context_input_ids
        ]
    # иначе обрабатывается только первый
    else:
        # Кол-во нулевых токенов, для выравнивания блока по длине
        padding_length = first_context_chunk_length - (len(first) % first_context_chunk_length)
        # Добавление нулевых токенов
        new_input_ids = np.array(
            [answer_input_ids + first + [0] * padding_length]
        )

    # Кол-во блоков
    count_chunks = new_input_ids.shape[0]

    # Маска, разделяющая вопрос и текст
    new_token_type_ids = [
        # вопрос блока
        [0] * answer_tokens_length
        # текст блока
        + [1] * (max_chunk_length - answer_tokens_length)
    ] * count_chunks

    # Маска "внимания" модели на все токены, кроме нулевых в последнем блоке
    new_attention_mask = (
        # во всех блоках, кроме последнего, "внимание" на все слова
        [[1] * max_chunk_length] * (count_chunks - 1)
        # в последнем блоке "внимание" только на ненулевые токены
        + [([1] * (max_chunk_length - padding_length)) + ([0] * padding_length)]
    )

    # Токенизированный текст в виде блоков, упакованный в torch
    new_tokenized = {
        'input_ids': torch.tensor(new_input_ids),
        'token_type_ids': torch.tensor(new_token_type_ids),
        'attention_mask': torch.tensor(new_attention_mask)
    }

    outputs = model(**new_tokenized)

    # Позиции в 2D списке токенов начала и конца наиболее вероятного ответа
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    # Пересчёт позиций начала и конца ответа для 1D списка токенов
    start_index = max_chunk_length + (
        start_index - max_chunk_length
        - (answer_tokens_length + overlapped_length)
        * (start_index // max_chunk_length)
    )
    end_index = max_chunk_length + (
        end_index - max_chunk_length
        - (answer_tokens_length + overlapped_length)
        * (end_index // max_chunk_length)
    )

    # Расширение диапазона ответа
    start_index = max(0, start_index - 2)  # Добавляем 2 токена перед началом
    end_index = min(len(tokens) - 1, end_index + 2)  # Добавляем 2 токена после конца

    # Составление ответа
    answer = ''.join([t.replace('▁', ' ') for t in tokens[start_index:end_index + 1]])
    
    answer = postprocess_answer(answer)  # Постобработка

    # Возвращаем полное предложение, содержащее ответ
    full_answer = get_full_sentence(text, answer)
    return full_answer