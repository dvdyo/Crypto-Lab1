#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Криптографія - Комп'ютерний практикум №1
Експериментальна оцінка ентропії на символ джерела відкритого тексту
"""

import math
import sys
import csv
from collections import Counter, defaultdict
import argparse


class TextEntropyAnalyzer:
    def __init__(self, text):
        """Ініціалізація аналізатора з попередньо обробленим текстом"""
        self.original_text = text
        self.processed_text = self.preprocess_text(text)
        self.processed_text_no_spaces = self.processed_text.replace(' ', '')
        
    def preprocess_text(self, text):
        """
        Попередня обробка тексту згідно з вимогами лабораторної роботи:
        - Перетворення на нижній регістр
        - Збереження лише російських літер та пробілів
        - Заміна кількох пробілів на один
        - Заміна ё на е, ъ на ь
        """
        # Перетворення на нижній регістр
        text = text.lower()
        
        # Заміна специфічних символів
        text = text.replace('ё', 'е')
        text = text.replace('ъ', 'ь')
        
        # Збереження лише російських літер та пробілів
        allowed_chars = set('абвгдежзийклмнопрстуфхцчшщьыэюя ')
        filtered_text = ''.join(char if char in allowed_chars else ' ' for char in text)
        
        # Заміна кількох пробілів на один
        while '  ' in filtered_text:
            filtered_text = filtered_text.replace('  ', ' ')
            
        return filtered_text.strip()
    
    def calculate_letter_frequencies(self, include_spaces=True):
        """Обчислення частоти кожної літери в тексті"""
        text = self.processed_text if include_spaces else self.processed_text_no_spaces
        
        if not text:
            return {}
        
        letter_counts = Counter(text)
        total_letters = len(text)
        
        frequencies = {
            letter: count / total_letters 
            for letter, count in letter_counts.items()
        }
        
        return frequencies
    
    def calculate_bigram_frequencies(self, include_spaces=True, overlapping=True):
        """
        Обчислення частоти біграм
        overlapping=True: ковзне вікно з кроком 1
        overlapping=False: неперекриваючі біграми з кроком 2
        """
        text = self.processed_text if include_spaces else self.processed_text_no_spaces
        
        if len(text) < 2:
            return {}
        
        bigram_counts = defaultdict(int)
        step = 1 if overlapping else 2
        
        for i in range(0, len(text) - 1, step):
            bigram = text[i:i+2]
            if len(bigram) == 2:
                bigram_counts[bigram] += 1
        
        total_bigrams = sum(bigram_counts.values())
        
        frequencies = {
            bigram: count / total_bigrams 
            for bigram, count in bigram_counts.items()
        }
        
        return frequencies
    
    def calculate_H1(self, include_spaces=True):
        """Обчислення ентропії H₁ на основі частот окремих літер"""
        frequencies = self.calculate_letter_frequencies(include_spaces)
        
        if not frequencies:
            return 0
        
        entropy = 0
        for freq in frequencies.values():
            if freq > 0:
                entropy -= freq * math.log2(freq)
        
        return entropy
    
    def calculate_H2(self, include_spaces=True, overlapping=True):
        """Обчислення ентропії H₂ на основі частот біграм"""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        
        if not bigram_frequencies:
            return 0
        
        entropy = 0
        for freq in bigram_frequencies.values():
            if freq > 0:
                entropy -= freq * math.log2(freq)
        
        # H₂ - це ентропія на символ, тому ділимо на 2
        return entropy / 2
    
    def export_bigram_matrix_csv(self, filename, include_spaces=True, overlapping=True):
        """Експорт матриці біграм у CSV файл з повною точністю"""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        
        # Отримання всіх унікальних символів
        if include_spaces:
            chars = sorted(set('абвгдежзийклмнопрстуфхцчшщьыэюя '))
        else:
            chars = sorted(set('абвгдежзийклмнопрстуфхцчшщьыэюя'))
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Заголовок: перший стовпець порожній, далі всі символи
                header = [''] + chars
                writer.writerow(header)
                
                # Кожен рядок: перший символ біграми, далі частоти
                for char1 in chars:
                    row = [char1]
                    for char2 in chars:
                        bigram = char1 + char2
                        freq = bigram_frequencies.get(bigram, 0)
                        row.append(freq)
                    writer.writerow(row)
            
            print(f"\nМатриця біграм успішно експортована в '{filename}'")
            print(f"Розмір матриці: {len(chars)}x{len(chars)}")
            print(f"Всього біграм: {len(bigram_frequencies)}")
            
        except IOError as e:
            print(f"Помилка при записі файлу: {e}")
            sys.exit(1)
    
    def print_letter_frequencies(self, include_spaces=True):
        """Виведення частот літер, відсортованих за спаданням"""
        frequencies = self.calculate_letter_frequencies(include_spaces)
        
        print("\n=== Частоти букв ===")
        print(f"{'Буква':<10}{'Частота':<15}{'Відсоток':<10}")
        print("-" * 35)
        
        # Сортування за частотою (за спаданням)
        sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        
        for letter, freq in sorted_freq:
            display_letter = '(пробіл)' if letter == ' ' else letter
            print(f"{display_letter:<10}{freq:<15.6f}{freq*100:<10.2f}%")
    
    def print_bigram_matrix(self, include_spaces=True, overlapping=True):
        """Виведення частот біграм у вигляді матриці"""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        
        # Отримання всіх унікальних символів
        if include_spaces:
            chars = sorted(set('абвгдежзийклмнопрстуфхцчшщьыэюя '))
        else:
            chars = sorted(set('абвгдежзийклмнопрстуфхцчшщьыэюя'))
        
        print("\n=== Матриця частот біграм ===")
        print("(Значення помножені на 1000 для зручності читання)")
        
        # Виведення заголовка
        print("   ", end="")
        for char in chars:
            display_char = '_' if char == ' ' else char
            print(f"{display_char:^3}", end="")
        print()
        
        # Виведення матриці
        for char1 in chars:
            display_char1 = '_' if char1 == ' ' else char1
            print(f"{display_char1:^3}", end="")
            for char2 in chars:
                bigram = char1 + char2
                freq = bigram_frequencies.get(bigram, 0) * 1000
                if freq > 0:
                    print(f"{freq:3.0f}", end="")
                else:
                    print("  .", end="")
            print()
    
    def print_top_bigrams(self, include_spaces=True, overlapping=True, top_n=20):
        """Виведення топ N найчастіших біграм"""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        
        print(f"\n=== Топ-{top_n} найчастіших біграм ===")
        print(f"{'Біграма':<10}{'Частота':<15}{'Відсоток':<10}")
        print("-" * 35)
        
        sorted_bigrams = sorted(bigram_frequencies.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        for bigram, freq in sorted_bigrams:
            display_bigram = bigram.replace(' ', '_')
            print(f"{display_bigram:<10}{freq:<15.6f}{freq*100:<10.2f}%")
    
    def calculate_redundancy(self, H_value, alphabet_size):
        """Обчислення надлишковості R = 1 - H∞/H₀"""
        H0 = math.log2(alphabet_size)
        R = 1 - (H_value / H0)
        return R


class BigramPredictor:
    """Клас для інтерактивного передбачення наступного символу на основі матриці біграм"""
    
    def __init__(self, csv_filename):
        """Завантаження матриці біграм з CSV файлу"""
        self.bigram_frequencies = {}
        self.alphabet = []
        self.load_from_csv(csv_filename)
    
    def load_from_csv(self, filename):
        """Завантаження матриці біграм з CSV"""
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # Читання заголовка (алфавіт)
                header = next(reader)
                self.alphabet = header[1:]  # Пропускаємо перший порожній елемент
                
                # Читання рядків матриці
                for row in reader:
                    if not row:
                        continue
                    
                    char1 = row[0]
                    for i, freq_str in enumerate(row[1:]):
                        if i >= len(self.alphabet):
                            break
                        char2 = self.alphabet[i]
                        try:
                            freq = float(freq_str)
                            if freq > 0:
                                self.bigram_frequencies[(char1, char2)] = freq
                        except ValueError:
                            continue
            
            print(f"Матриця біграм успішно завантажена з '{filename}'")
            print(f"Розмір алфавіту: {len(self.alphabet)}")
            print(f"Завантажено біграм: {len(self.bigram_frequencies)}")
            
        except FileNotFoundError:
            print(f"Помилка: Файл '{filename}' не знайдено")
            sys.exit(1)
        except Exception as e:
            print(f"Помилка при читанні CSV файлу: {e}")
            sys.exit(1)
    
    def predict_next(self, char):
        """Передбачення наступного символу після заданого"""
        char = char.lower()
        
        # Перевірка, чи символ є в алфавіті
        if char not in self.alphabet:
            return []
        
        # Знаходження всіх біграм, що починаються з цього символу
        predictions = []
        for (c1, c2), freq in self.bigram_frequencies.items():
            if c1 == char:
                predictions.append((c2, freq))
        
        # Сортування за частотою (за спаданням)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def run_interactive(self):
        """Запуск інтерактивного режиму передбачення"""
        print("\n" + "=" * 60)
        print("ІНТЕРАКТИВНИЙ РЕЖИМ ПЕРЕДБАЧЕННЯ НАСТУПНОГО СИМВОЛУ")
        print("=" * 60)
        print("\nВведіть символ, щоб побачити найбільш ймовірні наступні символи")
        print("Для пробілу введіть '_' (підкреслення)")
        print("Для виходу введіть 'exit' або 'quit'")
        print("Для виведення алфавіту введіть 'alphabet'")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nВведіть символ: ")
                
                # Не використовуємо strip() щоб зберегти пробіли
                if not user_input:
                    continue
                
                # Перевірка команд (тут можна використати strip для команд)
                if user_input.strip().lower() in ['exit', 'quit', 'вихід']:
                    print("До побачення!")
                    break
                
                if user_input.strip().lower() == 'alphabet':
                    print("\nАлфавіт матриці:")
                    display_alphabet = [c if c != ' ' else '(пробіл)' for c in self.alphabet]
                    print(', '.join(display_alphabet))
                    continue
                
                # Беремо тільки перший символ
                char = user_input[0]
                
                # Конвертуємо підкреслення в пробіл
                if char == '_':
                    char = ' '
                
                predictions = self.predict_next(char)
                
                if not predictions:
                    display_char = char if char != ' ' else '(пробіл)'
                    print(f"\nСимвол '{display_char}' не знайдено в алфавіті або немає даних про біграми")
                    continue
                
                # Виведення результатів
                display_char = char if char != ' ' else '(пробіл)'
                print(f"\n{'='*60}")
                print(f"Передбачення для символу '{display_char}':")
                print(f"{'='*60}")
                print(f"{'Ранг':<6}{'Символ':<15}{'Частота':<18}{'Відсоток':<12}")
                print("-" * 60)
                
                for rank, (next_char, freq) in enumerate(predictions, 1):
                    display_next = next_char if next_char != ' ' else '(пробіл)'
                    print(f"{rank:<6}{display_next:<15}{freq:<18.10f}{freq*100:<12.4f}%")
                
                print(f"\nВсього варіантів: {len(predictions)}")
                
            except KeyboardInterrupt:
                print("\n\nПерервано користувачем. До побачення!")
                break
            except Exception as e:
                print(f"Помилка: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Аналіз ентропії тексту для криптографічного практикуму'
    )
    
    # Основні аргументи
    parser.add_argument('filename', nargs='?', help='Шлях до текстового файлу для аналізу')
    parser.add_argument('--encoding', default='utf-8', 
                       help='Кодування файлу (за замовчуванням: utf-8)')
    parser.add_argument('--no-spaces', action='store_true',
                       help='Аналізувати текст без пробілів')
    parser.add_argument('--non-overlapping', action='store_true',
                       help='Використовувати біграми, що не перекриваються')
    
    # Нові аргументи для експорту та передбачення
    parser.add_argument('--export-csv', metavar='OUTPUT_FILE',
                       help='Експортувати матрицю біграм у CSV файл')
    parser.add_argument('--predict', metavar='CSV_FILE',
                       help='Запустити інтерактивний режим передбачення з CSV файлу')
    
    args = parser.parse_args()
    
    # Режим передбачення
    if args.predict:
        predictor = BigramPredictor(args.predict)
        predictor.run_interactive()
        return
    
    # Звичайний режим аналізу
    if not args.filename:
        parser.error("Необхідно вказати файл для аналізу або використати --predict")
    
    # Читання файлу
    try:
        with open(args.filename, 'r', encoding=args.encoding) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Помилка: Файл '{args.filename}' не знайдено")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Помилка: Не вдалося прочитати файл з кодуванням {args.encoding}")
        print("Спробуйте інше кодування, наприклад: --encoding cp1251")
        sys.exit(1)
    
    # Створення аналізатора
    analyzer = TextEntropyAnalyzer(text)
    
    print("=" * 50)
    print("АНАЛІЗ ЕНТРОПІЇ ТЕКСТУ")
    print("=" * 50)
    print(f"Файл: {args.filename}")
    print(f"Вихідний розмір тексту: {len(text)} символів")
    print(f"Оброблений текст: {len(analyzer.processed_text)} символів")
    print(f"Текст без пробілів: {len(analyzer.processed_text_no_spaces)} символів")
    
    # Експорт CSV якщо запитано
    if args.export_csv:
        analyzer.export_bigram_matrix_csv(
            args.export_csv,
            include_spaces=not args.no_spaces,
            overlapping=not args.non_overlapping
        )
        print("\nЕкспорт завершено. Завершення роботи.")
        return
    
    # Аналіз з пробілами
    print("\n" + "=" * 50)
    print("АНАЛІЗ З ПРОБІЛАМИ (32 символи в алфавіті)")
    print("=" * 50)
    
    analyzer.print_letter_frequencies(include_spaces=True)
    
    H1_with_spaces = analyzer.calculate_H1(include_spaces=True)
    print(f"\nH₁ = {H1_with_spaces:.4f} біт/символ")
    
    analyzer.print_top_bigrams(include_spaces=True, overlapping=True)
    
    H2_with_spaces = analyzer.calculate_H2(include_spaces=True, overlapping=True)
    print(f"\nH₂ = {H2_with_spaces:.4f} біт/символ")
    
    # Обчислення надлишковості для різних моделей
    R1_with_spaces = analyzer.calculate_redundancy(H1_with_spaces, 32)
    R2_with_spaces = analyzer.calculate_redundancy(H2_with_spaces, 32)
    
    print(f"\nНадлишковість R₁ = {R1_with_spaces:.4f} ({R1_with_spaces*100:.2f}%)")
    print(f"Надлишковість R₂ = {R2_with_spaces:.4f} ({R2_with_spaces*100:.2f}%)")
    
    # Аналіз без пробілів
    print("\n" + "=" * 50)
    print("АНАЛІЗ БЕЗ ПРОБІЛІВ (31 символ в алфавіті)")
    print("=" * 50)
    
    analyzer.print_letter_frequencies(include_spaces=False)
    
    H1_no_spaces = analyzer.calculate_H1(include_spaces=False)
    print(f"\nH₁ = {H1_no_spaces:.4f} біт/символ")
    
    analyzer.print_top_bigrams(include_spaces=False, overlapping=True)
    
    H2_no_spaces = analyzer.calculate_H2(include_spaces=False, overlapping=True)
    print(f"\nH₂ = {H2_no_spaces:.4f} біт/символ")
    
    # Обчислення надлишковості для різних моделей
    R1_no_spaces = analyzer.calculate_redundancy(H1_no_spaces, 31)
    R2_no_spaces = analyzer.calculate_redundancy(H2_no_spaces, 31)
    
    print(f"\nНадлишковість R₁ = {R1_no_spaces:.4f} ({R1_no_spaces*100:.2f}%)")
    print(f"Надлишковість R₂ = {R2_no_spaces:.4f} ({R2_no_spaces*100:.2f}%)")
    
    # Підсумки
    print("\n" + "=" * 50)
    print("ПІДСУМКОВА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 50)
    
    print(f"{'Модель':<25}{'З пробілами':<15}{'Без пробілів':<15}")
    print("-" * 55)
    print(f"{'H₁ (біт/символ)':<25}{H1_with_spaces:<15.4f}{H1_no_spaces:<15.4f}")
    print(f"{'H₂ (біт/символ)':<25}{H2_with_spaces:<15.4f}{H2_no_spaces:<15.4f}")
    print(f"{'Надлишковість R₁ (%)':<25}{R1_with_spaces*100:<15.2f}{R1_no_spaces*100:<15.2f}")
    print(f"{'Надлишковість R₂ (%)':<25}{R2_with_spaces*100:<15.2f}{R2_no_spaces*100:<15.2f}")
    
    # Опціонально: виведення матриці біграм (може бути дуже великою)
    print("\nБажаєте вивести повну матрицю біграм? (y/n): ", end="")
    if input().lower() == 'y':
        analyzer.print_bigram_matrix(include_spaces=True, overlapping=True)


if __name__ == "__main__":
    main()
