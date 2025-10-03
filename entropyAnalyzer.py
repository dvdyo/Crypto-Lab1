#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Криптографія - Комп'ютерний практикум №1
Експериментальна оцінка ентропії на символ джерела відкритого тексту
"""

import math
import sys
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


def main():
    parser = argparse.ArgumentParser(
        description='Аналіз ентропії тексту для криптографічного практикуму'
    )
    parser.add_argument('filename', help='Шлях до текстового файлу для аналізу')
    parser.add_argument('--encoding', default='utf-8', 
                       help='Кодування файлу (за замовчуванням: utf-8)')
    parser.add_argument('--no-spaces', action='store_true',
                       help='Аналізувати текст без пробілів')
    parser.add_argument('--non-overlapping', action='store_true',
                       help='Використовувати біграми, що не перекриваються')
    
    args = parser.parse_args()
    
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
