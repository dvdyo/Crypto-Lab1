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
import re

# =============================================================================
# CORE LOGIC CLASSES (UNCHANGED)
# =============================================================================

class TextEntropyAnalyzer:
    def __init__(self, text):
        # Ініціалізація аналізатора з попередньо обробленим текстом
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
        text = text.lower()
        
        text = text.replace('ё', 'е')
        text = text.replace('ъ', 'ь')
        
        allowed_chars = set('абвгдежзийклмнопрстуфхцчшщьыэюя ')
        filtered_text = ''.join(char if char in allowed_chars else ' ' for char in text)
        
        # Заміна кількох пробілів на один
        filtered_text = re.sub(r' +', ' ', filtered_text)

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
                
                header = [''] + chars
                writer.writerow(header)
                
                for char1 in chars:
                    row = [char1]
                    for char2 in chars:
                        bigram = char1 + char2
                        freq = bigram_frequencies.get(bigram, 0)
                        row.append(freq)
                    writer.writerow(row)
            
            print(f"Матриця біграм успішно експортована в '{filename}'")
            
        except IOError as e:
            print(f"Помилка при записі файлу: {e}", file=sys.stderr)
            sys.exit(1)
    
    def print_letter_frequencies(self, include_spaces=True):
        """Виведення частот літер, відсортованих за спаданням"""
        frequencies = self.calculate_letter_frequencies(include_spaces)
        
        print("\n=== Частоти букв ===")
        print(f"{'Буква':<10}{'Частота':<15}{'Відсоток':<10}")
        print("-" * 35)
        
        sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        
        for letter, freq in sorted_freq:
            display_letter = '(пробіл)' if letter == ' ' else letter
            print(f"{display_letter:<10}{freq:<15.6f}{freq*100:<10.2f}%")
    
    def print_top_bigrams(self, include_spaces=True, overlapping=True, top_n=20):
        """Виведення топ N найчастіших біграм"""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        
        print(f"\n=== Топ-{top_n} найчастіших біграм ===")
        print(f"Тип: {'перекриваючі' if overlapping else 'неперекриваючі'}")
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
        if H0 == 0: return 0
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
                
                header = next(reader)
                self.alphabet = header[1:]
                
                for row in reader:
                    if not row: continue
                    char1 = row[0]
                    for i, freq_str in enumerate(row[1:]):
                        if i >= len(self.alphabet): break
                        char2 = self.alphabet[i]
                        try:
                            freq = float(freq_str)
                            if freq > 0: self.bigram_frequencies[(char1, char2)] = freq
                        except ValueError: continue
            
            print(f"Матриця біграм успішно завантажена з '{filename}'")
            
        except FileNotFoundError:
            print(f"Помилка: Файл '{filename}' не знайдено", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Помилка при читанні CSV файлу: {e}", file=sys.stderr)
            sys.exit(1)
    
    def predict_next(self, char):
        """Передбачення наступного символу після заданого"""
        char = char.lower()
        if char not in self.alphabet: return []
        
        predictions = []
        for (c1, c2), freq in self.bigram_frequencies.items():
            if c1 == char:
                predictions.append((c2, freq))
        
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
        
        while True:
            try:
                user_input = input("\nВведіть символ: ")
                if not user_input: continue
                
                if user_input.strip().lower() in ['exit', 'quit', 'вихід']:
                    print("До побачення!")
                    break
                
                char = user_input[0]
                if char == '_': char = ' '
                
                predictions = self.predict_next(char)
                
                display_char = char if char != ' ' else '(пробіл)'
                if not predictions:
                    print(f"\nСимвол '{display_char}' не знайдено в алфавіті або немає даних про біграми")
                    continue
                
                print(f"\nПередбачення для символу '{display_char}':")
                print(f"{'Ранг':<6}{'Символ':<15}{'Частота':<18}{'Відсоток':<12}")
                print("-" * 60)
                
                for rank, (next_char, freq) in enumerate(predictions, 1):
                    display_next = next_char if next_char != ' ' else '(пробіл)'
                    print(f"{rank:<6}{display_next:<15}{freq:<18.10f}{freq*100:<12.4f}%")
                
            except KeyboardInterrupt:
                print("\n\nПерервано користувачем. До побачення!")
                break
            except Exception as e:
                print(f"Помилка: {e}")

# =============================================================================
# CLI HANDLER FUNCTIONS
# =============================================================================

def handle_analyze(args):
    """Handler for the 'analyze' command."""
    try:
        with open(args.filename, 'r', encoding=args.encoding) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Помилка: Файл '{args.filename}' не знайдено", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Помилка: Не вдалося прочитати файл з кодуванням {args.encoding}", file=sys.stderr)
        sys.exit(1)

    analyzer = TextEntropyAnalyzer(text)
    overlapping = args.bigrams == 'overlapping'

    # Determine which analysis modes to run
    modes_to_run = []
    if args.spaces in ['include', 'both']:
        modes_to_run.append(True)  # True means include_spaces
    if args.spaces in ['exclude', 'both']:
        modes_to_run.append(False)

    results = {}
    for include_spaces in modes_to_run:
        mode_key = "with_spaces" if include_spaces else "without_spaces"
        alphabet_size = 32 if include_spaces else 31

        H1 = analyzer.calculate_H1(include_spaces=include_spaces)
        H2 = analyzer.calculate_H2(include_spaces=include_spaces, overlapping=overlapping)
        R1 = analyzer.calculate_redundancy(H1, alphabet_size)
        R2 = analyzer.calculate_redundancy(H2, alphabet_size)
        
        results[mode_key] = {'H1': H1, 'H2': H2, 'R1': R1, 'R2': R2}

        if args.verbose:
            print("\n" + "─" * 50)
            mode_name = "З ПРОБІЛАМИ" if include_spaces else "БЕЗ ПРОБІЛІВ"
            print(f"ДЕТАЛЬНИЙ АНАЛІЗ ({mode_name})")
            print("─" * 50)
            analyzer.print_letter_frequencies(include_spaces=include_spaces)
            analyzer.print_top_bigrams(include_spaces=include_spaces, overlapping=overlapping)
    
    # Print final summary table
    print("\n" + "=" * 60)
    print("                ПІДСУМКОВІ РЕЗУЛЬТАТИ АНАЛІЗУ")
    print("=" * 60)
    print(f"Файл: {args.filename}")
    print(f"Тип біграм: {'перекриваючі' if overlapping else 'неперекриваючі'}")
    print("-" * 60)
    
    header = f"{'Метрика':<25}"
    if 'with_spaces' in results: header += f"{'З пробілами':<17}"
    if 'without_spaces' in results: header += f"{'Без пробілів':<17}"
    print(header)
    print("-" * 60)

    for metric_key, metric_name in [('H1', 'H₁ (біт/символ)'), ('H2', 'H₂ (біт/символ)'),
                                     ('R1', 'Надлишковість R₁ (%)'), ('R2', 'Надлишковість R₂ (%)')]:
        line = f"{metric_name:<25}"
        multiplier = 100 if '%' in metric_name else 1
        fmt = ".2f" if '%' in metric_name else ".4f"

        if 'with_spaces' in results:
            val = results['with_spaces'][metric_key] * multiplier
            line += f"{val:<17{fmt}}"
        if 'without_spaces' in results:
            val = results['without_spaces'][metric_key] * multiplier
            line += f"{val:<17{fmt}}"
        print(line)
    print("=" * 60)


def handle_export(args):
    """Handler for the 'export' command."""
    try:
        with open(args.in_filename, 'r', encoding=args.encoding) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Помилка: Файл '{args.in_filename}' не знайдено", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Помилка: Не вдалося прочитати файл з кодуванням {args.encoding}", file=sys.stderr)
        sys.exit(1)
    
    analyzer = TextEntropyAnalyzer(text)
    
    include_spaces = args.spaces == 'include'
    overlapping = args.bigrams == 'overlapping'
    
    print(f"Експорт матриці біграм з файлу '{args.in_filename}'...")
    print(f"  - Режим пробілів: {'з пробілами' if include_spaces else 'без пробілів'}")
    print(f"  - Тип біграм: {'перекриваючі' if overlapping else 'неперекриваючі'}")

    analyzer.export_bigram_matrix_csv(
        args.output,
        include_spaces=include_spaces,
        overlapping=overlapping
    )


def handle_predict(args):
    """Handler for the 'predict' command."""
    predictor = BigramPredictor(args.matrix_filename)
    predictor.run_interactive()

# =============================================================================
# MAIN FUNCTION AND ARGUMENT PARSING
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Експериментальна оцінка ентропії тексту.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Доступні команди')

    # --- ANALYZE command ---
    parser_analyze = subparsers.add_parser('analyze', help='Аналіз тексту для обчислення ентропії')
    parser_analyze.add_argument('filename', help='Шлях до текстового файлу для аналізу')
    parser_analyze.add_argument(
        '--spaces', choices=['include', 'exclude', 'both'], default='include',
        help='Як обробляти пробіли:\n'
             '  include: аналізувати текст з пробілами (за замовчуванням)\n'
             '  exclude: аналізувати текст без пробілів\n'
             '  both:    показати результати для обох випадків'
    )
    parser_analyze.add_argument(
        '--bigrams', choices=['overlapping', 'non-overlapping'], default='overlapping',
        help='Тип біграм для використання (за замовчуванням: overlapping)'
    )
    parser_analyze.add_argument('-v', '--verbose', action='store_true', help='Вивести детальні таблиці частот')
    parser_analyze.add_argument('--encoding', default='utf-8', help='Кодування вхідного файлу (за замовчуванням: utf-8)')
    parser_analyze.set_defaults(func=handle_analyze)

    # --- EXPORT command ---
    parser_export = subparsers.add_parser('export', help='Експорт матриці частот біграм у CSV файл')
    parser_export.add_argument('in_filename', help='Шлях до вхідного текстового файлу')
    parser_export.add_argument('-o', '--output', required=True, help='Шлях до вихідного CSV файлу')
    parser_export.add_argument(
        '--spaces', choices=['include', 'exclude'], default='include',
        help='Включати або виключати пробіли при побудові матриці (за замовчуванням: include)'
    )
    parser_export.add_argument(
        '--bigrams', choices=['overlapping', 'non-overlapping'], default='overlapping',
        help='Тип біграм для використання (за замовчуванням: overlapping)'
    )
    parser_export.add_argument('--encoding', default='utf-8', help='Кодування вхідного файлу (за замовчуванням: utf-8)')
    parser_export.set_defaults(func=handle_export)

    # --- PREDICT command ---
    parser_predict = subparsers.add_parser('predict', help='Запуск інтерактивного режиму передбачення з матриці')
    parser_predict.add_argument('matrix_filename', help='Шлях до CSV файлу з матрицею біграм')
    parser_predict.set_defaults(func=handle_predict)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
