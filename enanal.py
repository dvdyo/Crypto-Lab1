#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Криптографія - Комп'ютерний практикум №1
Експериментальна оцінка ентропії на символ джерела відкритого тексту.
"""

import math
import sys
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import argparse

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

CYRILLIC_ALPHABET = 'абвгдежзийклмнопрстуфхцчшщьыэюя'
CYRILLIC_ALPHABET_WITH_SPACE = CYRILLIC_ALPHABET + ' '

# Normalization table for str.translate()
CHAR_NORMALIZATION = str.maketrans({
    'ё': 'е',
    'ъ': 'ь'
})


class SpaceMode(Enum):
    """Enumeration for space handling modes."""
    WITH_SPACES = "with_spaces"
    WITHOUT_SPACES = "without_spaces"


@dataclass
class AnalysisConfig:
    """Configuration for text analysis."""
    include_spaces: bool = True
    overlapping: bool = True
    top_n: int = 20
    verbose: bool = False
    show_matrix: bool = False
    show_stats: bool = False


@dataclass
class TextStats:
    """Statistics about text processing."""
    original_length: int
    processed_length: int
    processed_no_spaces_length: Optional[int] = None
    
    def print_stats(self) -> None:
        """Print text processing statistics."""
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ОБРОБКИ ТЕКСТУ")
        print("=" * 60)
        print(f"Вхідний текст (символів):           {self.original_length:>12,}")
        print(f"Оброблений текст (символів):        {self.processed_length:>12,}")
        if self.processed_no_spaces_length is not None:
            print(f"Оброблений текст без пробілів:      {self.processed_no_spaces_length:>12,}")
        print(f"Видалено символів:                  {self.original_length - self.processed_length:>12,}")
        print("=" * 60)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_char_display(char: str) -> str:
    """Format a character for display, replacing space with readable text."""
    return '(пробіл)' if char == ' ' else char


def format_bigram_display(bigram: str) -> str:
    """Format a bigram for display, replacing spaces with underscores."""
    return bigram.replace(' ', '_')


def get_mode_name(include_spaces: bool) -> str:
    """Get a human-readable name for the analysis mode."""
    return "З ПРОБІЛАМИ" if include_spaces else "БЕЗ ПРОБІЛІВ"


def get_alphabet_size(include_spaces: bool) -> int:
    """Get the alphabet size based on whether spaces are included."""
    return 32 if include_spaces else 31


def get_alphabet_chars(include_spaces: bool) -> str:
    """Get the alphabet characters based on mode."""
    return CYRILLIC_ALPHABET_WITH_SPACE if include_spaces else CYRILLIC_ALPHABET


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """Handles text preprocessing operations."""
    
    # Pre-compile regex patterns - these are MUCH faster than Python loops
    _NON_CYRILLIC_PATTERN = re.compile(f'[^{CYRILLIC_ALPHABET} ]+')
    _MULTI_SPACE_PATTERN = re.compile(r' +')
    
    @staticmethod
    def preprocess(text: str) -> str:
        """
        Preprocesses text according to requirements:
        - Converts to lowercase
        - Normalizes specific letters (ё -> е, ъ -> ь)
        - Keeps only Russian letters and spaces
        - Replaces multiple spaces with single space
        
        Regex-based approach: Let C code (in re module) do the heavy lifting.
        """
        # Step 1: Lowercase and normalize (fast C operation)
        text = text.lower().translate(CHAR_NORMALIZATION)
        
        # Step 2: Replace all non-Cyrillic characters with space (regex is fast for this)
        text = TextPreprocessor._NON_CYRILLIC_PATTERN.sub(' ', text)
        
        # Step 3: Collapse multiple spaces into one (another fast regex operation)
        text = TextPreprocessor._MULTI_SPACE_PATTERN.sub(' ', text)
        
        # Step 4: Strip leading/trailing spaces
        return text.strip()


# =============================================================================
# CORE ANALYSIS CLASSES
# =============================================================================

class TextEntropyAnalyzer:
    """Analyzes text for statistical and entropy properties."""

    def __init__(self, text: str):
        self.original_text = text
        self.original_length = len(text)
        self.processed_text = TextPreprocessor.preprocess(text)
        self.processed_length = len(self.processed_text)
        self._processed_no_spaces_length = None  # Lazy calculation
        self._validate_text()

    def _validate_text(self) -> None:
        """Validates that the processed text is not empty."""
        if not self.processed_text:
            raise ValueError("Оброблений текст порожній. Перевірте вхідний файл.")

    def _get_text(self, include_spaces: bool) -> str:
        """Get text with or without spaces."""
        return self.processed_text if include_spaces else self.processed_text.replace(' ', '')
    
    def _get_no_spaces_length(self) -> int:
        """Lazy calculation of no-spaces length."""
        if self._processed_no_spaces_length is None:
            self._processed_no_spaces_length = len(self.processed_text.replace(' ', ''))
        return self._processed_no_spaces_length
    
    def get_stats(self, include_no_spaces: bool = False) -> TextStats:
        """Get text processing statistics."""
        return TextStats(
            original_length=self.original_length,
            processed_length=self.processed_length,
            processed_no_spaces_length=self._get_no_spaces_length() if include_no_spaces else None
        )

    @lru_cache(maxsize=4)
    def calculate_letter_frequencies(self, include_spaces: bool = True) -> Dict[str, float]:
        """Calculates the frequency of each letter in the text."""
        text = self._get_text(include_spaces)
        if not text:
            return {}

        total_letters = len(text)
        letter_counts = Counter(text)
        
        return {letter: count / total_letters for letter, count in letter_counts.items()}

    @lru_cache(maxsize=8)
    def calculate_bigram_frequencies(
        self, 
        include_spaces: bool = True, 
        overlapping: bool = True
    ) -> Dict[str, float]:
        """Calculates the frequency of bigrams - optimized version."""
        text = self._get_text(include_spaces)
        if len(text) < 2:
            return {}

        if overlapping:
            # Direct slicing in generator
            bigram_counts = Counter(text[i:i+2] for i in range(len(text) - 1))
            total_bigrams = len(text) - 1
        else:
            # Non-overlapping: step by 2
            bigram_counts = Counter(text[i:i+2] for i in range(0, len(text) - 1, 2) if i+2 <= len(text))
            total_bigrams = len(text) // 2
        
        return {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}
    
    def calculate_h1(self, include_spaces: bool = True) -> float:
        """Calculates entropy H₁ based on single letter frequencies."""
        frequencies = self.calculate_letter_frequencies(include_spaces)
        if not frequencies:
            return 0.0
        
        return -sum(
            freq * math.log2(freq) for freq in frequencies.values() if freq > 0
        )

    def calculate_h2(self, include_spaces: bool = True, overlapping: bool = True) -> float:
        """Calculates entropy H₂ based on bigram frequencies."""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        if not bigram_frequencies:
            return 0.0

        entropy = -sum(
            freq * math.log2(freq) for freq in bigram_frequencies.values() if freq > 0
        )
        return entropy / 2

    def calculate_redundancy(self, h_value: float, alphabet_size: int) -> float:
        """Calculates redundancy R = 1 - H/H₀."""
        if alphabet_size <= 1:
            return 0.0
        h0 = math.log2(alphabet_size)
        return 1 - (h_value / h0)

    def get_sorted_frequencies(
        self, 
        include_spaces: bool = True
    ) -> List[Tuple[str, float]]:
        """Get letter frequencies sorted by frequency (descending)."""
        frequencies = self.calculate_letter_frequencies(include_spaces)
        return sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    def get_sorted_bigrams(
        self, 
        include_spaces: bool = True, 
        overlapping: bool = True
    ) -> List[Tuple[str, float]]:
        """Get bigram frequencies sorted by frequency (descending)."""
        bigram_frequencies = self.calculate_bigram_frequencies(include_spaces, overlapping)
        return sorted(bigram_frequencies.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# REPORTING AND EXPORT
# =============================================================================

class TextEntropyReporter:
    """Handles formatting and printing of analysis results."""

    def __init__(self, analyzer: TextEntropyAnalyzer):
        self.analyzer = analyzer

    def print_letter_frequencies(
        self, 
        include_spaces: bool = True, 
        top_n: int = 20
    ) -> None:
        """Prints the top N letter frequencies."""
        sorted_freq = self.analyzer.get_sorted_frequencies(include_spaces)
        header_text = f"Топ-{top_n} найчастіших букв" if top_n != -1 else "Частоти всіх букв"

        print(f"\n=== {header_text} ===")
        print(f"{'Буква':<10}{'Частота':<15}{'Відсоток':<10}")
        print("-" * 35)

        items_to_print = sorted_freq if top_n == -1 else sorted_freq[:top_n]
        for letter, freq in items_to_print:
            display_letter = format_char_display(letter)
            print(f"{display_letter:<10}{freq:<15.6f}{freq*100:<10.2f}%")

    def print_top_bigrams(
        self, 
        include_spaces: bool = True, 
        overlapping: bool = True, 
        top_n: int = 20
    ) -> None:
        """Prints the top N most frequent bigrams."""
        sorted_bigrams = self.analyzer.get_sorted_bigrams(include_spaces, overlapping)
        header_text = f"Топ-{top_n} найчастіших біграм" if top_n != -1 else "Всі біграми"

        print(f"\n=== {header_text} ===")
        print(f"Тип: {'перекриваючі' if overlapping else 'неперекриваючі'}")
        print(f"{'Біграма':<10}{'Частота':<15}{'Відсоток':<10}")
        print("-" * 35)

        items_to_print = sorted_bigrams if top_n == -1 else sorted_bigrams[:top_n]
        for bigram, freq in items_to_print:
            display_bigram = format_bigram_display(bigram)
            print(f"{display_bigram:<10}{freq:<15.6f}{freq*100:<10.2f}%")

    def print_bigram_matrix(
        self, 
        include_spaces: bool = True, 
        overlapping: bool = True
    ) -> None:
        """Prints a condensed view of the bigram frequency matrix."""
        bigram_frequencies = self.analyzer.calculate_bigram_frequencies(
            include_spaces, overlapping
        )
        chars = sorted(list(get_alphabet_chars(include_spaces)))

        mode_name = get_mode_name(include_spaces)
        print(f"\n=== Матриця частот біграм ({mode_name}) ===")
        print(f"Тип: {'перекриваючі' if overlapping else 'неперекриваючі'}")
        print(
            "(Значення помножені на 1000 для зручності читання.\n"
            "Справжні значення можна переглянути у CSV файлі після експорту)"
        )

        # Print header
        print(f"{'':>4}", end="")
        for char in chars:
            print(f"{format_char_display(char)[0]:^3}", end="")
        print()

        # Print matrix rows
        for char1 in chars:
            print(f"{format_char_display(char1)[0]:^3}", end="")
            for char2 in chars:
                freq = bigram_frequencies.get(char1 + char2, 0) * 1000
                print(f"{freq:3.0f}" if freq > 0 else "  .", end="")
            print()

    def print_summary_results(
        self, 
        results: Dict[str, Dict[str, float]], 
        filename: str,
        overlapping: bool
    ) -> None:
        """Prints the summary results table."""
        print("\n" + "=" * 60)
        print("                ПІДСУМКОВІ РЕЗУЛЬТАТИ АНАЛІЗУ")
        print("=" * 60)
        print(f"Файл: {filename}")
        print(f"Тип біграм: {'перекриваючі' if overlapping else 'неперекриваючі'}")
        print("-" * 60)

        # Build header
        header = f"{'Метрика':<25}"
        if SpaceMode.WITH_SPACES.value in results:
            header += f"{'З пробілами':<17}"
        if SpaceMode.WITHOUT_SPACES.value in results:
            header += f"{'Без пробілів':<17}"
        print(header)
        print("-" * 60)

        # Print metrics
        metrics_to_print = [
            ('H1', 'H₁ (біт/символ)'),
            ('H2', 'H₂ (біт/символ)'),
            ('R1', 'Надлишковість R₁ (%)'),
            ('R2', 'Надлишковість R₂ (%)')
        ]
        for metric_key, metric_name in metrics_to_print:
            line = f"{metric_name:<25}"
            multiplier = 100 if '%' in metric_name else 1
            fmt = ".2f" if '%' in metric_name else ".4f"

            for mode in [SpaceMode.WITH_SPACES.value, SpaceMode.WITHOUT_SPACES.value]:
                if mode in results:
                    value = results[mode][metric_key] * multiplier
                    line += f"{value:<17{fmt}}"
            print(line)

        print("=" * 60)


class CSVExporter:
    """Handles export of analysis results to CSV format."""

    @staticmethod
    def export_bigram_matrix(
        analyzer: TextEntropyAnalyzer,
        filename: str,
        include_spaces: bool = True,
        overlapping: bool = True
    ) -> None:
        """Exports the bigram frequency matrix to a CSV file."""
        bigram_frequencies = analyzer.calculate_bigram_frequencies(
            include_spaces, overlapping
        )
        chars = sorted(list(get_alphabet_chars(include_spaces)))

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([''] + chars)
                for char1 in chars:
                    row = [char1] + [
                        bigram_frequencies.get(char1 + char2, 0) for char2 in chars
                    ]
                    writer.writerow(row)
            print(f"Матриця біграм успішно експортована в '{filename}'")
        except (IOError, PermissionError) as e:
            print(f"Помилка при записі файлу: {e}", file=sys.stderr)
            sys.exit(1)


# =============================================================================
# PREDICTION
# =============================================================================

class BigramPredictor:
    """Handles interactive prediction based on a loaded bigram matrix."""

    def __init__(self, csv_filename: str):
        self.bigram_frequencies: Dict[Tuple[str, str], float] = {}
        self.alphabet: List[str] = []
        self.load_from_csv(csv_filename)

    def load_from_csv(self, filename: str) -> None:
        """Loads a bigram matrix from a CSV file."""
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                self.alphabet = header[1:]
                
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
        except FileNotFoundError:
            print(f"Помилка: Файл '{filename}' не знайдено", file=sys.stderr)
            sys.exit(1)
        except (IOError, PermissionError) as e:
            print(f"Помилка доступу до файлу: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Помилка при читанні CSV файлу: {e}", file=sys.stderr)
            sys.exit(1)

    def predict_next(self, char: str) -> List[Tuple[str, float]]:
        """Predicts the most likely next characters."""
        char = char.lower()
        if char not in self.alphabet:
            return []

        predictions = [
            (c2, freq) for (c1, c2), freq in self.bigram_frequencies.items()
            if c1 == char
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

    def run_interactive(self) -> None:
        """Runs the interactive prediction loop."""
        self._print_welcome_message()

        while True:
            try:
                user_input = input("\nВведіть символ: ").strip()
                
                if not user_input or user_input.lower() in ['exit', 'quit', 'вихід']:
                    print("До побачення!")
                    break

                char = user_input[0]
                lookup_char = ' ' if char == '_' else char
                display_char = format_char_display(lookup_char)

                predictions = self.predict_next(lookup_char)
                if not predictions:
                    print(f"\nСимвол '{display_char}' не знайдено в алфавіті або немає даних про біграми")
                    continue

                self._print_predictions(display_char, predictions)
                
            except KeyboardInterrupt:
                print("\n\nПерервано користувачем. До побачення!")
                break
            except Exception as e:
                print(f"Помилка: {e}")

    def _print_welcome_message(self) -> None:
        """Prints the welcome message for interactive mode."""
        print("\n" + "=" * 60)
        print("ІНТЕРАКТИВНИЙ РЕЖИМ ПЕРЕДБАЧЕННЯ НАСТУПНОГО СИМВОЛУ")
        print("=" * 60)
        print("\nВведіть символ, щоб побачити найбільш ймовірні наступні символи")
        print("Для пробілу введіть '_' (підкреслення)")
        print("Для виходу введіть 'exit' або 'quit'")

    def _print_predictions(self, display_char: str, predictions: List[Tuple[str, float]]) -> None:
        """Prints the prediction results."""
        print(f"\nПередбачення для символу '{display_char}':")
        print(f"{'Ранг':<6}{'Символ':<15}{'Частота':<18}{'Відсоток':<12}")
        print("-" * 60)

        for rank, (next_char, freq) in enumerate(predictions, 1):
            display_next = format_char_display(next_char)
            print(f"{rank:<6}{display_next:<15}{freq:<18.10f}{freq*100:<12.4f}%")


# =============================================================================
# FILE OPERATIONS
# =============================================================================

class FileReader:
    """Handles reading text files with error handling."""

    @staticmethod
    def read_text_file(filename: str, encoding: str = 'utf-8') -> str:
        """Reads a text file with proper error handling."""
        try:
            with open(filename, 'r', encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            print(f"Помилка: Файл '{filename}' не знайдено", file=sys.stderr)
            sys.exit(1)
        except (PermissionError, IOError) as e:
            print(f"Помилка доступу до файлу: {e}", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError:
            print(
                f"Помилка: Не вдалося прочитати файл з кодуванням {encoding}. "
                "Спробуйте інше кодування.",
                file=sys.stderr
            )
            sys.exit(1)


# =============================================================================
# CLI HANDLER FUNCTIONS
# =============================================================================

def handle_analyze(args: argparse.Namespace) -> None:
    """Handler for the 'analyze' command."""
    text = FileReader.read_text_file(args.filename, args.encoding)
    
    try:
        analyzer = TextEntropyAnalyzer(text)
    except ValueError as e:
        print(f"Помилка: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Show stats if requested or verbose
    # Include no-spaces count only if we're analyzing without spaces or both
    if args.show_stats or args.verbose:
        include_no_spaces = args.spaces in ['exclude', 'both']
        analyzer.get_stats(include_no_spaces=include_no_spaces).print_stats()
    
    reporter = TextEntropyReporter(analyzer)
    overlapping = args.bigrams == 'overlapping'

    # Determine which modes to run
    if args.spaces == 'include':
        modes_to_run = [True]
    elif args.spaces == 'exclude':
        modes_to_run = [False]
    else:  # 'both'
        modes_to_run = [True, False]

    # Calculate results for each mode
    results = {}
    for include_spaces in modes_to_run:
        mode_key = SpaceMode.WITH_SPACES.value if include_spaces else SpaceMode.WITHOUT_SPACES.value
        alphabet_size = get_alphabet_size(include_spaces)

        h1 = analyzer.calculate_h1(include_spaces=include_spaces)
        h2 = analyzer.calculate_h2(include_spaces=include_spaces, overlapping=overlapping)
        r1 = analyzer.calculate_redundancy(h1, alphabet_size)
        r2 = analyzer.calculate_redundancy(h2, alphabet_size)

        results[mode_key] = {'H1': h1, 'H2': h2, 'R1': r1, 'R2': r2}

        # Print detailed analysis if verbose
        if args.verbose:
            print("\n" + "─" * 50)
            print(f"ДЕТАЛЬНИЙ АНАЛІЗ ({get_mode_name(include_spaces)})")
            print("─" * 50)
            reporter.print_letter_frequencies(include_spaces=include_spaces, top_n=args.top)
            reporter.print_top_bigrams(
                include_spaces=include_spaces,
                overlapping=overlapping,
                top_n=args.top
            )

    # Print summary
    reporter.print_summary_results(results, args.filename, overlapping)

    # Print bigram matrix if requested
    if args.show_matrix or args.verbose:
        for include_spaces in modes_to_run:
            reporter.print_bigram_matrix(include_spaces=include_spaces, overlapping=overlapping)


def handle_export(args: argparse.Namespace) -> None:
    """Handler for the 'export' command."""
    text = FileReader.read_text_file(args.in_filename, args.encoding)
    
    try:
        analyzer = TextEntropyAnalyzer(text)
    except ValueError as e:
        print(f"Помилка: {e}", file=sys.stderr)
        sys.exit(1)
    
    include_spaces = args.spaces == 'include'
    overlapping = args.bigrams == 'overlapping'

    print(f"Експорт матриці біграм з файлу '{args.in_filename}'...")
    print(f"  - Режим пробілів: {'з пробілами' if include_spaces else 'без пробілів'}")
    print(f"  - Тип біграм: {'перекриваючі' if overlapping else 'неперекриваючі'}")

    CSVExporter.export_bigram_matrix(
        analyzer, args.output, include_spaces=include_spaces, overlapping=overlapping
    )


def handle_predict(args: argparse.Namespace) -> None:
    """Handler for the 'predict' command."""
    predictor = BigramPredictor(args.matrix_filename)
    predictor.run_interactive()


# =============================================================================
# MAIN FUNCTION AND ARGUMENT PARSING
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser."""
    parser = argparse.ArgumentParser(
        description='Експериментальна оцінка ентропії тексту.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest='command', required=True, help='Доступні команди'
    )

    # --- ANALYZE command ---
    parser_analyze = subparsers.add_parser(
        'analyze', help='Аналіз тексту для обчислення ентропії'
    )
    parser_analyze.add_argument(
        'filename', help='Шлях до текстового файлу для аналізу'
    )
    parser_analyze.add_argument(
        '--spaces', choices=['include', 'exclude', 'both'], default='include',
        help='Як обробляти пробіли:\n'
             '  include: аналізувати текст з пробілами (за замовчуванням)\n'
             '  exclude: аналізувати текст без пробілів\n'
             '  both:    показати результати для обох випадків'
    )
    parser_analyze.add_argument(
        '--bigrams', choices=['overlapping', 'non-overlapping'],
        default='overlapping',
        help='Тип біграм для використання (за замовчуванням: overlapping)'
    )
    parser_analyze.add_argument(
        '-n', '--top', type=int, default=20, metavar='N',
        help='Показати топ N найчастіших елементів у детальному звіті.\n'
             'За замовчуванням: 20. Вкажіть -1, щоб показати всі.'
    )
    parser_analyze.add_argument(
        '-v', '--verbose', action='store_true',
        help='Вивести детальні таблиці частот та повну матрицю біграм'
    )
    parser_analyze.add_argument(
        '--show-matrix', action='store_true',
        help='Вивести повну матрицю частот біграм на екран'
    )
    parser_analyze.add_argument(
        '--show-stats', action='store_true',
        help='Показати статистику обробки тексту (кількість символів)'
    )
    parser_analyze.add_argument(
        '--encoding', default='utf-8',
        help='Кодування вхідного файлу (за замовчуванням: utf-8)'
    )
    parser_analyze.set_defaults(func=handle_analyze)

    # --- EXPORT command ---
    parser_export = subparsers.add_parser(
        'export', help='Експорт матриці частот біграм у CSV файл'
    )
    parser_export.add_argument(
        'in_filename', help='Шлях до вхідного текстового файлу'
    )
    parser_export.add_argument(
        '-o', '--output', required=True, help='Шлях до вихідного CSV файлу'
    )
    parser_export.add_argument(
        '--spaces', choices=['include', 'exclude'], default='include',
        help='Включати або виключати пробіли при побудові матриці\n'
             '(за замовчуванням: include)'
    )
    parser_export.add_argument(
        '--bigrams', choices=['overlapping', 'non-overlapping'],
        default='overlapping',
        help='Тип біграм для використання (за замовчуванням: overlapping)'
    )
    parser_export.add_argument(
        '--encoding', default='utf-8',
        help='Кодування вхідного файлу (за замовчуванням: utf-8)'
    )
    parser_export.set_defaults(func=handle_export)

    # --- PREDICT command ---
    parser_predict = subparsers.add_parser(
        'predict', help='Запуск інтерактивного режиму передбачення з матриці'
    )
    parser_predict.add_argument(
        'matrix_filename', help='Шлях до CSV файлу з матрицею біграм'
    )
    parser_predict.set_defaults(func=handle_predict)

    return parser


def main() -> None:
    """Parses command-line arguments and calls the appropriate handler."""
    parser = create_argument_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
