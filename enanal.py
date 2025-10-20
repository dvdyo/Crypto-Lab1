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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

CYRILLIC_ALPHABET = 'абвгдежзийклмнопрстуфхцчшщьыэюя'
CYRILLIC_ALPHABET_WITH_SPACE = CYRILLIC_ALPHABET + ' '

# Normalization table for str.translate()
NORMALIZATION_TABLE = str.maketrans({
    'ё': 'е',
    'ъ': 'ь'
})


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

def get_alphabet_chars(include_spaces: bool) -> str:
    """Get the alphabet characters based on mode."""
    return CYRILLIC_ALPHABET_WITH_SPACE if include_spaces else CYRILLIC_ALPHABET


def get_alphabet_size(include_spaces: bool) -> int:
    """Get the alphabet size based on whether spaces are included."""
    return len(get_alphabet_chars(include_spaces))


def validate_filename(filename: Union[str, Path], expected_ext: Optional[str] = None) -> Path:
    """
    Validate filename format and return Path object.
    
    Args:
        filename: Filename as string or Path
        expected_ext: Expected file extension (e.g., '.csv', '.txt')
    
    Returns:
        Path object
    
    Raises:
        ValueError: If filename is invalid or doesn't have expected extension
    """
    if not filename:
        raise ValueError("Ім'я файлу не може бути порожнім")
    
    path = Path(filename)
    
    # Check if filename is just whitespace
    if not str(path).strip():
        raise ValueError("Ім'я файлу не може бути порожнім")
    
    if expected_ext:
        if not path.suffix.lower() == expected_ext.lower():
            raise ValueError(
                f"Файл повинен мати розширення {expected_ext}, отримано: {path.suffix or '(відсутнє)'}"
            )
    
    return path


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

# Pre-compile regex patterns for performance
_NON_CYRILLIC_PATTERN = re.compile(f'[^{CYRILLIC_ALPHABET_WITH_SPACE}]+')
_MULTI_SPACE_PATTERN = re.compile(r' +')


def preprocess_text(text: str) -> str:
    """
    Preprocesses text according to requirements:
    - Converts to lowercase
    - Normalizes specific letters (ё -> е, ъ -> ь)
    - Keeps only Russian letters and spaces
    - Replaces multiple spaces with single space
    
    Regex-based approach: Let C code (in re module) do the heavy lifting.
    """
    if not text:
        return ""
    
    # Step 1: Lowercase and normalize (fast C operation)
    text = text.lower().translate(NORMALIZATION_TABLE)
    
    # Step 2: Replace all non-Cyrillic/non-space characters with space
    text = _NON_CYRILLIC_PATTERN.sub(' ', text)
    
    # Step 3: Collapse multiple spaces into one (another fast regex operation)
    text = _MULTI_SPACE_PATTERN.sub(' ', text)
    
    # Step 4: Strip leading/trailing spaces
    return text.strip()


# =============================================================================
# CORE ANALYSIS CLASSES
# =============================================================================

class TextEntropyAnalyzer:
    """Analyzes text for statistical and entropy properties."""

    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError("Текст повинен бути рядком (str)")
        
        self.original_text = text
        self.original_length = len(text)
        self.processed_text = preprocess_text(text)
        self.processed_length = len(self.processed_text)
        
        # Cache for text without spaces (lazy initialization)
        self._text_no_spaces: Optional[str] = None
        self._processed_no_spaces_length: Optional[int] = None
        
        self._validate_text()

    def _validate_text(self) -> None:
        """Validates that the processed text is not empty."""
        if not self.processed_text:
            raise ValueError(
                "Оброблений текст порожній. Перевірте, чи містить вхідний файл "
                "кириличні символи."
            )

    def _get_text(self, include_spaces: bool) -> str:
        """Get text with or without spaces (cached)."""
        if include_spaces:
            return self.processed_text
        
        # Lazy initialization of no-spaces text
        if self._text_no_spaces is None:
            self._text_no_spaces = self.processed_text.replace(' ', '')
        return self._text_no_spaces
    
    def _get_no_spaces_length(self) -> int:
        """Lazy calculation of no-spaces length."""
        if self._processed_no_spaces_length is None:
            self._processed_no_spaces_length = len(self._get_text(include_spaces=False))
        return self._processed_no_spaces_length
    
    def get_stats(self, include_no_spaces: bool = False) -> TextStats:
        """Get text processing statistics."""
        return TextStats(
            original_length=self.original_length,
            processed_length=self.processed_length,
            processed_no_spaces_length=self._get_no_spaces_length() if include_no_spaces else None
        )

    @lru_cache(maxsize=16)
    def calculate_letter_frequencies(self, include_spaces: bool = True) -> Dict[str, float]:
        """Calculates the frequency of each letter in the text."""
        text = self._get_text(include_spaces)
        if not text:
            return {}

        total_letters = len(text)
        letter_counts = Counter(text)
        
        return {letter: count / total_letters for letter, count in letter_counts.items()}

    @lru_cache(maxsize=16)
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

    @staticmethod
    def _format_for_display(text: str) -> str:
        """Format text for display, replacing spaces with underscores."""
        return text.replace(' ', '_')

    @staticmethod
    def _get_mode_name(include_spaces: bool) -> str:
        """Get a human-readable name for the analysis mode."""
        return "З ПРОБІЛАМИ" if include_spaces else "БЕЗ ПРОБІЛІВ"

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
            display_letter = self._format_for_display(letter)
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
            display_bigram = self._format_for_display(bigram)
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

        mode_name = self._get_mode_name(include_spaces)
        print(f"\n=== Матриця частот біграм ({mode_name}) ===")
        print(f"Тип: {'перекриваючі' if overlapping else 'неперекриваючі'}")
        print(
            "(Значення помножені на 1000 для зручності читання.\n"
            "Справжні значення можна переглянути у CSV файлі після експорту)"
        )

        # Helper function to format character for matrix display (single char)
        def format_matrix_char(char: str) -> str:
            return '_' if char == ' ' else char

        # Print header
        print(f"{'':>4}", end="")
        for char in chars:
            print(f"{format_matrix_char(char):^3}", end="")
        print()

        # Print matrix rows
        for char1 in chars:
            print(f"{format_matrix_char(char1):^3}", end="")
            for char2 in chars:
                freq = bigram_frequencies.get(char1 + char2, 0) * 1000
                print(f"{freq:3.0f}" if freq > 0 else "  .", end="")
            print()

    def print_summary_results(
        self, 
        results: Dict[str, Dict[str, float]], 
        filename: str,
        modes_analyzed: List[Tuple[bool, bool]]
    ) -> None:
        """Prints the summary results table."""
        # Group modes by bigram type
        modes_by_bigram = {}
        for include_spaces, overlapping in modes_analyzed:
            bigram_type = "overlapping" if overlapping else "non_overlapping"
            if bigram_type not in modes_by_bigram:
                modes_by_bigram[bigram_type] = []
            modes_by_bigram[bigram_type].append(include_spaces)
        
        # Print header
        print("\n" + "=" * 80)
        print("                    ПІДСУМКОВІ РЕЗУЛЬТАТИ АНАЛІЗУ")
        print("=" * 80)
        print(f"Файл: {filename}")
        
        # Print a table for each bigram type
        for bigram_type, space_modes in modes_by_bigram.items():
            overlapping = (bigram_type == "overlapping")
            bigram_label = "ПЕРЕКРИВАЮЧІ БІГРАМИ" if overlapping else "НЕПЕРЕКРИВАЮЧІ БІГРАМИ"
            
            print("-" * 80)
            print(f"\n{bigram_label}")
            print("-" * 80)
            
            # Build header
            header = f"{'Метрика':<30}"
            for include_spaces in space_modes:
                space_label = "з пробілами" if include_spaces else "без пробілів"
                header += f"{space_label}".center(25)
            print(header)
            print("-" * 80)
            
            # Print metrics
            metrics_to_print = [
                ('H1', 'H₁ (біт/символ)'),
                ('H2', 'H₂ (біт/символ)'),
                ('R1', 'Надлишковість R₁ (%)'),
                ('R2', 'Надлишковість R₂ (%)')
            ]
            for metric_key, metric_name in metrics_to_print:
                line = f"{metric_name:<30}"
                multiplier = 100 if '%' in metric_name else 1
                fmt = ".2f" if '%' in metric_name else ".4f"
                
                for include_spaces in space_modes:
                    mode_key = CLI._get_mode_key(include_spaces, overlapping)
                    if mode_key in results:
                        value = results[mode_key][metric_key] * multiplier
                        line += f"{value:^25{fmt}}"
                print(line)
        
        print("=" * 80)


def export_bigram_matrix(
    analyzer: TextEntropyAnalyzer,
    filename: Union[str, Path],
    include_spaces: bool = True,
    overlapping: bool = True
) -> None:
    """Exports the bigram frequency matrix to a CSV file."""
    filepath = validate_filename(filename, expected_ext='.csv')
    
    bigram_frequencies = analyzer.calculate_bigram_frequencies(
        include_spaces, overlapping
    )
    chars = sorted(list(get_alphabet_chars(include_spaces)))

    try:
        with filepath.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + chars)
            for char1 in chars:
                row = [char1] + [
                    bigram_frequencies.get(char1 + char2, 0) for char2 in chars
                ]
                writer.writerow(row)
        print(f"Матриця біграм успішно експортована в '{filepath}'")
    except (IOError, PermissionError) as e:
        print(f"Помилка при записі файлу '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# PREDICTION
# =============================================================================

class BigramPredictor:
    """Handles interactive prediction based on a loaded bigram matrix."""

    def __init__(self, csv_filename: Union[str, Path]):
        self.csv_filepath = validate_filename(csv_filename, expected_ext='.csv')
        
        self.bigram_frequencies: Dict[Tuple[str, str], float] = {}
        self.alphabet: List[str] = []
        self.load_from_csv(self.csv_filepath)

    @staticmethod
    def _format_for_display(text: str) -> str:
        """Format text for display, replacing spaces with underscores."""
        return text.replace(' ', '_')

    def load_from_csv(self, filepath: Path) -> None:
        """Loads a bigram matrix from a CSV file with validation."""
        try:
            with filepath.open('r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # Read and validate header
                header = next(reader)
                if len(header) < 2:
                    raise ValueError(
                        f"CSV файл '{filepath}' має невірну структуру: "
                        f"заголовок повинен містити принаймні 2 колонки"
                    )
                
                self.alphabet = header[1:]
                
                # Validate no duplicate characters in alphabet
                if len(self.alphabet) != len(set(self.alphabet)):
                    raise ValueError(
                        f"CSV файл '{filepath}' містить дублікати символів у заголовку"
                    )
                
                # Validate alphabet size using constants
                expected_sizes = [
                    len(CYRILLIC_ALPHABET),
                    len(CYRILLIC_ALPHABET_WITH_SPACE)
                ]
                if len(self.alphabet) not in expected_sizes:
                    raise ValueError(
                        f"CSV файл '{filepath}' має неочікуваний розмір алфавіту: "
                        f"{len(self.alphabet)} (очікувалось {expected_sizes})"
                    )
                
                row_count = 0
                for row in reader:
                    if not row:
                        continue
                    
                    row_count += 1
                    if len(row) < 2:
                        raise ValueError(
                            f"CSV файл '{filepath}': рядок {row_count + 1} має невірний формат"
                        )
                    
                    char1 = row[0]
                    if char1 not in self.alphabet:
                        raise ValueError(
                            f"CSV файл '{filepath}': символ '{char1}' у рядку {row_count + 1} "
                            f"відсутній у заголовку"
                        )
                    
                    for i, freq_str in enumerate(row[1:]):
                        if i >= len(self.alphabet):
                            break
                        char2 = self.alphabet[i]
                        try:
                            freq = float(freq_str)
                            if freq > 0:
                                self.bigram_frequencies[(char1, char2)] = freq
                        except ValueError:
                            # Skip invalid frequency values
                            continue
                
                # Validate that we read expected number of rows
                if row_count != len(self.alphabet):
                    raise ValueError(
                        f"CSV файл '{filepath}': кількість рядків ({row_count}) "
                        f"не відповідає розміру алфавіту ({len(self.alphabet)})"
                    )
                
            print(f"Матриця біграм успішно завантажена з '{filepath}'")
            print(f"  - Розмір алфавіту: {len(self.alphabet)}")
            print(f"  - Біграм завантажено: {len(self.bigram_frequencies)}")
            
        except FileNotFoundError:
            print(f"Помилка: Файл '{filepath}' не знайдено", file=sys.stderr)
            sys.exit(1)
        except (IOError, PermissionError) as e:
            print(f"Помилка доступу до файлу '{filepath}': {e}", file=sys.stderr)
            sys.exit(1)
        except StopIteration:
            print(f"Помилка: CSV файл '{filepath}' порожній або має невірну структуру", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Помилка валідації: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Непередбачена помилка при читанні CSV файлу '{filepath}': {e}", file=sys.stderr)
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
                display_char = self._format_for_display(lookup_char)

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
            display_next = self._format_for_display(next_char)
            print(f"{rank:<6}{display_next:<15}{freq:<18.10f}{freq*100:<12.4f}%")


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def read_text_file(filename: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Reads a text file with proper error handling and validation."""
    filepath = validate_filename(filename)
    
    try:
        content = filepath.read_text(encoding=encoding)
        if not content:
            print(
                f"Попередження: Файл '{filepath}' порожній",
                file=sys.stderr
            )
        return content
    except FileNotFoundError:
        print(f"Помилка: Файл '{filepath}' не знайдено", file=sys.stderr)
        print(f"Перевірте правильність шляху до файлу", file=sys.stderr)
        sys.exit(1)
    except (PermissionError, IOError) as e:
        print(f"Помилка доступу до файлу '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(
            f"Помилка: Не вдалося прочитати файл '{filepath}' з кодуванням {encoding}",
            file=sys.stderr
        )
        print(f"Деталі помилки: {e}", file=sys.stderr)
        print("Спробуйте вказати інше кодування за допомогою параметра --encoding", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# CLI HANDLER CLASS
# =============================================================================

class CLI:
    """Handles command-line interface operations."""
    
    @staticmethod
    def _validate_top_n(top_n: int) -> None:
        """Validate the top_n parameter."""
        if top_n < -1 or top_n == 0:
            raise ValueError(f"Параметр --top повинен бути -1 (всі) або > 0, отримано: {top_n}")
    
    @staticmethod
    def _get_mode_key(include_spaces: bool, overlapping: bool) -> str:
        """Generate a unique key for a specific analysis mode."""
        space_mode = "with_spaces" if include_spaces else "without_spaces"
        bigram_mode = "overlapping" if overlapping else "non_overlapping"
        return f"{space_mode}_{bigram_mode}"
    
    @staticmethod
    def _get_mode_name(include_spaces: bool) -> str:
        """Get a human-readable name for the analysis mode."""
        return "З ПРОБІЛАМИ" if include_spaces else "БЕЗ ПРОБІЛІВ"
    
    @staticmethod
    def handle_analyze(args: argparse.Namespace) -> None:
        """Handler for the 'analyze' command."""
        # Validate parameters
        try:
            CLI._validate_top_n(args.top)
        except ValueError as e:
            print(f"Помилка: {e}", file=sys.stderr)
            sys.exit(1)
        
        text = read_text_file(args.filename, args.encoding)
        
        try:
            analyzer = TextEntropyAnalyzer(text)
        except (ValueError, TypeError) as e:
            print(f"Помилка аналізу: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Show stats if requested or verbose
        if args.show_stats or args.verbose:
            include_no_spaces = args.spaces in ['exclude', 'both']
            analyzer.get_stats(include_no_spaces=include_no_spaces).print_stats()
        
        reporter = TextEntropyReporter(analyzer)

        # Determine which space modes to run
        if args.spaces == 'include':
            space_modes = [True]
        elif args.spaces == 'exclude':
            space_modes = [False]
        else:  # 'both'
            space_modes = [True, False]
        
        # Determine which bigram modes to run
        if args.bigrams == 'overlapping':
            bigram_modes = [True]
        elif args.bigrams == 'non-overlapping':
            bigram_modes = [False]
        else:  # 'both'
            bigram_modes = [True, False]

        # Calculate results for each combination of modes
        results = {}
        modes_analyzed = []
        
        for include_spaces in space_modes:
            for overlapping in bigram_modes:
                modes_analyzed.append((include_spaces, overlapping))
                mode_key = CLI._get_mode_key(include_spaces, overlapping)
                alphabet_size = get_alphabet_size(include_spaces)

                h1 = analyzer.calculate_h1(include_spaces=include_spaces)
                h2 = analyzer.calculate_h2(include_spaces=include_spaces, overlapping=overlapping)
                r1 = analyzer.calculate_redundancy(h1, alphabet_size)
                r2 = analyzer.calculate_redundancy(h2, alphabet_size)

                results[mode_key] = {'H1': h1, 'H2': h2, 'R1': r1, 'R2': r2}

                # Print detailed analysis if verbose
                if args.verbose:
                    print("\n" + "─" * 50)
                    print(f"ДЕТАЛЬНИЙ АНАЛІЗ ({CLI._get_mode_name(include_spaces)}, "
                          f"{'перекриваючі' if overlapping else 'неперекриваючі'})")
                    print("─" * 50)
                    reporter.print_letter_frequencies(include_spaces=include_spaces, top_n=args.top)
                    reporter.print_top_bigrams(
                        include_spaces=include_spaces,
                        overlapping=overlapping,
                        top_n=args.top
                    )

        # Print summary
        reporter.print_summary_results(results, args.filename, modes_analyzed)

        # Print bigram matrix if requested
        if args.show_matrix or args.verbose:
            for include_spaces, overlapping in modes_analyzed:
                reporter.print_bigram_matrix(include_spaces=include_spaces, overlapping=overlapping)

    @staticmethod
    def handle_export(args: argparse.Namespace) -> None:
        """Handler for the 'export' command."""
        text = read_text_file(args.in_filename, args.encoding)
        
        try:
            analyzer = TextEntropyAnalyzer(text)
        except (ValueError, TypeError) as e:
            print(f"Помилка аналізу: {e}", file=sys.stderr)
            sys.exit(1)
        
        include_spaces = args.spaces == 'include'
        overlapping = args.bigrams == 'overlapping'

        print(f"Експорт матриці біграм з файлу '{args.in_filename}'...")
        print(f"  - Режим пробілів: {'з пробілами' if include_spaces else 'без пробілів'}")
        print(f"  - Тип біграм: {'перекриваючі' if overlapping else 'неперекриваючі'}")

        export_bigram_matrix(
            analyzer, args.output, include_spaces=include_spaces, overlapping=overlapping
        )

    @staticmethod
    def handle_predict(args: argparse.Namespace) -> None:
        """Handler for the 'predict' command."""
        predictor = BigramPredictor(args.matrix_filename)
        predictor.run_interactive()

    @staticmethod
    def add_common_arguments(parser: argparse.ArgumentParser, 
                            include_spaces_both: bool = True,
                            include_bigrams_both: bool = True) -> None:
        """Add common arguments to a parser."""
        
        # Space handling argument
        space_choices = ['include', 'exclude', 'both'] if include_spaces_both else ['include', 'exclude']
        parser.add_argument(
            '--spaces', 
            choices=space_choices, 
            default='include',
            help=(
                'Як обробляти пробіли:\n'
                '  include: аналізувати текст з пробілами (за замовчуванням)\n'
                '  exclude: аналізувати текст без пробілів\n'
                + ('  both:    показати результати для обох випадків\n' if include_spaces_both else '')
            )
        )
        
        # Bigram type argument
        bigram_choices = ['overlapping', 'non-overlapping', 'both'] if include_bigrams_both else ['overlapping', 'non-overlapping']
        parser.add_argument(
            '--bigrams', 
            choices=bigram_choices,
            default='overlapping',
            help=(
                'Тип біграм для використання:\n'
                '  overlapping:     перекриваючі біграми (за замовчуванням)\n'
                '  non-overlapping: неперекриваючі біграми\n'
                + ('  both:            показати результати для обох типів\n' if include_bigrams_both else '')
            )
        )
        
        # Encoding argument
        parser.add_argument(
            '--encoding', 
            default='utf-8',
            help='Кодування вхідного файлу (за замовчуванням: utf-8)'
        )

    @staticmethod
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
            'analyze', 
            help='Аналіз тексту для обчислення ентропії',
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser_analyze.add_argument(
            'filename', 
            help='Шлях до текстового файлу для аналізу'
        )
        CLI.add_common_arguments(parser_analyze, include_spaces_both=True, include_bigrams_both=True)
        parser_analyze.add_argument(
            '-n', '--top', 
            type=int, 
            default=20, 
            metavar='N',
            help=(
                'Показати топ N найчастіших елементів у детальному звіті.\n'
                'За замовчуванням: 20. Вкажіть -1, щоб показати всі.'
            )
        )
        parser_analyze.add_argument(
            '-v', '--verbose', 
            action='store_true',
            help='Вивести детальні таблиці частот та повну матрицю біграм'
        )
        parser_analyze.add_argument(
            '--show-matrix', 
            action='store_true',
            help='Вивести повну матрицю частот біграм на екран'
        )
        parser_analyze.add_argument(
            '--show-stats', 
            action='store_true',
            help='Показати статистику обробки тексту (кількість символів)'
        )
        parser_analyze.set_defaults(func=CLI.handle_analyze)

        # --- EXPORT command ---
        parser_export = subparsers.add_parser(
            'export', 
            help='Експорт матриці частот біграм у CSV файл',
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser_export.add_argument(
            'in_filename', 
            help='Шлях до вхідного текстового файлу'
        )
        parser_export.add_argument(
            '-o', '--output', 
            required=True, 
            help='Шлях до вихідного CSV файлу'
        )
        CLI.add_common_arguments(parser_export, include_spaces_both=False, include_bigrams_both=False)
        parser_export.set_defaults(func=CLI.handle_export)

        # --- PREDICT command ---
        parser_predict = subparsers.add_parser(
            'predict', 
            help='Запуск інтерактивного режиму передбачення з матриці'
        )
        parser_predict.add_argument(
            'matrix_filename', 
            help='Шлях до CSV файлу з матрицею біграм'
        )
        parser_predict.set_defaults(func=CLI.handle_predict)

        return parser

    @staticmethod
    def run() -> None:
        """Main entry point for CLI."""
        parser = CLI.create_argument_parser()
        args = parser.parse_args()
        args.func(args)


if __name__ == "__main__":
    CLI.run()
