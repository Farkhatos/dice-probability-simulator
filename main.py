# ==============================
# Импорт библиотек
# ==============================
import random
import matplotlib.pyplot as plt
import csv
import argparse
from typing import List, Dict, Tuple
from tabulate import tabulate

# Установка matplotlib inline только для Jupyter
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# ==============================
# Глобальные константы
# ==============================
THEORETICAL_SUM_COUNTS = {
    3: 1, 4: 3, 5: 6, 6: 10, 7: 15, 8: 21, 9: 25, 10: 27,
    11: 27, 12: 25, 13: 21, 14: 15, 15: 10, 16: 6, 17: 3, 18: 1
}

# ==============================
# Теоретические вычисления
# ==============================
def is_prime(n: int) -> bool:
    """Checks if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def theoretical_three_prime() -> float:
    """Calculates theoretical probability that sum of three dice is prime."""
    favorable = sum(THEORETICAL_SUM_COUNTS.get(i, 0) for i in [2, 3, 5, 7, 11, 13, 17])
    return favorable / (6 ** 3)

def theoretical_three_divisible_by_nth() -> float:
    """Calculates theoretical probability that sum of three dice is divisible by Nth die."""
    total_combinations = 6 ** 3 * 6  # 1296
    favorable = 0
    for d in range(1, 7):
        for s in range(3, 19):
            if s % d == 0 and s in THEORETICAL_SUM_COUNTS:
                favorable += THEORETICAL_SUM_COUNTS[s]
    return favorable / total_combinations

def theoretical_three_divisible_by(divisor: int) -> float:
    """Calculates theoretical probability that sum of three dice is divisible by divisor."""
    favorable = sum(count for s, count in THEORETICAL_SUM_COUNTS.items() if s % divisor == 0)
    return favorable / (6 ** 3)

def theoretical_max_three(k: int) -> float:
    """Calculates theoretical probability that max of three dice equals k."""
    return (k ** 3 - (k - 1) ** 3) / (6 ** 3)

def theoretical_min_three(k: int) -> float:
    """Calculates theoretical probability that min of three dice equals k."""
    return ((7 - k) ** 3 - (6 - k) ** 3) / (6 ** 3)

# ==============================
# Визуализация и сохранение результатов
# ==============================
def plot_distribution(sum_counts: Dict[int, int], N: int, threshold: int, n: int, save: bool = False) -> None:
    """Plots histogram of sum distribution."""
    sums = list(sum_counts.keys())
    frequencies = [sum_counts[i] / n * 100 for i in sums]
    colors = ['#4e79a7' if i % 2 == 0 else '#e15759' for i in sums]

    plt.figure(figsize=(12, 6))
    plt.bar(sums, frequencies, color=colors, align='center', width=0.4)
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold={threshold}')
    plt.xlabel("Sum of N Dice")
    plt.ylabel("Probability (%)")
    plt.title(f"Distribution of Sums for {N} Dice (n={n})")
    plt.xticks(sums[::2])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(f"dice_distribution_N{N}_n{n}.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_csv(results: Dict[str, float], N: int, n: int, threshold: int, 
                        max_three_target: int, min_three_target: int, divisor: int, 
                        filename: str = "dice_results.csv", note: str = "") -> None:
    """Saves results to CSV with optional note."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Theory", "Note"])
        writer.writerow(["Even Sum", f"{results['prob_even']:.4f}", "~ 0.5", note])
        writer.writerow(["Odd Sum", f"{results['prob_odd']:.4f}", "~ 0.5", note])
        writer.writerow([f"Sum > {threshold}", f"{results['prob_greater']:.4f}", "-", note])
        writer.writerow([f"Sum <= {threshold}", f"{results['prob_less_equal']:.4f}", "-", note])
        writer.writerow([f"Extreme Sums (<= {N+2} or >= {6*N-2})", f"{results['prob_extreme']:.4f}", "-", note])
        writer.writerow(["All Dice Equal", f"{results['prob_all_same']:.6f}", f"{results['theoretical_all_same']:.6f}", note])
        writer.writerow(["Mean", f"{results['mean_deviation']:.4f}", f"{results['expected_sum']:.1f}", note])
        if N >= 3:
            writer.writerow(["Sum of Three Dice Odd", f"{results['prob_three_odd']:.4f}", "~ 0.5", note])
            writer.writerow(["Sum of Three Dice is Prime", f"{results['prob_three_prime']:.4f}", f"{results['theoretical_three_prime']:.4f}", note])
            writer.writerow([f"Max of Three Dice = {max_three_target}", f"{results['prob_max_three_target']:.4f}", f"{results['theoretical_max_three']:.4f}", note])
            writer.writerow([f"Min of Three Dice = {min_three_target}", f"{results['prob_three_min_target']:.4f}", f"{results['theoretical_min_three']:.4f}", note])
            writer.writerow([f"Sum of First Three Divisible by {N}th", f"{results['prob_three_divisible']:.4f}", f"{results['theoretical_three_divisible']:.4f}", note])
            writer.writerow([f"Sum of Three Dice Divisible by {divisor}", f"{results['prob_three_divisible_by']:.4f}", f"{results['theoretical_three_divisible_by']:.4f}", note])
        if N >= 6:
            writer.writerow(["Sum of First Three > Last Three", f"{results['prob_first_three_greater_last_three']:.4f}", "~ 0.5", note])

# ==============================
# Симуляция бросков
# ==============================
class DiceSimulation:
    """Class to manage dice simulation and analysis."""
    def __init__(self, N: int, n: int, threshold: int, max_three_target: int, min_three_target: int, divisor: int):
        self.N = N
        self.n = n
        self.threshold = threshold
        self.max_three_target = max_three_target
        self.min_three_target = min_three_target
        self.divisor = divisor
        self.expected_sum = 3.5 * N
        self.theoretical_all_same = 1 / (6 ** (N - 1)) if N >= 1 else 0
        self.theoretical_three_divisible = theoretical_three_divisible_by_nth() if N >= 3 else 0
        self.theoretical_three_prime = theoretical_three_prime() if N >= 3 else 0
        self.theoretical_max_three = theoretical_max_three(max_three_target) if N >= 3 else 0
        self.theoretical_min_three = theoretical_min_three(min_three_target) if N >= 3 else 0
        self.theoretical_three_divisible_by = theoretical_three_divisible_by(divisor) if N >= 3 else 0

    def simulate_dice(self) -> Tuple[List[int], List[List[int]]]:
        """Simulates n throws of N dice, returns sums and dice rolls."""
        sum_list = []
        dice_rolls = []
        for _ in range(self.n):
            dice = [random.randint(1, 6) for _ in range(self.N)]
            sum_list.append(sum(dice))
            dice_rolls.append(dice)
        return sum_list, dice_rolls

    def compute_basic_stats(self, sum_list: List[int], dice_rolls: List[List[int]]) -> Dict[str, float]:
        """Computes basic statistics for dice sums."""
        sum_counts = {i: 0 for i in range(self.N, 6 * self.N + 1)}
        even_sum = odd_sum = sum_greater = sum_less_equal = extreme_sums = all_same = 0
        deviations = []

        for current_sum, dice in zip(sum_list, dice_rolls):
            sum_counts[current_sum] += 1
            if current_sum % 2 == 0:
                even_sum += 1
            else:
                odd_sum += 1
            if current_sum > self.threshold:
                sum_greater += 1
            else:
                sum_less_equal += 1
            if current_sum <= self.N + 2 or current_sum >= 6 * self.N - 2:
                extreme_sums += 1
            if len(set(dice)) == 1:  # Оптимизация
                all_same += 1
            deviations.append(abs(current_sum - self.expected_sum))

        return {
            "sum_counts": sum_counts,
            "prob_even": even_sum / self.n,
            "prob_odd": odd_sum / self.n,
            "prob_greater": sum_greater / self.n,
            "prob_less_equal": sum_less_equal / self.n,
            "prob_extreme": extreme_sums / self.n,
            "prob_all_same": all_same / self.n,
            "mean_deviation": sum(deviations) / self.n
        }

    def analyze_three_dice_patterns(self, dice_rolls: List[List[int]]) -> Dict[str, float]:
        """Analyzes patterns for three dice if N >= 3."""
        if self.N < 3:
            return {
                "prob_three_odd": 0.0,
                "prob_three_prime": 0.0,
                "prob_max_three_target": 0.0,
                "prob_three_min_target": 0.0,
                "prob_three_divisible_by": 0.0,
                "prob_three_divisible": 0.0
            }

        three_odd_sum = three_prime_sum = max_three_target = three_min_target = three_divisible_by = three_divisible_by_nth = 0
        for dice in dice_rolls:
            # Используем первые три кубика для стабильности
            three_sum = sum(dice[i] for i in range(3))
            if three_sum % 2 == 1:
                three_odd_sum += 1
            if is_prime(three_sum):
                three_prime_sum += 1
            if max(dice[i] for i in range(3)) == self.max_three_target:
                max_three_target += 1
            if min(dice[i] for i in range(3)) == self.min_three_target:
                three_min_target += 1
            if three_sum % self.divisor == 0:
                three_divisible_by += 1
            first_three_sum = sum(dice[:3])
            nth_die = dice[-1]
            if nth_die != 0 and first_three_sum % nth_die == 0:  # Защита от деления на ноль
                three_divisible_by_nth += 1

        return {
            "prob_three_odd": three_odd_sum / self.n,
            "prob_three_prime": three_prime_sum / self.n,
            "prob_max_three_target": max_three_target / self.n,
            "prob_three_min_target": three_min_target / self.n,
            "prob_three_divisible_by": three_divisible_by / self.n,
            "prob_three_divisible": three_divisible_by_nth / self.n
        }

    def analyze_six_dice_patterns(self, dice_rolls: List[List[int]]) -> Dict[str, float]:
        """Analyzes patterns for six dice if N >= 6."""
        if self.N < 6:
            return {"prob_first_three_greater_last_three": 0.0}

        first_three_greater_last_three = 0
        for dice in dice_rolls:
            first_three_sum = sum(dice[:3])
            last_three_sum = sum(dice[-3:])
            if first_three_sum > last_three_sum:
                first_three_greater_last_three += 1

        return {"prob_first_three_greater_last_three": first_three_greater_last_three / self.n}

    def analyze_results(self, sum_list: List[int], dice_rolls: List[List[int]]) -> Dict[str, float]:
        """Combines all analysis results."""
        basic_stats = self.compute_basic_stats(sum_list, dice_rolls)
        three_dice_stats = self.analyze_three_dice_patterns(dice_rolls)
        six_dice_stats = self.analyze_six_dice_patterns(dice_rolls)

        results = {
            **basic_stats,
            **three_dice_stats,
            **six_dice_stats,
            "expected_sum": self.expected_sum,
            "theoretical_all_same": self.theoretical_all_same,
            "theoretical_three_divisible": self.theoretical_three_divisible,
            "theoretical_three_prime": self.theoretical_three_prime,
            "theoretical_max_three": self.theoretical_max_three,
            "theoretical_min_three": self.theoretical_min_three,
            "theoretical_three_divisible_by": self.theoretical_three_divisible_by
        }
        return results

    def print_results(self) -> Dict[str, float]:
        """Prints results in a formatted table."""
        results = self.analyze_results(*self.simulate_dice())
        table = [
            ["Even Sum", f"{results['prob_even']:.4f}", "~ 0.5"],
            ["Odd Sum", f"{results['prob_odd']:.4f}", "~ 0.5"],
            [f"Sum > {self.threshold}", f"{results['prob_greater']:.4f}", "-"],
            [f"Sum <= {self.threshold}", f"{results['prob_less_equal']:.4f}", "-"],
            [f"Extreme Sums (<= {self.N+2} or >= {6*self.N-2})", f"{results['prob_extreme']:.4f}", "-"],
            ["All Dice Equal", f"{results['prob_all_same']:.6f}", f"{results['theoretical_all_same']:.6f}"],
            ["Mean", f"{results['mean_deviation']:.4f}", f"{results['expected_sum']:.1f}"],
        ]
        if self.N >= 3:
            table.extend([
                ["Sum of Three Dice Odd", f"{results['prob_three_odd']:.4f}", "~ 0.5"],
                ["Sum of Three Dice is Prime", f"{results['prob_three_prime']:.4f}", f"{results['theoretical_three_prime']:.4f}"],
                [f"Max of Three Dice = {self.max_three_target}", f"{results['prob_max_three_target']:.4f}", f"{results['theoretical_max_three']:.4f}"],
                [f"Min of Three Dice = {self.min_three_target}", f"{results['prob_three_min_target']:.4f}", f"{results['theoretical_min_three']:.4f}"],
                [f"Sum of First Three Divisible by {self.N}th", f"{results['prob_three_divisible']:.4f}", f"{results['theoretical_three_divisible']:.4f}"],
                [f"Sum of Three Dice Divisible by {self.divisor}", f"{results['prob_three_divisible_by']:.4f}", f"{results['theoretical_three_divisible_by']:.4f}"]
            ])
        if self.N >= 6:
            table.append(["Sum of First Three > Last Three", f"{results['prob_first_three_greater_last_three']:.4f}", "~ 0.5"])

        print(f"\nDice Simulation Results for N={self.N}, n={self.n}, threshold={self.threshold}, "
              f"max_three_target={self.max_three_target}, min_three_target={self.min_three_target}, divisor={self.divisor}")
        print(tabulate(table, headers=["Metric", "Value", "Theory"], tablefmt="fancy_grid", numalign="center"))
        return results

# ==============================
# Главная функция
# ==============================
def main() -> None:
    """Main function to run the dice simulation with text input or command-line arguments."""
    parser = argparse.ArgumentParser(description="Dice Probability Explorer: Simulate dice throws and analyze probabilities.")
    parser.add_argument("--N", type=int, default=None, help="Number of dice (positive integer)")
    parser.add_argument("--n", type=int, default=None, help="Number of simulations (positive integer)")
    parser.add_argument("--threshold", type=int, default=None, help="Threshold sum for comparisons")
    parser.add_argument("--max_three_target", type=int, default=None, help="Target maximum for three dice (1-6)")
    parser.add_argument("--min_three_target", type=int, default=None, help="Target minimum for three dice (1-6)")
    parser.add_argument("--divisor", type=int, default=None, help="Divisor for sum of three dice (1-18)")
    parser.add_argument("--save_plot", action="store_true", help="Save histogram to file")
    parser.add_argument("--csv_filename", default="dice_results.csv", help="CSV output filename")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--note", default="", help="Optional note for CSV output")

    # Проверяем, запущен ли код в Jupyter (интерактивно) или как скрипт
    try:
        get_ipython()  # Если в Jupyter, это не вызовет ошибку
        args = parser.parse_args([])  # Пустые аргументы для Jupyter
        is_jupyter = True
    except NameError:
        args = parser.parse_args()
        is_jupyter = False

    try:
        if is_jupyter or any(arg is None for arg in [args.N, args.n, args.threshold, args.max_three_target, 
                                                     args.min_three_target, args.divisor]):
            # Текстовый ввод для Jupyter или если аргументы не заданы
            N = int(input("Enter number of dice (N): ")) if args.N is None else args.N
            n = int(input("Enter number of simulations (n): ")) if args.n is None else args.n
            if N < 1 or n < 1:
                raise ValueError("N and n must be positive")
            
            while True:
                try:
                    threshold = int(input(f"Enter threshold sum (from {N} to {6*N}): ")) if args.threshold is None else args.threshold
                    if N <= threshold <= 6 * N:
                        break
                    print(f"Error: threshold must be in range [{N}, {6*N}]")
                except ValueError:
                    print("Error: enter an integer")
            
            while True:
                try:
                    max_three_target = int(input("Enter target maximum for three dice (1 to 6): ")) if args.max_three_target is None else args.max_three_target
                    if 1 <= max_three_target <= 6:
                        break
                    print("Error: target maximum must be in range [1, 6]")
                except ValueError:
                    print("Error: enter an integer")
            
            while True:
                try:
                    min_three_target = int(input("Enter target minimum for three dice (1 to 6): ")) if args.min_three_target is None else args.min_three_target
                    if 1 <= min_three_target <= 6:
                        break
                    print("Error: target minimum must be in range [1, 6]")
                except ValueError:
                    print("Error: enter an integer")
            
            while True:
                try:
                    divisor = int(input("Enter divisor for sum of three dice (1 to 18): ")) if args.divisor is None else args.divisor
                    if 1 <= divisor <= 18:
                        break
                    print("Error: divisor must be in range [1, 18]")
                except ValueError:
                    print("Error: enter an integer")
            
            save_plot_input = input("Save plot? (yes/no): ") if not args.save_plot else "yes" if args.save_plot else "no"
            save_plot = save_plot_input.lower() == 'yes'
            csv_filename = input("Enter CSV filename: ") or "dice_results.csv" if args.csv_filename == "dice_results.csv" else args.csv_filename
            seed_input = input("Enter random seed (or leave empty for no seed): ") if args.seed is None else str(args.seed)
            seed = int(seed_input) if seed_input and seed_input.isdigit() else None
            note = input("Enter note for CSV (optional): ") or "" if args.note == "" else args.note
        else:
            N = args.N
            n = args.n
            threshold = args.threshold
            max_three_target = args.max_three_target
            min_three_target = args.min_three_target
            divisor = args.divisor
            save_plot = args.save_plot
            csv_filename = args.csv_filename
            seed = args.seed
            note = args.note

        # Установка seed для воспроизводимости
        if seed is not None:
            random.seed(seed)

        sim = DiceSimulation(N, n, threshold, max_three_target, min_three_target, divisor)
        results = sim.print_results()
        plot_distribution(results["sum_counts"], N, threshold, n, save=save_plot)
        save_results_to_csv(results, N, n, threshold, max_three_target, min_three_target, divisor, filename=csv_filename, note=note)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
