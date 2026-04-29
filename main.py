from algorithms.hybrid import run_hybrid


def main():
    print("Running Hybrid...\n")

    best_solution, best_fitness = run_hybrid()

    print("\n===== FINAL RESULT =====")
    print("Best Solution:\n", best_solution)
    print("Best Fitness:", best_fitness)


if __name__ == "__main__":
    main()