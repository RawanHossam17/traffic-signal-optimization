import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from algorithms.ga import run_ga
from algorithms.hybrid import run_hybrid
from algorithms.pso import run_pso
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation


ROOT = Path(__file__).parent
GRAPH_FILES = {
    "Line chart": ROOT / "line_all.png",
    "Box plot": ROOT / "box_all.png",
    "Bar chart": ROOT / "bar_all.png",
}


class FitnessIndividual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def run_algorithm(name, mutation_type, crossover_type, selection_type, seed):
    set_seed(seed)

    if name == "PSO":
        solution, fitness = run_pso()
    elif name == "GA":
        solution, fitness = run_ga(
            mutation_type=mutation_type,
            crossover_type=crossover_type,
            seed=seed,
        )
    else:
        solution, fitness = run_hybrid(
            mutation_type=mutation_type,
            crossover_type=crossover_type,
            selection_type=selection_type,
            seed=seed,
        )

    return {
        "Algorithm": name,
        "Best Timings": ", ".join(str(int(round(value))) for value in solution),
        "Fitness": round(float(fitness), 4),
    }


def evaluate_manual(timings_text, seed):
    timings = [float(value.strip()) for value in timings_text.split(",") if value.strip()]
    if len(timings) != 6:
        raise ValueError("Enter exactly six green-light timings.")
    if any(value < 10 or value > 60 for value in timings):
        raise ValueError("Each timing must be between 10 and 60 seconds.")

    set_seed(seed)
    simulation = TrafficSimulation(num_intersections=6)
    individual = FitnessIndividual(timings)
    fitness = calculate_fitness(individual, simulation)

    return {
        "Algorithm": "Manual",
        "Best Timings": ", ".join(str(int(round(value))) for value in timings),
        "Fitness": round(float(fitness), 4),
    }


def show_result_cards(results):
    if not results:
        return

    best = min(results, key=lambda row: row["Fitness"])
    cols = st.columns(3)
    cols[0].metric("Best Algorithm", best["Algorithm"])
    cols[1].metric("Best Fitness", best["Fitness"])
    cols[2].metric("Tested Results", len(results))


def show_existing_graphs():
    st.subheader("Generated Graphs")
    tabs = st.tabs(list(GRAPH_FILES.keys()))

    for tab, (title, path) in zip(tabs, GRAPH_FILES.items()):
        with tab:
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)
            else:
                st.warning(f"{path.name} was not found. Run `py -m experiments.plot` to generate it.")


def main():
    st.set_page_config(
        page_title="Traffic Signal Optimization",
        page_icon="🚦",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
        }
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Traffic Signal Optimization")
    st.caption("PSO, GA, and Hybrid optimization dashboard for six traffic intersections.")

    with st.sidebar:
        st.header("Controls")

        algorithm = st.selectbox("Algorithm", ["Hybrid", "PSO", "GA"])
        mutation_type = st.radio(
            "Mutation",
            [1, 2],
            format_func=lambda value: "Type 1: small change" if value == 1 else "Type 2: reset / shuffle",
        )
        crossover_type = st.radio(
            "Crossover",
            [1, 2],
            format_func=lambda value: "Type 1: average / uniform" if value == 1 else "Type 2: choose genes / two-point",
        )
        selection_type = st.radio(
            "Selection",
            [1, 2],
            format_func=lambda value: "Type 1: tournament" if value == 1 else "Type 2: random parents",
        )
        seed_enabled = st.checkbox("Use fixed seed", value=True)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1) if seed_enabled else None

        st.divider()
        manual_timings = st.text_input("Manual timings", value="30, 30, 30, 30, 30, 30")

    if "results" not in st.session_state:
        st.session_state.results = []

    action_cols = st.columns([1, 1, 1, 2])
    run_selected = action_cols[0].button("Run Selected", type="primary")
    compare_all = action_cols[1].button("Compare All")
    evaluate = action_cols[2].button("Evaluate Manual")
    clear = action_cols[3].button("Clear Results")

    if clear:
        st.session_state.results = []

    try:
        if run_selected:
            with st.spinner(f"Running {algorithm}..."):
                st.session_state.results.append(
                    run_algorithm(algorithm, mutation_type, crossover_type, selection_type, seed)
                )

        if compare_all:
            with st.spinner("Running PSO, GA, and Hybrid..."):
                st.session_state.results.extend(
                    [
                        run_algorithm("PSO", mutation_type, crossover_type, selection_type, seed),
                        run_algorithm("GA", mutation_type, crossover_type, selection_type, seed),
                        run_algorithm("Hybrid", mutation_type, crossover_type, selection_type, seed),
                    ]
                )

        if evaluate:
            with st.spinner("Evaluating manual timings..."):
                st.session_state.results.append(evaluate_manual(manual_timings, seed))
    except Exception as exc:
        st.error(str(exc))

    show_result_cards(st.session_state.results)

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Run Results")
        if st.session_state.results:
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Algorithm")["Fitness"])
        else:
            st.info("Choose an action above to start running the optimizers.")

    with right:
        st.subheader("Project Meaning")
        st.write(
            """
            Each solution contains six green-light timings. The simulation sends
            vehicles through the intersections and calculates fitness from waiting
            time plus a congestion penalty. Lower fitness means better traffic flow.
            """
        )
        st.code("fitness = average_wait_time + 2 * congestion", language="text")

    show_existing_graphs()


if __name__ == "__main__":
    main()
