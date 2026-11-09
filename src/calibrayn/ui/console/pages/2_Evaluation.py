import pandas as pd
import streamlit as st

from calibrayn.evaluation import report
from calibrayn.evaluation.benchmark import (
    BenchmarkError,
    evaluate_benchmark,
    from_library,
    from_rows,
    table_rows,
)
from calibrayn.evaluation.dataset import DatasetError, build_dataset, save_dataset
from calibrayn.evaluation.personas import PERSONAS
from calibrayn.evaluation.retrieval import evaluate_retrieval
from calibrayn.evaluation.simulation import Setup, run_personas
from calibrayn.hub.rows import HubError, describe_rows, load_rows
from calibrayn.llm.base import LLMError
from calibrayn.ui.components.palette import AMBER, BLUE
from calibrayn.ui.components.sidebar import show_console_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Evaluation", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_console_sidebar(workspace)

st.title("Evaluation")
st.caption(
    "Two checks for the report: does search find the right notes, and does the schedule help?"
)

search_tab, public_tab, simulation_tab = st.tabs(
    ["Search check", "Public benchmark", "Learner simulation"]
)

with search_tab:
    st.write(
        "Questions are written from your own notes. Each one remembers which topic it came from. "
        "The check then asks whether search finds that topic."
    )
    count = st.slider("How many questions", 10, 100, 30, key="eval_count")
    if st.button("Write questions from my notes", key="eval_build"):
        try:
            with st.spinner("Writing questions..."):
                items = build_dataset(
                    workspace.library, workspace.writer, count, workspace.client.name
                )
            st.session_state["eval_items"] = items
            st.session_state.pop("eval_scores", None)
        except (DatasetError, LLMError) as error:
            st.error(str(error))

    items = st.session_state.get("eval_items")
    if items:
        st.success(f"{len(items)} questions are ready.")
        if workspace.client.name == "mock":
            st.warning(
                "The offline model writes template questions. Connect a real model for the report."
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Level": item.level, "Question": item.question, "Document": item.document}
                    for item in items
                ]
            ),
            hide_index=True,
        )
        if st.button("Check search", key="eval_check"):
            st.session_state["eval_scores"] = evaluate_retrieval(workspace.retriever, items)

    scores = st.session_state.get("eval_scores")
    if scores:
        columns = st.columns(len(scores.hit_at) + 1)
        for column, (k, value) in zip(columns, scores.hit_at.items(), strict=False):
            column.metric(f"Found in top {k}", f"{value:.0%}")
        columns[-1].metric("Average rank score", f"{scores.mrr:.2f}")
        st.caption("Average rank score is 1.00 when the right topic is always first.")
        if scores.misses:
            st.write(f"{scores.misses} questions did not find their topic at all.")
        try:
            comparison = evaluate_benchmark(
                from_library(workspace.library, items), workspace.retriever.embedder
            )
        except BenchmarkError as error:
            st.error(str(error))
        else:
            st.subheader("Compared with a keyword search and with chance")
            st.dataframe(pd.DataFrame(table_rows(comparison)), hide_index=True)
            if comparison.corpus_size <= 10:
                st.warning(
                    f"There are only {comparison.corpus_size} topics, so every system looks good. "
                    "Use many more topics for the report."
                )
        if items:
            path = save_dataset(
                items, workspace.settings.data_dir / "evaluation" / "questions.jsonl"
            )
            st.download_button(
                "Download the questions",
                path.read_bytes(),
                file_name="questions.jsonl",
                key="eval_download",
            )

with public_tab:
    st.write(
        "Score search on a public dataset from the Hugging Face Hub, next to a keyword search and "
        "chance. Each row needs a question and the passage that answers it."
    )
    repo = st.text_input(
        "Dataset on the Hub", key="hub_repo", help="For example: organisation/name"
    )
    config = st.text_input("Configuration (leave empty if there is none)", key="hub_config")
    split = st.text_input("Split", value="test", key="hub_split")
    rows_wanted = st.slider("How many rows", 50, 2000, 500, key="hub_limit")

    if st.button("Show the columns", disabled=not repo, key="hub_inspect"):
        try:
            sample = load_rows(repo, config or None, split, 3)
            st.session_state["hub_columns"] = describe_rows(sample)
        except HubError as error:
            st.error(str(error))
    if st.session_state.get("hub_columns"):
        st.dataframe(pd.DataFrame(st.session_state["hub_columns"]), hide_index=True)

    left, right = st.columns(2)
    question_column = left.text_input("Question column", key="hub_question")
    passage_column = right.text_input("Passage column", key="hub_context")
    ready = bool(repo and question_column and passage_column)
    if st.button("Run the benchmark", disabled=not ready, key="hub_run"):
        try:
            with st.spinner("Scoring..."):
                rows = load_rows(repo, config or None, split, rows_wanted)
                benchmark = from_rows(rows, question_column, passage_column)
                st.session_state["hub_report"] = evaluate_benchmark(
                    benchmark, workspace.retriever.embedder
                )
        except (HubError, BenchmarkError) as error:
            st.error(str(error))
    outcome = st.session_state.get("hub_report")
    if outcome:
        st.write(
            f"{outcome.queries} questions over {outcome.corpus_size} passages "
            f"({outcome.skipped} rows skipped)."
        )
        st.dataframe(pd.DataFrame(table_rows(outcome)), hide_index=True)
        st.caption(
            "A system is useful only if it clearly beats random guessing and keyword search."
        )

with simulation_tab:
    st.write(
        "Simulated learners study the same notes with plain SM-2 and with the "
        "confidence-adjusted schedule. A positive difference means the adjusted schedule kept more."
    )
    chosen = st.multiselect(
        "Learners", sorted(PERSONAS), default=sorted(PERSONAS), key="sim_personas"
    )
    for name in chosen:
        st.caption(f"{name}: {PERSONAS[name].summary}")
    left, middle, right = st.columns(3)
    days = left.slider("Days", 14, 120, 60, key="sim_days")
    seed_count = middle.slider("Repeats", 3, 30, 10, key="sim_seeds")
    topics = right.slider("Topics", 10, 80, 40, key="sim_topics")

    if st.button("Run simulation", disabled=not chosen, key="sim_run"):
        with st.spinner("Simulating..."):
            st.session_state["sim_results"] = run_personas(
                chosen, seed_count, Setup(topics=topics, days=days), workspace.tuning
            )

    results = st.session_state.get("sim_results")
    if results:
        rows = [row for row in report.comparison_rows(results) if row["metric"] == "retention"]
        table = pd.DataFrame(
            [
                {
                    "Learner": row["persona"],
                    "Plain SM-2": row["plain"],
                    "Adjusted": row["adjusted"],
                    "Difference": row["difference"],
                    "Low": row["low"],
                    "High": row["high"],
                }
                for row in rows
            ]
        )
        st.subheader("How much each learner remembers at the end")
        st.dataframe(table, hide_index=True)
        st.bar_chart(
            table.set_index("Learner")[["Plain SM-2", "Adjusted"]], color=[BLUE, AMBER], stack=False
        )
        st.caption("Low and High show the 95 percent range of the difference.")

        picked = st.selectbox(
            "Weekly memory for", [item.persona for item in results], key="sim_curve"
        )
        curve = next(item for item in results if item.persona == picked)
        weekly = pd.DataFrame(
            {
                "Week": range(1, len(curve.plain_curve) + 1),
                "Plain SM-2": curve.plain_curve,
                "Adjusted": curve.adjusted_curve,
            }
        )
        st.line_chart(weekly.set_index("Week"), color=[BLUE, AMBER])

        full = report.comparison_rows(results)
        csv_text = "\n".join(
            [",".join(full[0])] + [",".join(str(value) for value in row.values()) for row in full]
        )
        st.download_button(
            "Download all results (CSV)",
            csv_text,
            file_name="persona_comparison.csv",
            key="sim_download",
        )
