# PHM Demo Tutorial

This tutorial walks through the minimal pipeline implemented in `src/phm_demo.py`.

1. **Generate Example Signals**
   - Random reference signals are created for four labels.
   - A single test signal is generated.
   - A simple processing plan is prepared using the operator factory.
2. **Build The Graph**
   - The demo uses the `StateGraph` API to define a linear flow:
     planner -> execute -> decide -> report.
   - Each node updates the shared state and passes it along.
3. **Run The Demo**
   - Execute `python -m src.phm_demo` to process the signals and print the predicted label.

The demo showcases how operators can be combined in a LangGraph graph. The `plan`
list can be extended with additional operators to perform more complex analysis.
