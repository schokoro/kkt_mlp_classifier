# S04 fix: incomplete stage timing

You are the production-code executor; directly fix the failure documented by independent tests in `tests/test_s04_comparison_cost_contract.py`. Two cases expose partial sums when any successful own or shared stage has unknown `seconds`; complete scenario and aggregate cost must remain null, not a partial sum. Preserve S03 semantics, shared G×I accounting, source files and data; do not modify tests or commit. Run the focused tests and report results.
