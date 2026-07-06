"""
Eval harness for ML Deployment Concierge.

Fast mode (default): exercises guardrails + builder + fast_check (no venv / subprocess).
Full mode (--full):  runs the complete orchestrator including venv smoke-test.
"""
import os
import sys
import time
import shutil
import tempfile
import builder_agent as builder
import validator_agent as validator

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_CASES_DIR = os.path.join(WORKSPACE_DIR, "eval_cases")

TEST_CASES = [
    {
        "name": "Clean Sklearn Model",
        "filename": "clean_sklearn_model.pkl",
        "expected": "success",
        "description": "Standard sklearn .pkl — checks that joblib import triggers missing_import detection, builder adds joblib, fast_check then passes.",
    },
    {
        "name": "Dependency Conflict (Keras + audioop)",
        "filename": "dependency_conflict_model.keras",
        "expected": "success",
        "description": "Keras model injected with audioop import — tests skill-memory lookup + removal of bad import.",
    },
    {
        "name": "Corrupted Model",
        "filename": "corrupted_model.keras",
        "expected": "failed",
        "description": "Junk-byte .keras file — builder generates bundle, validator detects bad_import (non_existent_package_xyz), pipeline terminates cleanly.",
    },
    {
        "name": "Oversized Model",
        "filename": "oversized_model.bin",
        "expected": "failed",
        "description": "11 MB file — guardrail rejects immediately before any agent runs.",
    },
]


def run_fast_case(case: dict) -> dict:
    """Exercise one test case using fast_check (no subprocess)."""
    model_path = os.path.join(EVAL_CASES_DIR, case["filename"])
    start = time.time()
    retries = 0

    # ── Guardrail ─────────────────────────────────────────────────────────────
    try:
        framework = builder.detect_framework(model_path)
    except (FileNotFoundError, ValueError) as e:
        elapsed = time.time() - start
        actual = "failed"
        return {
            **case,
            "actual": actual,
            "passed": actual == case["expected"],
            "retries": 0,
            "time": elapsed,
            "note": f"Guardrail: {e}",
        }

    # ── Initial build ──────────────────────────────────────────────────────────
    bundle_dir = tempfile.mkdtemp(prefix="eval_bundle_")
    try:
        model_filename = os.path.basename(model_path)
        shutil.copy2(model_path, os.path.join(bundle_dir, model_filename))

        inject_audioop = "conflict" in model_filename.lower()
        inject_bad = "corrupted" in model_filename.lower()

        reqs    = builder.generate_requirements(framework)
        app     = builder.generate_gradio_app(framework, model_filename,
                                              inject_audioop=inject_audioop,
                                              inject_bad_import=inject_bad)
        dfile   = builder.generate_dockerfile(framework)
        readme  = builder.generate_readme(framework, model_filename)
        builder.write_files(bundle_dir, app, reqs, dfile, readme)

        # ── Validation + self-correction loop (max 3 retries) ─────────────────
        max_retries = 3
        skill_memory_path = os.path.join(WORKSPACE_DIR, "skill_memory.json")
        success = False
        note = ""

        while retries <= max_retries:
            ok, logs = validator.fast_check(bundle_dir)
            if ok:
                success = True
                note = "fast_check passed"
                break

            retries += 1
            if retries > max_retries:
                note = f"Max retries reached. Last log: {logs.splitlines()[-1]}"
                break

            err_report = validator.classify_error(logs)
            # Corrupted model → hard stop
            if err_report["category"] == "corrupted_model":
                note = "Corrupted model detected — stopping."
                break

            rev_reqs, rev_df, rev_app = builder.builder_revise(
                bundle_dir, framework, err_report, skill_memory_path
            )
            builder.write_files(bundle_dir, rev_app, rev_reqs, rev_df, readme)

        actual = "success" if success else "failed"
        return {
            **case,
            "actual": actual,
            "passed": actual == case["expected"],
            "retries": retries,
            "time": time.time() - start,
            "note": note,
        }
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)


def print_results(results: list[dict]):
    print("\n" + "=" * 85)
    print("                         ML DEPLOYMENT CONCIERGE — EVAL RESULTS")
    print("=" * 85)
    header = f"{'Test Case':<35} {'Expected':<10} {'Actual':<10} {'Eval':<6} {'Retries':<8} {'Time':>7}"
    print(header)
    print("-" * 85)
    all_passed = True
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"{r['name']:<35} {r['expected']:<10} {r['actual']:<10} {tag:<9} {r['retries']:<8} {r['time']:>6.2f}s")
        print(f"  > {r['note']}")
    print("=" * 85)
    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"Overall: {overall}   ({sum(r['passed'] for r in results)}/{len(results)} cases correct)")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    mode = "--full" if "--full" in sys.argv else "--fast"
    print(f"\nRunning eval harness in {mode} mode...")

    if mode == "--fast":
        results = [run_fast_case(c) for c in TEST_CASES]
        print_results(results)
    else:
        # Full mode: use orchestrator with real subprocess smoke-test
        from orchestrator import run_orchestrator
        results = []
        for case in TEST_CASES:
            model_path = os.path.join(EVAL_CASES_DIR, case["filename"])
            start = time.time()
            final_event = None
            retry_count = 0
            print(f"\n▶  {case['name']} ({case['filename']})")
            for event in run_orchestrator(model_path, WORKSPACE_DIR, max_retries=3):
                print(f"   [{event['agent']}] {event['message']}")
                retry_count = event.get("retry", 0)
                if event["type"] in ("success", "error"):
                    final_event = event
            actual = "success" if final_event and final_event["type"] == "success" else "failed"
            results.append({
                **case,
                "actual": actual,
                "passed": actual == case["expected"],
                "retries": retry_count,
                "time": time.time() - start,
                "note": (final_event or {}).get("message", ""),
            })
        print_results(results)
