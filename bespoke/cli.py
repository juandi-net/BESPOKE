"""BESPOKE CLI — single entrypoint for the full pipeline."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def find_latest_adapter() -> Optional[Path]:
    """Find the most recently modified adapter directory with an sft/ subfolder."""
    from bespoke.config import config

    if not config.adapters_dir.exists():
        return None

    candidates = []
    for d in config.adapters_dir.iterdir():
        if d.is_dir() and (d / "sft").is_dir():
            candidates.append(d)

    if not candidates:
        return None

    return max(candidates, key=lambda p: (p / "sft").stat().st_mtime)


def find_previous_scorecard(adapter_name: str) -> Optional[dict]:
    """Find the most recent scorecard JSON for the given adapter."""
    import json
    from bespoke.config import config

    if not config.scorecards_dir.exists():
        return None

    cards = sorted(config.scorecards_dir.glob(f"{adapter_name}-*.json"))
    if not cards:
        return None

    with open(cards[-1]) as f:
        return json.load(f)


def benchmark_exists() -> bool:
    """Check if a benchmark YAML has been generated."""
    from bespoke.config import config
    return (config.benchmark_dir / "benchmark.yaml").exists()


# ── Commands ──────────────────────────────────────────────────────────


def cmd_capture(args):
    """Stage 1: Ingest interactions, compute embeddings, write to DB."""
    web = getattr(args, "web", False)
    all_sources = getattr(args, "all", False)

    if (web or all_sources) and getattr(args, "jsonl", None):
        print("Error: --web and --all cannot be used with --jsonl")
        sys.exit(1)

    if web or all_sources:
        from bespoke.capture.web_capture import run_web_capture
        web_stats = run_web_capture()
        print(f"Web: {web_stats['interactions_captured']} captured "
              f"from {web_stats['conversations_fetched']} conversations")

    if not web or all_sources:
        from bespoke.capture.pipeline import run_capture

        since = None
        if args.since:
            since = datetime.fromisoformat(args.since)
        elif args.backfill:
            since = datetime(2020, 1, 1)

        jsonl_path = Path(args.jsonl) if args.jsonl else None
        session_dir = Path(args.session_dir) if args.session_dir else None

        stats = run_capture(
            session_dir=session_dir,
            jsonl_path=jsonl_path,
            since=since,
        )


def cmd_extract(args):
    """Stage 2a: Classify interactions and produce training pairs."""
    if args.reset:
        from bespoke.db.init import get_connection
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        print(f"Resetting Stage 2a: deleting {count} training pairs, "
              "clearing all extraction outputs...")
        conn.execute("DELETE FROM training_pairs")
        conn.execute("""
            UPDATE interactions SET
                domain = NULL, quality_score = NULL,
                reasoning_primitives = NULL, feedback_class = NULL,
                feedback_raw = NULL, processed_2a_at = NULL,
                stage2a_fail_count = 0
        """)
        conn.commit()
        conn.close()
        print("Reset complete. All Stage 1 data preserved.")

    from bespoke.teach.stage2a import run_stage_2a
    run_stage_2a()


def cmd_train(args):
    """Stage 3: DoRA fine-tuning via MLX."""
    if args.deadline or args.max:
        from bespoke.train.train_sft import run_search

        print("BESPOKE Stage 3: Autoresearch Search Loop")
        print("=" * 50)

        run_search(
            deadline=args.deadline or "06:00",
            max_experiments=args.max,
        )
    else:
        from bespoke.train.train_sft import run_sft_training

        print("BESPOKE Stage 3: SFT Training")
        print("=" * 50)

        adapter_dir = run_sft_training(
            adapter_name=args.adapter_name,
            domain_filter=args.domain,
        )

        print(f"\nAdapter saved to: {adapter_dir}")


def cmd_eval(args):
    """Score adapter against benchmark, compare with previous, keep or revert."""
    from bespoke.train.evaluate import (
        run_evaluation, compare_scorecards,
        save_scorecard, log_experiment,
    )
    from bespoke.serve.server import start_server
    from bespoke.config import config

    adapter_name = args.adapter_name

    print("BESPOKE Evaluation")
    print("=" * 50)

    proc = None
    try:
        print("Starting temporary server for evaluation...")
        proc = start_server(port=config.base_model.llama_server_port)

        scorecard = run_evaluation(
            adapter_name=adapter_name,
            server_port=config.base_model.llama_server_port,
        )

        if "error" in scorecard:
            print(f"Evaluation failed: {scorecard['error']}")
            sys.exit(1)

        # Compare with previous scorecard
        previous = find_previous_scorecard(adapter_name)
        comparison = None
        if previous:
            comparison = compare_scorecards(scorecard, previous)
            print(f"\nDecision: {comparison['decision'].upper()}")
            print(f"Reasoning: {comparison['reasoning']}")
        else:
            print("\nNo previous scorecard — this is the baseline.")

        save_scorecard(scorecard, comparison)
        exp_id = log_experiment(scorecard, comparison)
        print(f"Experiment logged as #{exp_id}")

    finally:
        if proc:
            proc.terminate()
            proc.wait()
            print("Temporary server stopped.")


def cmd_serve(args):
    """Start llama.cpp server with optional adapter."""
    from bespoke.serve.server import start_server

    adapter_path = None
    if args.adapter:
        adapter_path = Path(args.adapter)
    else:
        latest = find_latest_adapter()
        if latest:
            gguf_files = list(latest.glob("*.gguf"))
            if gguf_files:
                adapter_path = gguf_files[0]
                print(f"Using latest adapter: {adapter_path}")
            else:
                print(f"Latest adapter at {latest} has no GGUF file. Serving naked base model.")
        else:
            print("No adapters found. Serving naked base model.")

    proc = start_server(adapter_path=adapter_path, port=args.port)

    print(f"\nServer running. API endpoint: http://localhost:{args.port}/v1")
    print("Press Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nServer stopped.")


def cmd_run(args):
    """Full pipeline: capture → interview (if needed) → extract → train → eval."""
    from bespoke.capture.pipeline import run_capture
    from bespoke.teach.stage2a import run_stage_2a
    from bespoke.train.train_sft import run_sft_training
    from bespoke.train.evaluate import (
        run_evaluation, compare_scorecards,
        save_scorecard, log_experiment,
    )
    from bespoke.serve.server import start_server
    from bespoke.db.init import get_connection
    from bespoke.config import config

    # Step 1: Capture
    print("\n" + "=" * 50)
    print("STEP 1: Capture")
    print("=" * 50)
    capture_stats = run_capture()
    print(f"Captured {capture_stats['interactions_captured']} new interactions")

    # Step 2: Benchmark (if not initialized and not skipped)
    if not getattr(args, "skip_interview", False) and not benchmark_exists():
        print("\n" + "=" * 50)
        print("STEP 2: Benchmark Interview")
        print("=" * 50)
        print("No benchmark found. Starting benchmark interview...")
        print("The interview takes 10-15 minutes and only runs once.\n")
        from bespoke.benchmark.interview import run_interview
        from bespoke.benchmark.reconcile import write_interview_benchmark
        interview_result = run_interview()
        write_interview_benchmark(interview_result)

    # Step 3: Extract (benchmark-informed)
    print("\n" + "=" * 50)
    print("STEP 3: Extract")
    print("=" * 50)

    conn = get_connection()
    pairs_before = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
    conn.close()

    extract_stats = run_stage_2a()

    conn = get_connection()
    pairs_after = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
    sft_count = conn.execute(
        "SELECT COUNT(*) FROM training_pairs WHERE pair_type = 'sft'"
    ).fetchone()[0]
    conn.close()

    new_examples = pairs_after - pairs_before

    # Gate: minimum SFT pairs for training
    min_sft = 50
    if sft_count < min_sft:
        print(f"\nOnly {sft_count} SFT training pairs total. Need at least {min_sft}. "
              "Skipping train + eval.")
        return

    min_required = config.pipeline.min_new_examples_for_training
    if new_examples < min_required:
        print(f"\nSkipping train + eval: only {new_examples} new training pairs "
              f"(minimum: {min_required})")
        return

    print(f"{new_examples} new training pairs produced")

    # Step 4: Train
    print("\n" + "=" * 50)
    print("STEP 4: Train")
    print("=" * 50)
    if getattr(args, "search_deadline", None):
        from bespoke.train.train_sft import run_search
        run_search(deadline=args.search_deadline)
    else:
        adapter_name = "general-v1"
        adapter_dir = run_sft_training(adapter_name=adapter_name)
        print(f"Adapter saved to: {adapter_dir}")

    if not getattr(args, "search_deadline", None):
        # Step 5: Eval (only for single-pass training)
        print("\n" + "=" * 50)
        print("STEP 5: Evaluate")
        print("=" * 50)

        proc = None
        try:
            print("Starting temporary server for evaluation...")
            proc = start_server(port=config.base_model.llama_server_port)

            scorecard = run_evaluation(
                adapter_name=adapter_name,
                server_port=config.base_model.llama_server_port,
            )

            if "error" in scorecard:
                print(f"Evaluation failed: {scorecard['error']}")
                return

            previous = find_previous_scorecard(adapter_name)
            comparison = None
            if previous:
                comparison = compare_scorecards(scorecard, previous)
                print(f"\nDecision: {comparison['decision'].upper()}")
                print(f"Reasoning: {comparison['reasoning']}")

            save_scorecard(scorecard, comparison)
            exp_id = log_experiment(scorecard, comparison)
            print(f"Experiment logged as #{exp_id}")

        finally:
            if proc:
                proc.terminate()
                proc.wait()

    # Step 6: Trajectory highlights
    from bespoke.trajectory.report import generate_highlights
    generate_highlights()

    print("=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


def cmd_trajectory(args):
    """Stage 2c: Trajectory analysis — read-only queries over the warehouse."""
    from bespoke.trajectory.report import generate_report

    generate_report(months=args.months, domain=args.domain)


def cmd_benchmark_interview(args):
    """Interactive benchmark interview."""
    import json
    from bespoke.benchmark.prescan import generate_prescan_summary
    from bespoke.benchmark.interview import run_interview
    from bespoke.benchmark.reconcile import write_interview_benchmark

    prescan = generate_prescan_summary()
    print(f"\nPrescan: {prescan['total_interactions']} interactions across "
          f"{prescan['total_sessions']} sessions")

    result = run_interview(prescan=prescan)
    if result:
        write_interview_benchmark(result)
        print(json.dumps(result, indent=2))


# ── Entrypoint ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="bespoke",
        description="Compound frontier model interactions into bespoke models.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # capture
    p_capture = subparsers.add_parser("capture", help="Ingest new interactions")
    p_capture.add_argument("--jsonl", type=str, help="Ingest from JSONL export")
    p_capture.add_argument("--session-dir", type=str, help="Override session directory")
    p_capture.add_argument("--since", type=str, help="Only process after this ISO datetime")
    p_capture.add_argument("--backfill", action="store_true", help="Force full re-process")
    p_capture.add_argument("--web", action="store_true",
                           help="Capture claude.ai conversations")
    p_capture.add_argument("--all", action="store_true",
                           help="Capture all sources (Claude Code + claude.ai)")
    p_capture.set_defaults(func=cmd_capture)

    # extract
    p_extract = subparsers.add_parser("extract", help="Classify and curate training data")
    p_extract.add_argument("--reset", action="store_true",
                            help="Wipe all Stage 2a outputs before running (one-time use before backfill)")
    p_extract.set_defaults(func=cmd_extract)

    # train
    p_train = subparsers.add_parser("train", help="Fine-tune adapter")
    p_train.add_argument("--domain", type=str, help="Filter training data by domain")
    p_train.add_argument("--adapter-name", type=str, default="general-v1",
                         help="Name for the adapter")
    p_train.add_argument("--deadline", type=str,
                         help="Loop experiments until HH:MM (e.g. 06:00)")
    p_train.add_argument("--max", type=int,
                         help="Max number of experiments to run")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Score adapter, keep or revert")
    p_eval.add_argument("--adapter-name", type=str, default="general-v1",
                        help="Adapter to evaluate")
    p_eval.set_defaults(func=cmd_eval)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start local model server")
    p_serve.add_argument("--adapter", type=str, help="Specific adapter path")
    p_serve.add_argument("--port", type=int, default=8080, help="Server port")
    p_serve.set_defaults(func=cmd_serve)

    # trajectory
    p_traj = subparsers.add_parser("trajectory",
                                   help="Trajectory analysis over the warehouse")
    p_traj.add_argument("--months", type=int, help="Limit to last N months")
    p_traj.add_argument("--domain", type=str, help="Filter to one domain")
    p_traj.set_defaults(func=cmd_trajectory)

    # run
    p_run = subparsers.add_parser("run",
                                  help="Full pipeline: capture → interview → extract → train → eval")
    p_run.add_argument("--skip-interview", action="store_true",
                       help="Skip benchmark interview even if no benchmark exists (uses generic extraction)")
    p_run.add_argument("--search-deadline", type=str,
                       help="Run search loop during train step (e.g. 06:00)")
    p_run.set_defaults(func=cmd_run)

    # benchmark (subparser group)
    p_benchmark = subparsers.add_parser("benchmark", help="Benchmark management")
    benchmark_sub = p_benchmark.add_subparsers(dest="benchmark_command")

    p_interview = benchmark_sub.add_parser("interview",
                                           help="Define your quality standards")
    p_interview.set_defaults(func=cmd_benchmark_interview)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "benchmark" and not getattr(args, "benchmark_command", None):
        p_benchmark.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
