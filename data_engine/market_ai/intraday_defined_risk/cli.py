from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import run_backtest
from .backtest_dataset_tools import (
    build_dense_backtest_dataset,
    compare_benchmark_results,
    inspect_backtest_dataset,
)
from .collector import capture_decision_time_snapshot, run_daily_chain_schedule, run_decision_time_schedule
from .dataset import (
    assess_structured_dataset,
    build_structured_dataset_from_rolling,
    enrich_structured_chain_deltas,
    refresh_structured_training_dataset_from_rolling,
    scaffold_dataset_layout,
)
from .learning import LearningStore
from .monitor import run_live
from .ops_runtime import (
    RuntimeMode,
    build_operator_status_report,
    build_paper_live_report,
    build_shadow_live_report,
    flatten_all_emergency,
    load_runtime_config,
    set_runtime_mode,
)
from .pipeline import refresh_training_data, run_training_data_pipeline
from .research import (
    DEFAULT_BACKTEST_START,
    append_research_backtest_dataset_from_rolling,
    build_monitor_times,
    build_research_backtest_dataset,
    build_research_backtest_dataset_from_rolling,
    optimize_backtest_parameters,
)


def _load_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if config_path.suffix == ".json":
        return json.loads(config_path.read_text())
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard only
        raise SystemExit("PyYAML is required for YAML config files.") from exc
    return yaml.safe_load(config_path.read_text()) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="intraday-defined-risk-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live_parser = subparsers.add_parser("run_live")
    live_parser.add_argument("--config", required=True)

    backtest_parser = subparsers.add_parser("run_backtest")
    backtest_parser.add_argument("--data_path", required=True)
    backtest_parser.add_argument("--start")
    backtest_parser.add_argument("--end")
    backtest_parser.add_argument("--output_json")
    backtest_parser.add_argument("--entry_times", default="AUTO")
    backtest_parser.add_argument("--round_trip_cost_rupees_per_lot", type=float, default=0.0)
    backtest_parser.add_argument("--fail_on_sparse_dataset", action="store_true")

    inspect_dataset_parser = subparsers.add_parser("inspect_dataset")
    inspect_dataset_parser.add_argument("--data_path", required=True)
    inspect_dataset_parser.add_argument("--start")
    inspect_dataset_parser.add_argument("--end")

    dense_dataset_parser = subparsers.add_parser("build_dataset")
    dense_dataset_parser.add_argument("--rolling_root", required=True)
    dense_dataset_parser.add_argument("--start", required=True)
    dense_dataset_parser.add_argument("--end", required=True)
    dense_dataset_parser.add_argument("--interval", default="5m")
    dense_dataset_parser.add_argument("--output_path", required=True)

    compare_parser = subparsers.add_parser("compare_results")
    compare_parser.add_argument("--old_json", required=True)
    compare_parser.add_argument("--new_json", required=True)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--window", type=int, default=60)
    calibrate_parser.add_argument("--db_path", default="/tmp/intraday_defined_risk_learning.sqlite3")

    dataset_parser = subparsers.add_parser("validate_dataset")
    dataset_parser.add_argument("--data_root", required=True)

    scaffold_parser = subparsers.add_parser("scaffold_dataset")
    scaffold_parser.add_argument("--data_root", required=True)

    build_parser = subparsers.add_parser("build_structured_dataset")
    build_parser.add_argument("--rolling_root", required=True)
    build_parser.add_argument("--data_root", required=True)
    build_parser.add_argument("--trade_blotter")
    build_parser.add_argument("--scrip_master")
    build_parser.add_argument("--option_chain_db")
    build_parser.add_argument("--from_date")
    build_parser.add_argument("--to_date")

    snapshot_parser = subparsers.add_parser("collect_chain_snapshot")
    snapshot_parser.add_argument("--data_root", required=True)
    snapshot_parser.add_argument("--underlying_id", type=int, default=13)
    snapshot_parser.add_argument("--underlying_seg", default="IDX_I")
    snapshot_parser.add_argument("--expiry")
    snapshot_parser.add_argument("--decision_time")
    snapshot_parser.add_argument("--raw_snapshot_root")
    snapshot_parser.add_argument("--creds_path")

    schedule_parser = subparsers.add_parser("run_chain_schedule")
    schedule_parser.add_argument("--data_root", required=True)
    schedule_parser.add_argument("--underlying_id", type=int, default=13)
    schedule_parser.add_argument("--underlying_seg", default="IDX_I")
    schedule_parser.add_argument("--expiry")
    schedule_parser.add_argument("--times", default="09:30,10:00,13:00")
    schedule_parser.add_argument("--poll_seconds", type=float, default=15.0)
    schedule_parser.add_argument("--raw_snapshot_root")
    schedule_parser.add_argument("--creds_path")

    daily_schedule_parser = subparsers.add_parser("run_daily_chain_schedule")
    daily_schedule_parser.add_argument("--data_root", required=True)
    daily_schedule_parser.add_argument("--underlying_id", type=int, default=13)
    daily_schedule_parser.add_argument("--underlying_seg", default="IDX_I")
    daily_schedule_parser.add_argument("--expiry")
    daily_schedule_parser.add_argument("--times", default="09:30,10:00,13:00")
    daily_schedule_parser.add_argument("--poll_seconds", type=float, default=15.0)
    daily_schedule_parser.add_argument("--raw_snapshot_root")
    daily_schedule_parser.add_argument("--creds_path")
    daily_schedule_parser.add_argument("--max_reports", type=int)
    daily_schedule_parser.add_argument("--max_days", type=int)

    enrich_parser = subparsers.add_parser("enrich_chain_deltas")
    enrich_parser.add_argument("--data_root", required=True)
    enrich_parser.add_argument("--risk_free_rate", type=float, default=0.06)

    research_parser = subparsers.add_parser("build_research_dataset")
    research_parser.add_argument("--structured_root", required=True)
    research_parser.add_argument("--output_root", required=True)
    research_parser.add_argument("--from_date", default=DEFAULT_BACKTEST_START)
    research_parser.add_argument("--to_date")
    research_parser.add_argument("--risk_free_rate", type=float, default=0.06)

    rolling_research_parser = subparsers.add_parser("build_research_dataset_from_rolling")
    rolling_research_parser.add_argument("--rolling_root", required=True)
    rolling_research_parser.add_argument("--output_root", required=True)
    rolling_research_parser.add_argument("--from_date", default=DEFAULT_BACKTEST_START)
    rolling_research_parser.add_argument("--to_date")
    rolling_research_parser.add_argument("--risk_free_rate", type=float, default=0.06)
    rolling_research_parser.add_argument("--monitor_start", default="09:30")
    rolling_research_parser.add_argument("--monitor_end", default="15:00")
    rolling_research_parser.add_argument("--monitor_step_minutes", type=int, default=5)

    append_research_parser = subparsers.add_parser("append_research_dataset_from_rolling")
    append_research_parser.add_argument("--rolling_root", required=True)
    append_research_parser.add_argument("--output_root", required=True)
    append_research_parser.add_argument("--from_date", default="2025-01-01")
    append_research_parser.add_argument("--to_date")
    append_research_parser.add_argument("--risk_free_rate", type=float, default=0.06)
    append_research_parser.add_argument("--monitor_start", default="09:30")
    append_research_parser.add_argument("--monitor_end", default="15:00")
    append_research_parser.add_argument("--monitor_step_minutes", type=int, default=5)

    structured_refresh_parser = subparsers.add_parser("refresh_structured_training_dataset")
    structured_refresh_parser.add_argument("--rolling_root", required=True)
    structured_refresh_parser.add_argument("--data_root", required=True)
    structured_refresh_parser.add_argument("--trade_blotter")
    structured_refresh_parser.add_argument("--scrip_master")
    structured_refresh_parser.add_argument("--from_date", default="2025-01-01")
    structured_refresh_parser.add_argument("--to_date")

    refresh_pipeline_parser = subparsers.add_parser("refresh_training_data")
    refresh_pipeline_parser.add_argument("--rolling_root", required=True)
    refresh_pipeline_parser.add_argument("--structured_root", required=True)
    refresh_pipeline_parser.add_argument("--research_root", required=True)
    refresh_pipeline_parser.add_argument("--trade_blotter")
    refresh_pipeline_parser.add_argument("--scrip_master")
    refresh_pipeline_parser.add_argument("--option_chain_db")
    refresh_pipeline_parser.add_argument("--raw_snapshot_root")
    refresh_pipeline_parser.add_argument("--status_path")
    refresh_pipeline_parser.add_argument("--start_date", default="2025-01-01")
    refresh_pipeline_parser.add_argument("--to_date")
    refresh_pipeline_parser.add_argument("--underlying", default="NIFTY")
    refresh_pipeline_parser.add_argument("--security_id", type=int, default=13)
    refresh_pipeline_parser.add_argument("--segment", default="NSE_FNO")
    refresh_pipeline_parser.add_argument("--interval", default="1")
    refresh_pipeline_parser.add_argument("--monitor_start", default="09:30")
    refresh_pipeline_parser.add_argument("--monitor_end", default="15:00")
    refresh_pipeline_parser.add_argument("--monitor_step_minutes", type=int, default=5)
    refresh_pipeline_parser.add_argument("--creds_path")

    pipeline_parser = subparsers.add_parser("run_training_data_pipeline")
    pipeline_parser.add_argument("--rolling_root", required=True)
    pipeline_parser.add_argument("--structured_root", required=True)
    pipeline_parser.add_argument("--research_root", required=True)
    pipeline_parser.add_argument("--trade_blotter")
    pipeline_parser.add_argument("--scrip_master")
    pipeline_parser.add_argument("--option_chain_db")
    pipeline_parser.add_argument("--raw_snapshot_root")
    pipeline_parser.add_argument("--status_path")
    pipeline_parser.add_argument("--start_date", default="2025-01-01")
    pipeline_parser.add_argument("--to_date")
    pipeline_parser.add_argument("--underlying", default="NIFTY")
    pipeline_parser.add_argument("--security_id", type=int, default=13)
    pipeline_parser.add_argument("--underlying_seg", default="IDX_I")
    pipeline_parser.add_argument("--rolling_segment", default="NSE_FNO")
    pipeline_parser.add_argument("--interval", default="1")
    pipeline_parser.add_argument("--monitor_start", default="09:30")
    pipeline_parser.add_argument("--monitor_end", default="15:00")
    pipeline_parser.add_argument("--monitor_step_minutes", type=int, default=5)
    pipeline_parser.add_argument("--refresh_after", default="15:20")
    pipeline_parser.add_argument("--poll_seconds", type=float, default=30.0)
    pipeline_parser.add_argument("--creds_path")
    pipeline_parser.add_argument("--max_days", type=int)

    optimize_parser = subparsers.add_parser("optimize_backtest")
    optimize_parser.add_argument("--data_path", required=True)
    optimize_parser.add_argument("--start")
    optimize_parser.add_argument("--end")
    optimize_parser.add_argument("--db_dir", default="/tmp/intraday_defined_risk_opt")

    ops_status_parser = subparsers.add_parser("ops_status")
    ops_status_parser.add_argument("--no_write", action="store_true")

    mode_parser = subparsers.add_parser("set_runtime_mode")
    mode_parser.add_argument("--mode", choices=[mode.value for mode in RuntimeMode], required=True)
    mode_parser.add_argument("--live_arm", action="store_true")
    mode_parser.add_argument("--source", default="cli")

    flatten_parser = subparsers.add_parser("flatten_all")
    flatten_parser.add_argument("--source", default="cli")

    reports_parser = subparsers.add_parser("write_operational_reports")
    reports_parser.add_argument("--no_write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run_live":
        run_live(_load_config(args.config))
        return 0
    if args.command == "run_backtest":
        result = run_backtest(
            args.data_path,
            start=args.start,
            end=args.end,
            entry_times=None if args.entry_times.upper() == "AUTO" else {piece.strip() for piece in args.entry_times.split(",") if piece.strip()},
            round_trip_cost_rupees_per_lot=args.round_trip_cost_rupees_per_lot,
            fail_on_sparse_dataset=args.fail_on_sparse_dataset,
        )
        summary_json = json.dumps(result["summary"], indent=2, sort_keys=True)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(summary_json + "\n")
        print(summary_json)
        return 0
    if args.command == "inspect_dataset":
        report = inspect_backtest_dataset(args.data_path, start=args.start, end=args.end)
        print(report.to_json())
        return 0
    if args.command == "build_dataset":
        report = build_dense_backtest_dataset(
            args.rolling_root,
            args.output_path,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
        )
        print(report.to_json())
        return 0
    if args.command == "compare_results":
        print(json.dumps(compare_benchmark_results(args.old_json, args.new_json), indent=2, sort_keys=True))
        return 0
    if args.command == "calibrate":
        store = LearningStore(args.db_path)
        result = store.calibrate(window=args.window)
        print(json.dumps({
            "status": result.status,
            "reasons": result.reasons,
            "current_parameters": result.current_parameters.__dict__,
            "candidate_parameters": result.candidate_parameters.__dict__ if result.candidate_parameters else None,
            "objective_before": result.objective_before,
            "objective_after": result.objective_after,
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "validate_dataset":
        report = assess_structured_dataset(args.data_root)
        print(report.to_json())
        return 0
    if args.command == "scaffold_dataset":
        paths = scaffold_dataset_layout(args.data_root)
        print(json.dumps({key: str(value) for key, value in paths.__dict__.items()}, indent=2, sort_keys=True))
        return 0
    if args.command == "build_structured_dataset":
        report = build_structured_dataset_from_rolling(
            args.rolling_root,
            args.data_root,
            trade_blotter_path=args.trade_blotter,
            scrip_master_path=args.scrip_master,
            option_chain_db_path=args.option_chain_db,
            from_date=args.from_date,
            to_date=args.to_date,
        )
        print(report.to_json())
        return 0
    if args.command == "collect_chain_snapshot":
        report = capture_decision_time_snapshot(
            args.data_root,
            underlying_id=args.underlying_id,
            underlying_seg=args.underlying_seg,
            expiry=args.expiry,
            decision_time=args.decision_time,
            raw_snapshot_root=args.raw_snapshot_root,
            creds_path=args.creds_path,
        )
        print(report.to_json())
        return 0
    if args.command == "run_chain_schedule":
        reports = run_decision_time_schedule(
            args.data_root,
            decision_times=[piece.strip() for piece in args.times.split(",") if piece.strip()],
            underlying_id=args.underlying_id,
            underlying_seg=args.underlying_seg,
            expiry=args.expiry,
            raw_snapshot_root=args.raw_snapshot_root,
            creds_path=args.creds_path,
            poll_seconds=args.poll_seconds,
        )
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
        return 0
    if args.command == "run_daily_chain_schedule":
        reports = run_daily_chain_schedule(
            args.data_root,
            decision_times=[piece.strip() for piece in args.times.split(",") if piece.strip()],
            underlying_id=args.underlying_id,
            underlying_seg=args.underlying_seg,
            expiry=args.expiry,
            raw_snapshot_root=args.raw_snapshot_root,
            creds_path=args.creds_path,
            poll_seconds=args.poll_seconds,
            max_reports=args.max_reports,
            max_days=args.max_days,
        )
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
        return 0
    if args.command == "enrich_chain_deltas":
        report = enrich_structured_chain_deltas(
            args.data_root,
            risk_free_rate=args.risk_free_rate,
        )
        print(report.to_json())
        return 0
    if args.command == "build_research_dataset":
        report = build_research_backtest_dataset(
            args.structured_root,
            args.output_root,
            from_date=args.from_date,
            to_date=args.to_date,
            risk_free_rate=args.risk_free_rate,
        )
        print(report.to_json())
        return 0
    if args.command == "build_research_dataset_from_rolling":
        report = build_research_backtest_dataset_from_rolling(
            args.rolling_root,
            args.output_root,
            from_date=args.from_date,
            to_date=args.to_date,
            risk_free_rate=args.risk_free_rate,
            monitor_times=build_monitor_times(
                start=args.monitor_start,
                end=args.monitor_end,
                step_minutes=args.monitor_step_minutes,
            ),
        )
        print(report.to_json())
        return 0
    if args.command == "append_research_dataset_from_rolling":
        report = append_research_backtest_dataset_from_rolling(
            args.rolling_root,
            args.output_root,
            from_date=args.from_date,
            to_date=args.to_date,
            risk_free_rate=args.risk_free_rate,
            monitor_times=build_monitor_times(
                start=args.monitor_start,
                end=args.monitor_end,
                step_minutes=args.monitor_step_minutes,
            ),
        )
        print(report.to_json())
        return 0
    if args.command == "refresh_structured_training_dataset":
        report = refresh_structured_training_dataset_from_rolling(
            args.rolling_root,
            args.data_root,
            trade_blotter_path=args.trade_blotter,
            scrip_master_path=args.scrip_master,
            from_date=args.from_date,
            to_date=args.to_date,
        )
        print(report.to_json())
        return 0
    if args.command == "refresh_training_data":
        report = refresh_training_data(
            rolling_root=args.rolling_root,
            structured_root=args.structured_root,
            research_root=args.research_root,
            trade_blotter_path=args.trade_blotter,
            scrip_master_path=args.scrip_master,
            option_chain_db_path=args.option_chain_db,
            raw_snapshot_root=args.raw_snapshot_root,
            status_path=args.status_path,
            start_date=args.start_date,
            to_date=args.to_date,
            underlying=args.underlying,
            security_id=args.security_id,
            segment=args.segment,
            interval=args.interval,
            monitor_times=build_monitor_times(
                start=args.monitor_start,
                end=args.monitor_end,
                step_minutes=args.monitor_step_minutes,
            ),
            creds_path=args.creds_path,
        )
        print(report.to_json())
        return 0
    if args.command == "run_training_data_pipeline":
        reports = run_training_data_pipeline(
            rolling_root=args.rolling_root,
            structured_root=args.structured_root,
            research_root=args.research_root,
            trade_blotter_path=args.trade_blotter,
            scrip_master_path=args.scrip_master,
            option_chain_db_path=args.option_chain_db,
            raw_snapshot_root=args.raw_snapshot_root,
            status_path=args.status_path,
            start_date=args.start_date,
            to_date=args.to_date,
            underlying=args.underlying,
            security_id=args.security_id,
            underlying_seg=args.underlying_seg,
            rolling_segment=args.rolling_segment,
            interval=args.interval,
            monitor_start=args.monitor_start,
            monitor_end=args.monitor_end,
            monitor_step_minutes=args.monitor_step_minutes,
            refresh_after=args.refresh_after,
            poll_seconds=args.poll_seconds,
            creds_path=args.creds_path,
            max_days=args.max_days,
        )
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
        return 0
    if args.command == "optimize_backtest":
        result = optimize_backtest_parameters(
            args.data_path,
            start=args.start,
            end=args.end,
            db_dir=args.db_dir,
        )
        print(result.to_json())
        return 0
    if args.command == "ops_status":
        report = build_operator_status_report(write=not args.no_write)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "set_runtime_mode":
        config = set_runtime_mode(args.mode, live_arm=args.live_arm, source=args.source)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "flatten_all":
        result = flatten_all_emergency(source=args.source)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "write_operational_reports":
        paths = {
            "shadow_live_report": build_shadow_live_report(write=not args.no_write),
            "paper_live_report": build_paper_live_report(write=not args.no_write),
            "operator_status_report": build_operator_status_report(write=not args.no_write),
            "runtime_config": load_runtime_config().to_dict(),
        }
        print(json.dumps(paths, indent=2, sort_keys=True, default=str))
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
