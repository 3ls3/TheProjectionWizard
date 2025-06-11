#!/usr/bin/env python3
"""
Pipeline Debug Log Viewer for The Projection Wizard
Utility for viewing, filtering, and analyzing local development logs.

Usage:
    python scripts/python/debug_logs.py --list                    # List all runs with logs
    python scripts/python/debug_logs.py --run abc123             # View all logs for run
    python scripts/python/debug_logs.py --run abc123 --stage automl  # View specific stage
    python scripts/python/debug_logs.py --tail abc123            # Tail logs in real-time
    python scripts/python/debug_logs.py --json abc123 --stage validation  # View JSON logs
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict
import time
from datetime import datetime


def get_project_root() -> Path:
    """Get the project root directory."""
    script_path = Path(__file__).resolve()
    # Go up from scripts/python/ to project root
    return script_path.parent.parent.parent


def get_runs_with_logs() -> List[str]:
    """Get list of run IDs that have log directories."""
    project_root = get_project_root()
    runs_dir = project_root / "data" / "runs"
    
    if not runs_dir.exists():
        return []
    
    runs_with_logs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "logs").exists():
            runs_with_logs.append(run_dir.name)
    
    return sorted(runs_with_logs, reverse=True)  # Most recent first


def get_log_files_for_run(run_id: str) -> Dict[str, List[Path]]:
    """Get categorized log files for a specific run."""
    project_root = get_project_root()
    logs_dir = project_root / "data" / "runs" / run_id / "logs"
    
    if not logs_dir.exists():
        return {"human": [], "json": []}
    
    human_logs = []
    json_logs = []
    
    for log_file in logs_dir.iterdir():
        if log_file.is_file():
            if log_file.suffix == ".log":
                human_logs.append(log_file)
            elif log_file.suffix == ".jsonl":
                json_logs.append(log_file)
    
    return {
        "human": sorted(human_logs),
        "json": sorted(json_logs)
    }


def list_runs():
    """List all runs with available logs."""
    runs = get_runs_with_logs()
    
    if not runs:
        print("🤷 No runs with local logs found.")
        print("💡 Make sure LOCAL_DEV_LOGGING=true when running the API")
        return
    
    print(f"📂 Found {len(runs)} runs with local logs:")
    print()
    
    for run_id in runs:
        logs = get_log_files_for_run(run_id)
        human_count = len(logs["human"])
        json_count = len(logs["json"])
        
        print(f"🔮 {run_id}")
        print(f"   📝 {human_count} human-readable log files")
        print(f"   📊 {json_count} structured JSON log files")
        
        # Show file modification times
        project_root = get_project_root()
        logs_dir = project_root / "data" / "runs" / run_id / "logs"
        if logs_dir.exists():
            newest_file = max(logs_dir.iterdir(), key=lambda f: f.stat().st_mtime)
            mod_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
            print(f"   ⏰ Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def view_logs(run_id: str, stage: Optional[str] = None, json_format: bool = False):
    """View logs for a specific run and optionally a specific stage."""
    logs = get_log_files_for_run(run_id)
    
    if not logs["human"] and not logs["json"]:
        print(f"❌ No logs found for run {run_id}")
        return
    
    # Determine which files to show
    target_files = []
    
    if json_format:
        if stage:
            # Look for specific stage JSON file
            stage_files = [f for f in logs["json"] if stage in f.name]
            if stage_files:
                target_files = stage_files
            else:
                print(f"❌ No JSON logs found for stage '{stage}' in run {run_id}")
                return
        else:
            target_files = logs["json"]
    else:
        if stage:
            # Look for specific stage log file
            stage_files = [f for f in logs["human"] if stage in f.name]
            if stage_files:
                target_files = stage_files
            else:
                print(f"❌ No human-readable logs found for stage '{stage}' in run {run_id}")
                return
        else:
            target_files = logs["human"]
    
    # Display the logs
    for log_file in target_files:
        print(f"📄 {log_file.name}")
        print("=" * 80)
        
        try:
            if json_format:
                # Pretty-print JSON logs
                with open(log_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                log_data = json.loads(line)
                                print(f"Line {line_num:3d}: {json.dumps(log_data, indent=2)}")
                            except json.JSONDecodeError:
                                print(f"Line {line_num:3d}: {line}")
                            print()
            else:
                # Show human-readable logs as-is
                with open(log_file, 'r') as f:
                    content = f.read()
                    print(content)
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
        
        print()


def tail_logs(run_id: str, stage: Optional[str] = None):
    """Tail logs in real-time using system tail command."""
    logs = get_log_files_for_run(run_id)
    
    if not logs["human"]:
        print(f"❌ No logs found for run {run_id}")
        return
    
    # Determine which files to tail
    if stage:
        stage_files = [f for f in logs["human"] if stage in f.name]
        if not stage_files:
            print(f"❌ No logs found for stage '{stage}' in run {run_id}")
            return
        target_files = stage_files
    else:
        target_files = logs["human"]
    
    # Use system tail command for real-time following
    print(f"🔄 Tailing logs for run {run_id}" + (f" stage {stage}" if stage else ""))
    print("Press Ctrl+C to stop")
    print()
    
    try:
        if len(target_files) == 1:
            # Single file - use simple tail
            subprocess.run(["tail", "-f", str(target_files[0])])
        else:
            # Multiple files - use tail with file headers
            file_paths = [str(f) for f in target_files]
            subprocess.run(["tail", "-f"] + file_paths)
    except KeyboardInterrupt:
        print("\n🛑 Stopped tailing logs")
    except FileNotFoundError:
        print("❌ 'tail' command not found. Falling back to Python implementation.")
        _python_tail(target_files)


def _python_tail(files: List[Path]):
    """Python implementation of tail -f for systems without tail command."""
    file_handles = {}
    file_positions = {}
    
    # Open all files and remember positions
    for file_path in files:
        try:
            fh = open(file_path, 'r')
            file_handles[file_path] = fh
            fh.seek(0, 2)  # Seek to end
            file_positions[file_path] = fh.tell()
        except Exception as e:
            print(f"❌ Error opening {file_path}: {e}")
    
    try:
        while True:
            for file_path, fh in file_handles.items():
                fh.seek(file_positions[file_path])
                new_lines = fh.readlines()
                if new_lines:
                    if len(file_handles) > 1:
                        print(f"\n==> {file_path.name} <==")
                    for line in new_lines:
                        print(line.rstrip())
                    file_positions[file_path] = fh.tell()
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
    except KeyboardInterrupt:
        print("\n🛑 Stopped tailing logs")
    finally:
        for fh in file_handles.values():
            fh.close()


def clean_logs(run_id: Optional[str] = None):
    """Clean up old log files."""
    if run_id:
        # Clean specific run
        project_root = get_project_root()
        logs_dir = project_root / "data" / "runs" / run_id / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*"))
            for log_file in log_files:
                log_file.unlink()
            logs_dir.rmdir()
            print(f"🗑️ Cleaned logs for run {run_id}")
        else:
            print(f"❌ No logs found for run {run_id}")
    else:
        # Clean all runs
        runs = get_runs_with_logs()
        total_cleaned = 0
        for run_id in runs:
            project_root = get_project_root()
            logs_dir = project_root / "data" / "runs" / run_id / "logs"
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*"))
                for log_file in log_files:
                    log_file.unlink()
                logs_dir.rmdir()
                total_cleaned += 1
        print(f"🗑️ Cleaned logs for {total_cleaned} runs")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Debug log viewer for The Projection Wizard pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           List all runs with logs
  %(prog)s --run abc123                     View all logs for run abc123
  %(prog)s --run abc123 --stage automl      View automl stage logs for run abc123
  %(prog)s --run abc123 --json              View JSON logs for run abc123
  %(prog)s --tail abc123                    Tail logs for run abc123 in real-time
  %(prog)s --clean                          Clean all log files
  %(prog)s --clean --run abc123             Clean logs for specific run
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all runs with logs")
    parser.add_argument("--run", help="Run ID to view logs for")
    parser.add_argument("--stage", help="Specific pipeline stage to view")
    parser.add_argument("--json", action="store_true", help="View JSON structured logs instead of human-readable")
    parser.add_argument("--tail", help="Tail logs in real-time for specified run ID")
    parser.add_argument("--clean", action="store_true", help="Clean up log files")
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(get_project_root())
    
    if args.list:
        list_runs()
    elif args.tail:
        tail_logs(args.tail, args.stage)
    elif args.clean:
        clean_logs(args.run)
    elif args.run:
        view_logs(args.run, args.stage, args.json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 