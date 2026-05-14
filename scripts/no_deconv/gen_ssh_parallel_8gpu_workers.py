#!/usr/bin/env python3
"""Emit 8-GPU parallel SSH worker scripts + launchers from sequential batch_node*.sh files.

Each worker sets CUDA_VISIBLE_DEVICES=<gpu> and runs its assigned bash -lc jobs sequentially.
Round-robin assignment: job index i -> GPU (i % 8).

Skips samples whose NO_DECONV_MERGE already has all three *_3D_indexed.tif masks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def merge_done(merge: Path) -> bool:
    return all(
        (merge / ch / "segmentation_3D_masks" / f"{ch}_3D_indexed.tif").is_file()
        for ch in ("488nm_crop", "560nm_crop", "642nm_crop")
    )


def parse_merge_from_bash_lc(line: str) -> Path | None:
    m = re.search(r'export NO_DECONV_MERGE="([^"]+)"', line)
    if not m:
        return None
    return Path(m.group(1))


def extract_bash_lc_lines(batch_path: Path) -> list[str]:
    out: list[str] = []
    for line in batch_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("bash -lc '") and s.endswith("'"):
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch315", type=Path, required=True)
    ap.add_argument("--batch316", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ngpu", type=int, default=8)
    args = ap.parse_args()

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ngpu = args.ngpu

    def emit(node: str, batch: Path) -> None:
        lines = extract_bash_lc_lines(batch)
        kept: list[tuple[str, Path]] = []
        skipped = 0
        for bl in lines:
            merge = parse_merge_from_bash_lc(bl)
            if merge is None:
                raise SystemExit(f"Could not parse MERGE from line in {batch}")
            if merge_done(merge):
                skipped += 1
                continue
            kept.append((bl, merge))

        workers: list[list[str]] = [[] for _ in range(ngpu)]
        for i, (bl, _merge) in enumerate(kept):
            workers[i % ngpu].append(bl)

        for g in range(ngpu):
            wpath = out_dir / f"worker_{node}_gpu{g}.sh"
            log = out_dir / f"worker_{node}_gpu{g}.log"
            parts = [
                "#!/usr/bin/env bash",
                f"# Auto-generated: GPU {g} on {node}, {len(workers[g])} job(s).",
                "set +e",
                f"export CUDA_VISIBLE_DEVICES={g}",
                f'LOG="{log}"',
                f'echo "=== worker {node} gpu{g} start $(date -Is) ===" >>"$LOG"',
            ]
            for bl in workers[g]:
                parts.append(f'echo "=== RUN $(date -Is) gpu{g} ===" >>"$LOG"')
                parts.append(f'{bl} >>"$LOG" 2>&1')
                parts.append(f'echo "=== exit $? gpu{g} $(date -Is) ===" >>"$LOG"')
            parts.append(f'echo "=== worker {node} gpu{g} end $(date -Is) ===" >>"$LOG"')
            wpath.write_text("\n".join(parts) + "\n")
            wpath.chmod(0o755)

        launch = out_dir / f"launch_parallel_{node}.sh"
        launch_lines = [
            "#!/usr/bin/env bash",
            f"# Start {ngpu} GPU workers on {node} (shared FS).",
            "set -euo pipefail",
            f'DIR="{out_dir}"',
        ]
        for gg in range(ngpu):
            launch_lines.append(
                f'nohup bash "$DIR/worker_{node}_gpu{gg}.sh" </dev/null '
                f'>>"$DIR/nohup_launch_{node}_gpu{gg}.out" 2>&1 &'
            )
        launch_lines.extend(
            [
                "wait",
                f'echo "all workers finished $(date -Is)" >>"$DIR/launch_parallel_{node}.log"',
                "",
            ]
        )
        launch.write_text("\n".join(launch_lines))
        launch.chmod(0o755)
        meta = out_dir / f"meta_{node}.txt"
        meta.write_text(
            f"node={node}\n"
            f"source_batch={batch}\n"
            f"total_bash_lc_in_batch={len(lines)}\n"
            f"skipped_already_done={skipped}\n"
            f"jobs_to_run={len(kept)}\n"
            f"per_gpu_counts={[len(w) for w in workers]!r}\n"
        )

    emit("315", args.batch315)
    emit("316", args.batch316)
    print("Wrote workers + launch_parallel_315.sh / launch_parallel_316.sh ->", out_dir)


if __name__ == "__main__":
    main()
