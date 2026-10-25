# perpsfm task runner. `just <task>` (https://github.com/casey/just)

default:
    @just --list

# format + lint + test
check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

# pull the full Velocity archive into data/ (incremental by default)
backfill:
    cargo run --release --features metal --bin perpsfm -- backfill

# refresh: fetch only newer-than-existing data
backfill-incremental:
    cargo run --release --features metal --bin perpsfm -- backfill --incremental

# start the live DLOB collector (runs until Ctrl-C)
collect:
    cargo run --release --features metal --bin perpsfm -- collect

build-features:
    cargo run --release --features metal --bin perpsfm -- features

train:
    cargo run --release --features metal --bin perpsfm -- train

evaluate:
    cargo run --release --features metal --bin perpsfm -- evaluate

backtest:
    cargo run --release --features metal --bin perpsfm -- backtest
