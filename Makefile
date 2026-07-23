.PHONY: check sync rust autoresearch locks

sync:
	sfw cargo fetch --locked
	UV_CACHE_DIR=.uv-cache sfw uv sync --project autoresearch --frozen --group dev

locks:
	python3 scripts/check_action_pins.py
	python3 scripts/check_engine_lock.py

rust:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
	cargo test --workspace --all-targets --locked --offline

autoresearch:
	UV_CACHE_DIR=.uv-cache uv run --project autoresearch --frozen --no-sync ruff format --check autoresearch
	UV_CACHE_DIR=.uv-cache uv run --project autoresearch --frozen --no-sync ruff check autoresearch
	UV_CACHE_DIR=.uv-cache uv run --project autoresearch --frozen --no-sync pytest -c autoresearch/pyproject.toml --rootdir=autoresearch autoresearch

check: locks rust autoresearch
