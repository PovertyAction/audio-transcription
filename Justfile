# Set the shell to use
# set shell := ["nu", "-c"]
# Set shell for Windows

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Set path to virtual environment's python

venv_dir := ".venv"
python := venv_dir + if os_family() == "windows" { "/Scripts/python.exe" } else { "/bin/python3" }

# Display system information
system-info:
    @echo "CPU architecture: {{ arch() }}"
    @echo "Operating system type: {{ os_family() }}"
    @echo "Operating system: {{ os() }}"

# Clean venv
[linux]
clean:
    rm -rf .venv

# Clean venv
[macos]
clean:
    rm -rf .venv

# Clean venv
[windows]
clean:
    if (Test-Path ".venv") { Remove-Item ".venv" -Recurse -Force }

# Clean outputs
[linux]
clean-outputs:
    rm -f output/*.csv output/*.duckdb output/*.json output/*.parquet

# Clean outputs
[macos]
clean-outputs:
    rm -f output/*.csv output/*.duckdb output/*.json output/*.parquet

# Clean outputs
[windows]
clean-outputs:
    if (Test-Path "output") { Get-ChildItem "output" -Include "*.csv","*.duckdb","*.json","*.parquet" | Remove-Item -Force }

[windows]
clean:
    if (Test-Path ".venv") { Remove-Item ".venv" -Recurse -Force }


# Setup environment
get-started: pre-install venv

# Update project software versions in requirements
update-reqs:
    uv lock
    pre-commit autoupdate

# create virtual environment
venv:
    uv sync
    uv pip install git+https://github.com/huggingface/transformers
    uv pip install --upgrade "mistral-common[audio]"
    uv tool install pre-commit
    uv run pre-commit install

activate-venv:
    uv shell

# launch jupyter lab
lab:
    uv run jupyter lab

# Preview the quarto project
preview-docs:
    quarto preview

# Build the quarto project
build-docs:
    quarto render

# Lint python code
lint-py:
    uv run ruff check

# Format python code
fmt-python:
    uv run ruff format

# Format a single python file, "f"
fmt-py f:
    uv run ruff format {{ f }}

# Lint sql scripts
lint-sql:
    uv run sqlfluff fix --dialect duckdb

# Format all markdown and config files
fmt-markdown:
    uv run mdformat .

# Format a single markdown file, "f"
fmt-md f:
    uv run mdformat {{ f }}

# Check format of all markdown files
fmt-check-markdown:
    uv run mdformat --check .

fmt-all: lint-py fmt-python lint-sql fmt-markdown

# Run pre-commit hooks
pre-commit-run:
    pre-commit run

[windows]
pre-install:
    winget install Casey.Just astral-sh.uv GitHub.cli Posit.Quarto OpenJS.NodeJS
    npm install -g markdownlint-cli

[linux]
pre-install:
    brew install just uv gh markdownlint-cli ffmpeg

[macos]
pre-install:
    brew install just uv gh markdownlint- ffmpeg
    brew install --cask quarto
