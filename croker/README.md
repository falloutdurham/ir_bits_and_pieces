# Croker

![Croker](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzd6OWxtbGhoZ2w0M3ZyejVhZ2twNGdvdTJnbHo5MmNkeHhlOGV5bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XYAyfOJE5OpeU/giphy.gif)

A tool for generating 'golden' QREL (Query Relevance) datasets for Information Retrieval (IR) evaluation.


## Features

- Generates realistic search queries from documents using LLMs
- Uses BM25 to find potentially relevant documents for each query
- Judges document relevance using LLM
- Outputs results in TREC format and/or Parquet format
- Supports YAML configuration for easy parameter management

## Installation

### Using UV (recommended)

```bash
# Create a new virtual environment and install the package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### Basic Usage

```bash
# Set your Fireworks API key
export FIREWORKS_API_KEY=your-api-key

# Run with basic parameters
croker --input docs.parquet --output qrels.txt --queries_output queries.parquet --num_queries 100 --top_k 20
```

### Using YAML Configuration

Create a YAML configuration file:

```yaml
# config.yaml
input: "docs.parquet"
output: "qrels.txt"
queries_output: "queries.parquet"
num_queries: 100
top_k: 20
```

Then run:

```bash
croker --config config.yaml
```

### Optional Filtered Output

```bash
croker --input docs.parquet --output qrels.txt --queries_output queries.parquet \
  --filtered_output filtered.parquet --min_relevance 2
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input` | Input parquet file with documents | Required |
| `output` | Output QREL file in TREC format | Required |
| `queries_output` | Output parquet file for queries with doc ids | Required |
| `filtered_output` | Optional output parquet file for filtered relevance judgments | Optional |
| `min_relevance` | Minimum relevance score for filtered output | 0 |
| `id_col` | Document ID column name | "id" |
| `text_col` | Document text column name | "text" |
| `num_queries` | Number of queries to generate | 100 |
| `top_k` | Number of documents to judge per query | 20 |
| `perfect_relevance` | Relevance score for seed document | 3 |
| `model` | Fireworks model to use | "accounts/fireworks/models/llama-v3p1-8b-instruct" |
| `query_prompt` | Path to custom query prompt template file | Optional |
| `judge_prompt` | Path to custom judge prompt template file | Optional |
