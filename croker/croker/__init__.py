#!/usr/bin/env python
"""
QREL Generator Script

Generates a QREL file for IR evaluation from a document parquet file by:
1. Generating queries for documents using Fireworks LLM
2. Using BM25 to find potentially relevant documents for each query
3. Using LLM to judge relevance of top-k documents
4. Outputting a QREL file compatible with ranx

Enhanced with:
- YAML configuration support
- Parquet output capability with relevance filtering

Usage:
    python qrel_generator.py --input docs.parquet --output qrels.txt --num_queries 100 --top_k 20
    python qrel_generator.py --config config.yaml
"""

import argparse
import os
import random
import uuid
import yaml
from typing import List, Dict, Optional, Tuple, Callable, Any, Union

import polars as pl
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os
import fireworks.client


DEFAULT_QUERY_PROMPT = """
You are an expert at creating realistic search engine queries. 
Given a document, your task is to create a search query that someone might use to find this document.
The query should be realistic, concise, and directly related to the main topic of the document.
Do not make the query too specific or copy verbatim from the document.
Make it similar to what a real user would type into a search engine.

DOCUMENT:
{document}

QUERY:
"""

DEFAULT_JUDGE_PROMPT = """
You are an expert at judging document relevance to search queries.
Your task is to determine how relevant a document is to a given query on a scale from 0-2:

0: Not relevant - The document has no relation to the query or only mentions terms in passing.
1: Somewhat relevant - The document partially addresses the query or contains relevant information but is not comprehensive.
2: Highly relevant - The document directly addresses the main aspects of the query and provides substantial information.

Please provide just the numeric score (0, 1, or 2) without any explanation.

QUERY: {query}

DOCUMENT: {document}

RELEVANCE SCORE (0-2):
"""


class QrelGenerator:
    """
    QREL Generator for IR evaluation datasets
    """

    def __init__(
        self,
        query_prompt_template: str = DEFAULT_QUERY_PROMPT,
        judge_prompt_template: str = DEFAULT_JUDGE_PROMPT,
        model_name: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ):
        """
        Initialize the QREL Generator

        Args:
            llm_client: Fireworks client for LLM API calls
            query_prompt_template: Template for query generation prompt
            judge_prompt_template: Template for relevance judgment prompt
            model_name: Model to use for LLM calls
        """
        # No need to store a client instance when using fireworks SDK directly
        self.query_prompt_template = query_prompt_template
        self.judge_prompt_template = judge_prompt_template
        self.model_name = model_name

        self.documents = {}  # doc_id: text
        self.queries = {}  # query_id: text
        self.qrels = []  # list of (query_id, doc_id, relevance) tuples

        # BM25 components
        self.bm25 = None
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}

    def add_documents(self, documents: Dict[str, str]) -> None:
        """Add documents to the generator"""
        self.documents.update(documents)
        # Reset BM25 index when documents change
        self.bm25 = None

    def _initialize_bm25(self) -> None:
        """Initialize BM25 index with current documents"""
        doc_ids = list(self.documents.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.idx_to_doc_id = {idx: doc_id for doc_id, idx in self.doc_id_to_idx.items()}

        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [self.documents[doc_id].split() for doc_id in doc_ids]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

    def get_bm25_top_k(self, query: str, k: int = 20) -> List[str]:
        """
        Get top-k documents for a query using BM25

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of document IDs
        """
        if self.bm25 is None:
            self._initialize_bm25()

        # Tokenize query and get scores
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_idx = np.argsort(doc_scores)[-k:][::-1]

        # Convert indices back to doc_ids
        return [self.idx_to_doc_id[idx] for idx in top_k_idx]

    def generate_query(self, doc_id: str) -> str:
        """
        Generate a query for a document using LLM

        Args:
            doc_id: Document ID

        Returns:
            Generated query text
        """
        document = self.documents[doc_id]
        prompt = self.query_prompt_template.format(document=document)

        # Use Fireworks SDK directly
        try:
            completion = fireworks.client.Completion.create(
                model=self.model_name, prompt=prompt, temperature=0.7, max_tokens=50
            )

            return completion.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating queries for document {doc_id}: {e}")
            return []

    def judge_relevance(self, query: str, doc_id: str) -> int:
        """
        Judge relevance of a document to a query using LLM

        Args:
            query: Query text
            doc_id: Document ID

        Returns:
            Relevance score (0-2)
        """
        document = self.documents[doc_id]
        prompt = self.judge_prompt_template.format(query=query, document=document)

        # Use Fireworks SDK directly
        try:
            completion = fireworks.client.Completion.create(
                model=self.model_name, prompt=prompt, temperature=0.2, max_tokens=5
            )

            # Extract numeric score
            result = completion.choices[0].text.strip()
            try:
                # Extract first digit from response
                for char in result:
                    if char.isdigit():
                        score = int(char)
                        if 0 <= score <= 2:
                            return score
                # Default to 0 if no valid digit found
                return 0
            except:
                return 0
        except Exception as e:
            print(f"Error judging queries for document {doc_id}: {e}")
            return 0

    def generate_qrels(
        self,
        num_queries: int,
        top_k: int = 20,
        perfect_relevance: int = 3,
        resample: bool = True,
    ) -> List[Tuple[str, str, int]]:
        """
        Generate QREL dataset

        Args:
            num_queries: Number of queries to generate
            top_k: Number of documents to judge per query
            perfect_relevance: Relevance score for seed document
            resample: Whether to allow resampling documents for queries

        Returns:
            List of (query_id, doc_id, relevance) tuples
        """
        # Select seed documents for query generation
        if resample or num_queries > len(self.documents):
            # Sample with replacement
            seed_docs = random.choices(list(self.documents.keys()), k=num_queries)
        else:
            # Sample without replacement
            seed_docs = random.sample(list(self.documents.keys()), k=num_queries)

        # Generate queries and judgments
        for i, doc_id in enumerate(tqdm(seed_docs, desc="Generating queries")):
            query_id = f"Q{i+1}"

            # Generate query
            query_text = self.generate_query(doc_id)
            self.queries[query_id] = query_text

            # Add perfect match judgment
            self.qrels.append((query_id, doc_id, perfect_relevance))

            # Get top-k BM25 results
            top_k_docs = self.get_bm25_top_k(query_text, k=top_k)

            # Judge only top-k documents (excluding the seed document)
            for candidate_doc_id in tqdm(
                top_k_docs,
                desc=f"Judging docs for query {i+1}/{num_queries}",
                leave=False,
            ):
                if candidate_doc_id != doc_id:
                    relevance = self.judge_relevance(query_text, candidate_doc_id)
                    self.qrels.append((query_id, candidate_doc_id, relevance))

        return self.qrels

    def export_trec_format(self, output_file: str) -> None:
        """
        Export QREL judgments in TREC format

        Args:
            output_file: Path to output file
        """
        with open(output_file, "w") as f:
            for query_id, doc_id, relevance in sorted(self.qrels):
                f.write(f"{query_id} 0 {doc_id} {relevance}\n")

    def export_queries(self, output_file: str) -> None:
        """
        Export queries to a file

        Args:
            output_file: Path to output file
        """
        with open(output_file, "w") as f:
            for query_id, query_text in sorted(self.queries.items()):
                f.write(f"{query_id}\t{query_text}\n")

    def export_query_parquet(self, output_file: str) -> None:
        """
        Export queries with only their seed document IDs as a parquet file

        Args:
            output_file: Path to output parquet file
        """
        # Store mapping of query_id to seed doc_id
        query_to_seed_doc = {}

        # Find the seed document for each query (the one with perfect_relevance)
        for query_id, doc_id, relevance in self.qrels:
            # The seed document is the one with the highest relevance score (perfect_relevance)
            # We assume this is 3 by default
            if relevance >= 3:
                query_to_seed_doc[query_id] = doc_id

        # Create data for export
        data = []
        for query_id, query_text in self.queries.items():
            if query_id in query_to_seed_doc:
                data.append((query_id, query_text, query_to_seed_doc[query_id]))

        # Create dataframe
        df = pl.DataFrame(
            {
                "qid": [item[0] for item in data],
                "query_text": [item[1] for item in data],
                "doc_id": [item[2] for item in data],
            }
        )

        # Write to parquet
        df.write_parquet(output_file)
        print(
            f"Exported {len(data)} queries with their seed document IDs to {output_file}"
        )

    def export_parquet(self, output_file: str, min_relevance: int = 0) -> None:
        """
        Export filtered QREL judgments as a parquet file

        Args:
            output_file: Path to output parquet file
            min_relevance: Minimum relevance score to include (filter threshold)
        """
        # Filter qrels by minimum relevance score
        filtered_qrels = [
            (q_id, d_id, rel) for q_id, d_id, rel in self.qrels if rel >= min_relevance
        ]

        # Create dataframe
        df = pl.DataFrame(
            {
                "query_id": [q_id for q_id, _, _ in filtered_qrels],
                "query": [self.queries.get(q_id, "") for q_id, _, _ in filtered_qrels],
                "doc_id": [d_id for _, d_id, _ in filtered_qrels],
                "relevance": [rel for _, _, rel in filtered_qrels],
            }
        )

        # Write to parquet
        df.write_parquet(output_file)
        print(
            f"Exported {len(filtered_qrels)} filtered relevance judgments to {output_file}"
        )


def load_config(config_file: str) -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate QREL dataset from documents")
    parser.add_argument("--config", help="Path to YAML configuration file")

    # Standard command line arguments (can be overridden by config file)
    parser.add_argument("--input", help="Input parquet file with documents")
    parser.add_argument("--output", help="Output QREL file in TREC format (required)")
    parser.add_argument(
        "--queries_output",
        help="Output parquet file for queries with doc ids (required)",
    )
    parser.add_argument(
        "--filtered_output",
        help="Optional output parquet file for filtered relevance judgments",
    )
    parser.add_argument(
        "--min_relevance",
        type=int,
        default=0,
        help="Minimum relevance score for parquet output",
    )
    parser.add_argument("--id_col", default="id", help="Document ID column name")
    parser.add_argument("--text_col", default="text", help="Document text column name")
    parser.add_argument(
        "--num_queries", type=int, default=100, help="Number of queries to generate"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of documents to judge per query"
    )
    parser.add_argument(
        "--perfect_relevance",
        type=int,
        default=3,
        help="Relevance score for seed document",
    )
    parser.add_argument("--query_prompt", help="Path to query prompt template file")
    parser.add_argument("--judge_prompt", help="Path to judge prompt template file")
    parser.add_argument(
        "--fireworks_api_key", help="Fireworks API key (defaults to env var)"
    )
    parser.add_argument(
        "--model",
        default="accounts/fireworks/models/mixtral-8x7b-instruct",
        help="Fireworks model to use",
    )

    args = parser.parse_args()

    # Load configuration from file if specified
    config = {}
    if args.config:
        config = load_config(args.config)

    # Override config with command line arguments
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None and arg != "config":
            config[arg] = value

    # Check for required parameters
    if "input" not in config:
        raise ValueError("Input file required. Specify --input or in config file.")
    if "output" not in config:
        raise ValueError(
            "Output QREL file required. Specify --output or in config file."
        )
    if "queries_output" not in config:
        raise ValueError(
            "Queries output file required. Specify --queries_output or in config file."
        )

    # Set up Fireworks client
    api_key = config.get("fireworks_api_key") or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "Fireworks API key required. Set FIREWORKS_API_KEY environment variable or use --fireworks_api_key"
        )

    # Configure Fireworks client
    fireworks.client.api_key = api_key

    # Load custom prompts if provided
    query_prompt = DEFAULT_QUERY_PROMPT
    judge_prompt = DEFAULT_JUDGE_PROMPT

    if "query_prompt" in config:
        with open(config["query_prompt"], "r") as f:
            query_prompt = f.read()

    if "judge_prompt" in config:
        with open(config["judge_prompt"], "r") as f:
            judge_prompt = f.read()

    # Load documents
    print(f"Loading documents from {config['input']}")
    df = pl.read_parquet(config["input"])

    # Ensure required columns exist
    id_col = config.get("id_col", "id")
    text_col = config.get("text_col", "text")
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Required columns not found. Need {id_col} and {text_col}")

    # Convert to dictionary
    documents = dict(zip(df[id_col].to_list(), df[text_col].to_list()))
    print(f"Loaded {len(documents)} documents")

    # Initialize generator
    generator = QrelGenerator(
        query_prompt_template=query_prompt,
        judge_prompt_template=judge_prompt,
        model_name=config.get(
            "model", "accounts/fireworks/models/mixtral-8x7b-instruct"
        ),
    )
    generator.add_documents(documents)

    # Generate QRELs
    num_queries = config.get("num_queries", 100)
    top_k = config.get("top_k", 20)
    perfect_relevance = config.get("perfect_relevance", 3)
    print(
        f"Generating {num_queries} queries and judging top-{top_k} documents for each"
    )
    qrels = generator.generate_qrels(
        num_queries=num_queries, top_k=top_k, perfect_relevance=perfect_relevance
    )

    # Export TREC format (always required)
    generator.export_trec_format(config["output"])
    print(f"Exported {len(qrels)} relevance judgments to {config['output']}")

    # Export query parquet (always required)
    generator.export_query_parquet(config["queries_output"])

    # Export filtered relevance judgments if requested
    if "filtered_output" in config:
        min_relevance = config.get("min_relevance", 0)
        generator.export_parquet(config["filtered_output"], min_relevance=min_relevance)


if __name__ == "__main__":
    main()