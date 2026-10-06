"""Command-line interface and orchestrator for function calling.

Parses command-line arguments, loads schemas and test prompts,
initializes the model, runs two-phase constrained decoding,
and serializes the results to disk.
"""

import argparse
from llm_sdk import Small_LLM_Model
from src.constrained import build_vocab_map, generate_function_call
from src.loader import (
    load_function_definitions,
    load_prompts,
    save_results,
)
from src.trie import PrefixTrie


def parse_args() -> argparse.Namespace:
    """Configure and parse command-line arguments.

    Returns:
        Parsed arguments Namespace containing file paths.
    """
    parser = argparse.ArgumentParser(
        description="LLM Function Calling via Constrained Decoding",
        epilog="Made by borabi (42 School)"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file with function definitions",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file with user prompts",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file for generated calls",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end function calling pipeline."""
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    prompts = load_prompts(args.input)

    model = Small_LLM_Model()
    trie = PrefixTrie([func.name for func in functions])
    id_to_token = build_vocab_map(model)

    results = []
    for item in prompts:
        res = generate_function_call(
            model=model,
            id_to_token=id_to_token,
            trie=trie,
            functions=functions,
            query=item.prompt,
        )
        results.append(res)

    save_results(args.output, results)


if __name__ == "__main__":
    main()
