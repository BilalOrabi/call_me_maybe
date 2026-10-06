"""Data loading and persistence layer with robust error handling.

This module provides functions to safely load and validate function
definitions, parse natural language prompts, and save execution results.
"""

import json
from pathlib import Path
import sys

from pydantic import ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptItem


def load_function_definitions(
    file_path: Path | str,
) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        file_path: Filesystem path to the definitions JSON file.

    Returns:
        List of validated FunctionDefinition models.
    """
    definitions: list[FunctionDefinition] = []
    path = Path(file_path)
    if not path.is_file():
        sys.stderr.write(f"Error: File '{path}' does not exist.\n")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        sys.stderr.write(f"Error: Corrupted JSON in '{path}': {err}\n")
        sys.exit(1)

    if not isinstance(raw_data, list):
        sys.stderr.write(
            f"Error: Expected a JSON array in '{path}', "
            f"got {type(raw_data).__name__}.\n"
        )
        sys.exit(1)

    for item in raw_data:
        try:
            definitions.append(FunctionDefinition.model_validate(item))
        except ValidationError as err:
            sys.stderr.write(
                f"Error: Corrupted JSON schema keys '{path}': {err}\n"
            )
            sys.exit(1)

    return definitions


def load_prompts(file_path: Path | str) -> list[PromptItem]:
    """Load and validate natural language prompts from a JSON file.

    Args:
        file_path: Filesystem path to the prompts JSON file.

    Returns:
        List of validated PromptItem models.
    """
    prompt: list[PromptItem] = []
    path = Path(file_path)

    if not path.is_file():
        sys.stderr.write(f"Error: File '{path}' does not exist.\n")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as file_pointer:
            raw_data = json.load(file_pointer)
    except json.JSONDecodeError as err:
        sys.stderr.write(f"Error: Corrupted JSON in '{path}': {err}\n")
        sys.exit(1)

    if not isinstance(raw_data, list):
        sys.stderr.write(
            f"Error: Expected a JSON array in '{path}', "
            f"got {type(raw_data).__name__}.\n"
        )
        sys.exit(1)

    for item in raw_data:
        try:
            prompt.append(PromptItem.model_validate(item))
        except ValidationError as err:
            sys.stderr.write(
                f"Error: Corrupted JSON schema keys '{path}': {err}\n"
            )
            sys.exit(1)

    return prompt


def save_results(
    file_path: Path | str, results: list[FunctionCallResult]
) -> None:
    """Serialize and write function calling results to disk as JSON.

    Args:
        file_path: Destination path where the output file will be written.
        results: List of FunctionCallResult models to serialize.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = [result.model_dump() for result in results]

    try:
        with open(path, "w", encoding="utf-8") as file_pointer:
            json.dump(
                data_to_save, file_pointer, ensure_ascii=False, indent=2
            )
    except OSError as err:
        sys.stderr.write(
            f"Error: Failed to write output file '{path}': {err}\n"
        )
        sys.exit(1)
