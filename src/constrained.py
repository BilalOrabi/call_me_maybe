"""Constrained decoding engine for LLM function calling.

This module implements two-phase constrained decoding:
Phase 1 selects a function name from schema via PrefixTrie logit masking.
Phase 2 extracts typed parameter arguments using schema-guided type masking.
"""

import json
import re
from typing import Any
import numpy as np

from llm_sdk import Small_LLM_Model
from src.models import FunctionCallResult, FunctionDefinition
from src.trie import PrefixTrie


def build_vocab_map(model: Small_LLM_Model) -> list[str]:
    """Build an array mapping token IDs to their string representations.

    Args:
        model: Loaded Small_LLM_Model instance.

    Returns:
        A list of strings where list[token_id] is the token string.
    """
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as fp:
        vocab_dict: dict[str, int] = json.load(fp)

    id_to_token: list[str] = [""] * len(vocab_dict)

    for token, token_id in vocab_dict.items():
        id_to_token[token_id] = token

    return id_to_token


def select_function_name(
    model: Small_LLM_Model,
    trie: PrefixTrie,
    id_to_token: list[str],
    prompt: str,
    max_tokens: int = 15,
) -> str:
    """Select a valid function name using PrefixTrie logit masking.

    Args:
        model: Loaded Small_LLM_Model instance.
        trie: PrefixTrie containing all registered function names.
        id_to_token: Vocabulary mapping token ID to string.
        prompt: Context prompt containing instructions and query.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The chosen function name string.
    """
    input_ids: list[int] = model.encode(prompt).tolist()[0]
    current_name = ""
    steps = 0

    while current_name not in trie.words and steps < max_tokens:
        steps += 1

        logits: list[float] = model.get_logits_from_input_ids(input_ids)
        logits_array = np.array(logits, dtype=np.float32)

        for token_id, token_str in enumerate(id_to_token):
            if not trie.can_continue(current_name, token_str):
                logits_array[token_id] = -float("inf")

        best_token_id = int(np.argmax(logits_array))
        current_name += id_to_token[best_token_id]
        input_ids.append(best_token_id)

    return current_name


def is_valid_number_token(token: str) -> bool:
    """Check if a token can form part of a number or act as a stop delimiter.

    Args:
        token: Candidate token string from vocabulary.

    Returns:
        True if token contains only numeric characters or stop delimiters.
    """
    if "\n" in token or "," in token:
        return True

    cleaned = token.strip()

    if not cleaned:
        return True
    return all(char in "0123456789.-" for char in cleaned)


def parse_number_value(
    raw_text: str,
    param_type: str = "number",
) -> int | float:
    """Sanitize and parse a generated string into an int or float.

    Args:
        raw_text: Raw emitted string from the model.
        param_type: Target schema type ('number' for float, 'integer' for int).

    Returns:
        Parsed integer or float value, or 0 if conversion fails.
    """
    cleaned = raw_text.strip(" \t\n\r,\"')")

    try:
        result = float(cleaned)
    except ValueError:
        return 0.0 if param_type == "number" else 0

    if param_type == "integer":
        return int(round(result))

    return float(result)


def clean_string_delimiters(raw_text: str) -> str:
    """Strip closing function call syntax and outer whitespace or quotes.

    Args:
        raw_text: Raw decoded string emitted by the model.

    Returns:
        Sanitized string with trailing syntax and outer quotes stripped.
    """
    sliced = raw_text.split('"')[0]
    return sliced.strip(" \t\n\r\"'")


def extract_quoted_literal(query: str, candidate: str) -> str | None:
    """Extract verbatim quoted literal from query if candidate matches.

    Args:
        query: Full user prompt query.
        candidate: Candidate string decoded by the model.

    Returns:
        The verbatim quoted substring from query, or None if no match.
    """
    for q_match in re.findall(r"['\"](.*?)['\"]", query):
        clean_match = q_match.strip()
        clean_cand = candidate.strip()
        if clean_match == clean_cand or clean_cand in clean_match:
            return str(q_match)
    return None


def expand_path_or_token(query: str, candidate: str) -> str:
    """Expand candidate if it represents a file path or URL in the query.

    Args:
        query: Full user prompt query.
        candidate: Candidate string decoded by the model.

    Returns:
        Full path token from query if matched, otherwise candidate.
    """
    if not candidate:
        return candidate
    for token in query.split():
        clean_tok = token.strip(" \t\n\r,\"')")
        if candidate in clean_tok and any(s in clean_tok for s in ("/", "\\")):
            return clean_tok
    return candidate


def align_word_casing(query: str, candidate: str) -> str:
    """Align casing of candidate with query, prioritizing all-caps tokens.

    Args:
        query: Full user prompt query.
        candidate: Candidate string decoded by the model.

    Returns:
        Casing-aligned string matching the query's surface representation.
    """
    for word in re.findall(r"\b\w+\b", query):
        if (
            word.isupper()
            and len(word) > 1
            and word.upper() == candidate.upper()
        ):
            return str(word)
    m = re.search(r"\b" + re.escape(candidate) + r"\b", query, re.IGNORECASE)
    if m:
        return str(m.group(0))
    return candidate


def sanitize_string_value(query: str, raw_decoded: str) -> str:
    """Sanitize and align a decoded string parameter against query context.

    Args:
        query: Full user prompt query.
        raw_decoded: Raw string decoded from the model's generated tokens.

    Returns:
        Cleaned, aligned, and validated string parameter value.
    """
    decoded = clean_string_delimiters(raw_decoded)
    quoted = extract_quoted_literal(query, decoded)
    if quoted is not None:
        return quoted
    decoded = expand_path_or_token(query, decoded)
    return align_word_casing(query, decoded)


def extract_single_argument(
    model: Small_LLM_Model,
    id_to_token: list[str],
    query: str,
    fn_name: str,
    param_name: str,
    param_type: str,
    fn_desc: str = "",
    prev_args: str = "",
    max_tokens: int = 35,
) -> Any:
    """Extract a single typed parameter value using call-prefix prompt.

    Args:
        model: Loaded Small_LLM_Model instance.
        id_to_token: Vocabulary mapping token ID to string.
        query: User input query.
        fn_name: Name of the selected function.
        param_name: Name of the parameter to extract.
        param_type: Expected schema type ('number', 'integer', 'string').
        fn_desc: Schema description of the function.
        prev_args: Already extracted arguments formatted as prefix.
        max_tokens: Maximum tokens to generate for this argument.

    Returns:
        Extracted and sanitized Python value.
    """
    quote = '"' if param_type == "string" else ""
    desc_prefix = f"Task: {fn_desc}\n" if fn_desc else ""
    call_prefix = f"Call: {fn_name}({prev_args}{param_name}={quote}"
    prompt = f"{desc_prefix}User: {query}\n{call_prefix}"

    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated_ids: list[int] = []

    stop_delimiters: tuple[str, ...]
    if param_type == "string":
        stop_delimiters = ('"', "\n")
    else:
        stop_delimiters = (",", ")", "\n")

    for _ in range(max_tokens):
        logits: list[float] = model.get_logits_from_input_ids(input_ids)
        logits_array = np.array(logits, dtype=np.float32)

        if param_type in ("number", "integer"):
            for token_id, token_str in enumerate(id_to_token):
                if not is_valid_number_token(token_str):
                    logits_array[token_id] = -float("inf")

        best_token_id = int(np.argmax(logits_array))
        chosen_str = id_to_token[best_token_id]

        if generated_ids and any(d in chosen_str for d in stop_delimiters):
            if param_type == "string":
                generated_ids.append(best_token_id)
            break

        generated_ids.append(best_token_id)
        input_ids.append(best_token_id)

    raw_decoded = model.decode(generated_ids)

    if param_type in ("number", "integer"):
        return parse_number_value(raw_decoded, param_type=param_type)
    return sanitize_string_value(query, raw_decoded)


def extract_arguments(
    model: Small_LLM_Model,
    id_to_token: list[str],
    query: str,
    fn_def: FunctionDefinition,
) -> dict[str, Any]:
    """Extract all arguments for a function using cumulative context.

    Args:
        model: Loaded Small_LLM_Model instance.
        id_to_token: Vocabulary mapping token ID to string.
        query: User input query.
        fn_def: Validated FunctionDefinition schema.

    Returns:
        Dictionary mapping parameter names to extracted values.
    """
    parameters: dict[str, Any] = {}

    for param_name, param_def in fn_def.parameters.items():
        parts = []
        for k, v in parameters.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={v}")

        prev_args = ", ".join(parts)
        if prev_args:
            prev_args += ", "

        value = extract_single_argument(
            model=model,
            id_to_token=id_to_token,
            query=query,
            fn_name=fn_def.name,
            param_name=param_name,
            param_type=param_def.type,
            fn_desc=fn_def.description,
            prev_args=prev_args,
        )
        parameters[param_name] = value

    return parameters


def generate_function_call(
    model: Small_LLM_Model,
    id_to_token: list[str],
    trie: PrefixTrie,
    functions: list[FunctionDefinition],
    query: str,
) -> FunctionCallResult:
    """Orchestrate constrained decoding.

    Args:
        model: Loaded Small_LLM_Model instance.
        id_to_token: Vocabulary mapping token ID to string.
        trie: PrefixTrie with registered function names.
        functions: List of available FunctionDefinitions.
        query: User input query.

    Returns:
        Validated FunctionCallResult containing function name and arguments.
    """
    descriptions = "\n".join(
        f"- {func.name}: {func.description}" for func in functions
    )

    intro = (
        "Instructions: Select the single function from the list "
        "that best matches the query.\n\n"
    )
    selection_prompt = (
        f"{intro}"
        f"Available functions:\n{descriptions}\n\n"
        f"Query: {query}\n"
        f"Selected function: "
    )

    fn_name = select_function_name(
        model=model,
        trie=trie,
        id_to_token=id_to_token,
        prompt=selection_prompt,
    )

    fn_def = next((func for func in functions if func.name == fn_name), None)

    if fn_def is None:
        return FunctionCallResult(prompt=query, name=fn_name, parameters={})

    parameters = extract_arguments(
        model=model,
        id_to_token=id_to_token,
        query=query,
        fn_def=fn_def,
    )

    return FunctionCallResult(
        prompt=query,
        name=fn_name,
        parameters=parameters,
    )
