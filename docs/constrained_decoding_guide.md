# Constrained Decoding Engine: End-to-End Architecture & Technical Guide

This document provides an exhaustive, production-grade breakdown of the function-calling engine implemented in [`src/constrained.py`](file:///d:/call_me_maybe/src/constrained.py). It explains the theory of logit-level constrained decoding, details every helper function and standard library method, and walks through complete end-to-end execution traces.

---

## Table of Contents
1. [Core Philosophy & Architecture](#1-core-philosophy--architecture)
2. [The LLM Generation Cycle & Logit Masking](#2-the-llm-generation-cycle--logit-masking)
3. [Underlying Methods & Standard Library Primitives](#3-underlying-methods--standard-library-primitives)
   - [JSON Serialization (`json.load`)](#json-serialization-jsonload)
   - [Numerical Computation (`numpy.argmax`, `numpy.array`)](#numerical-computation-numpyargmax-numpyarray)
   - [Regular Expressions (`re.search`, `re.findall`, `re.escape`, `\b`)](#regular-expressions-research-refindall-reescape-b)
4. [Function-by-Function Technical Breakdown](#4-function-by-function-technical-breakdown)
   - [Vocab & Function Selection](#vocab--function-selection)
   - [Numeric Type Handlers](#numeric-type-handlers)
   - [String Cleaning & Information Extraction Primitives](#string-cleaning--information-extraction-primitives)
   - [Argument Extraction & Orchestration](#argument-extraction--orchestration)
5. [Step-by-Step Execution Traces](#5-step-by-step-execution-traces)
   - [Trace 1: Numeric Function (`fn_add_numbers`)](#trace-1-numeric-function-fn_add_numbers)
   - [Trace 2: String & Path Function (`fn_read_file`)](#trace-2-string--path-function-fn_read_file)
6. [Defensive Design & Tradeoff Analysis](#6-defensive-design--tradeoff-analysis)

---

## 1. Core Philosophy & Architecture

### The Problem with Small Language Models (0.6B)
When instructed via plain text prompts to emit structured JSON:
```text
"Read the user query and return a valid JSON object matching the schema..."
```
Small models (such as `Qwen/Qwen3-0.6B`) fail over 70% of the time. Common failure modes include:
- Hallucinating function names not present in the schema.
- Emitting conversational preamble (`"Sure! Here is the JSON function call:"`).
- Truncating closing brackets, omitting quotes, or inventing new parameter keys.
- **Entity Reuse Bias**: Repeating previously extracted argument values across multiple parameters.

### The Solution: Two-Phase Logit-Level Constrained Decoding
Rather than hoping the model adheres to prompt instructions, **Constrained Decoding** mathematically forces the model to emit only valid tokens by intervening at the final layer of the neural network.

```
┌────────────────────────────────────────────────────────────────────────┐
│ User Query: "What is the sum of 265 and 345?"                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Function Name Selection                                       │
│ • Build PrefixTrie containing valid schema names                       │
│ • Step token-by-token: Mask all vocab tokens that don't match the Trie │
│ • Output: Guaranteed valid function name ("fn_add_numbers")            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Schema-Guided Parameter Extraction                            │
│ • Retrieve parameter definitions from FunctionDefinition               │
│ • Format context with Call-Prefix Prompting: "Call: fn_name("          │
│ • Inject Cumulative Context to prevent argument reuse: "a=265, b="    │
│ • Restrict logits based on type:                                       │
│     - number / integer: Mask to [0-9.-] and stop delimiters            │
│     - string: Free generation bounded by quotes and clean primitives   │
│ • Output: Validated FunctionCallResult                                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The LLM Generation Cycle & Logit Masking

To understand how constrained decoding operates, observe the 7 steps of the auto-regressive generation loop:

```
1. Prompt Text ──► 2. Tokenizer (BPE) ──► 3. Input IDs [892, 16, ...]
                                                      │
                                                      ▼
6. Token Selection ◄── 5. Logits Tensor ◄── 4. Transformer Forward Pass
   (Argmax / Best)     (Scores for 151,936 tokens)
         │                     ▲
         │                     │
         │             [ CONSTRAINED DECODING ]
         │             Invalid Token IDs forced to -∞
         ▼
7. Append Token to Input IDs ──► Repeat loop for next token
```

### The Mathematics of $-\infty$ Masking
The model projects its hidden states to a raw, unnormalized vector of scores called **logits** $\mathbf{z} \in \mathbb{R}^V$, where $V$ is the vocabulary size (151,936 for Qwen). Under standard generation, probabilities $P(w_i)$ are computed via the **Softmax** function:

$$P(w_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

When our validation logic determines that token $k$ is illegal (e.g., trying to generate letters during an integer parameter), we set:

$$z_k = -\infty$$

Because $e^{-\infty} = 0$:

$$P(w_k) = \frac{0}{\sum_{j=1}^{V} e^{z_j}} = 0$$

Under greedy decoding (argmax), the model **cannot pick token $k$ under any circumstance**. It is mathematically impossible.

---

## 3. Underlying Methods & Standard Library Primitives

Below is an in-depth explanation of every library method and pattern utilized in `src/constrained.py`.

### JSON Serialization (`json.load`)
```python
with open(vocab_path, "r", encoding="utf-8") as fp:
    vocab_dict: dict[str, int] = json.load(fp)
```
- **What it does**: Reads a UTF-8 JSON file containing a mapping from token string representations to integer token IDs (`{"<|endoftext|>": 151643, ...}`).
- **Why it matters**: Deserializes the model's vocabulary file from disk into memory as a standard Python dictionary so we can build our $O(1)$ fast lookup array.

---

### Numerical Computation (`numpy.argmax`, `numpy.array`)

```python
logits_array = np.array(logits, dtype=np.float32)
best_token_id = int(np.argmax(logits_array))
```
- **`np.array(logits, dtype=np.float32)`**:
  The LLM SDK returns logits as a standard Python `list[float]`. Iterating or modifying a Python list of 151,936 items in pure Python introduces significant overhead. Converting it to a contiguous C-level memory buffer via NumPy allows vector operations and fast index manipulation.
- **`np.argmax(logits_array)`**:
  Scans the array and returns the **index of the maximum value**. Since invalid tokens have their score forced to `-float("inf")`, `np.argmax` selects the highest-scoring **valid** token. We cast to `int` because NumPy returns a `numpy.int64` scalar.

---

### Regular Expressions (`re` Module)

Regular expressions are utilized in our clean string extraction layer to align model predictions with the original user query without hardcoding any function or parameter names.

#### 1. `re.findall(r"['\"](.*?)['\"]", query)`
Used in [`extract_quoted_literal`](file:///d:/call_me_maybe/src/constrained.py#L138-L153):
- **`['\"]`**: Character class matching either a single quote `'` or a double quote `"`.
- **`(.*?)`**: A capturing group `(...)`. 
  - `.` matches any character except a newline.
  - `*?` is a **non-greedy (lazy) quantifier**. It matches as few characters as possible until the closing quote is encountered. If we used greedy `.*`, the string `'hello' and 'world'` would match the whole block from the first `'` to the last `'`.
- **`re.findall`**: Scans `query` from left to right and returns a list of all non-overlapping matches for the capturing group.
  - *Example*: `query = "Reverse the string 'hello'"` $\to$ returns `["hello"]`.

#### 2. `re.escape(candidate)`
Used in [`align_word_casing`](file:///d:/call_me_maybe/src/constrained.py#L175-L195):
- **What it does**: Escapes any characters in `candidate` that have special meaning in regular expressions (such as `.`, `*`, `?`, `+`, `(`, `)`, `[`, `]`, `^`, `$`).
- **Why it is critical**: If `candidate` is `"cat"`, `re.escape` produces `"cat"`. But if `candidate` is `"user.profile"`, unescaped `.` would match *any* character. `re.escape("user.profile")` yields `"user\\.profile"`, ensuring it only matches the literal period.

#### 3. `\b` (Word Boundary)
Used in `r"\b" + re.escape(candidate) + r"\b"`:
- **What it does**: Matches the boundary between a word character (`\w`: letters, digits, underscore) and a non-word character (spaces, punctuation, start/end of string).
- **Why it is critical**: Prevents false positive substring matches.
  - Without `\b`: Searching for `"cat"` in `"The caterpillar sat"` matches `"cat"` inside `"caterpillar"`.
  - With `\b`: Searching for `\bcat\b` will **only** match the standalone word `"cat"`.

#### 4. `re.IGNORECASE` Flag
- Tells the regex engine to treat uppercase and lowercase letters as identical during matching (`"shrek"` matches `"Shrek"` or `"SHREK"`).
- We use this to detect that the model emitted the word `"shrek"`, search for where that entity appeared in the user's prompt, and extract the exact casing (`"Shrek"`) written by the user.

---

## 4. Function-by-Function Technical Breakdown

Here is the exact responsibility of every function in [`src/constrained.py`](file:///d:/call_me_maybe/src/constrained.py).

### Vocab & Function Selection

#### `build_vocab_map(model: Small_LLM_Model) -> list[str]`
- **Purpose**: Creates an $O(1)$ direct array mapping token IDs to their string tokens.
- **Implementation**: Reads the tokenizer vocabulary file (`vocab.json`). Allocates a list of size `len(vocab_dict)` and sets `id_to_token[token_id] = token`.
- **Complexity**: $O(V)$ setup time, $O(1)$ token lookup during decoding.

#### `select_function_name(model, trie, id_to_token, prompt, max_tokens=15) -> str`
- **Purpose**: Implements **Phase 1** constrained decoding.
- **Workflow**:
  1. Encodes the selection prompt into `input_ids`.
  2. In an auto-regressive loop, computes logits for the next token.
  3. Iterates over `id_to_token`. For each token, tests `trie.can_continue(current_name, token_str)`.
  4. If appending `token_str` violates the Trie of valid schema functions, its logit is set to `-float("inf")`.
  5. Selects the highest valid token via `np.argmax`, appends it to `current_name`, and continues until `current_name` is an exact match in `trie.words`.

---

### Numeric Type Handlers

#### `is_valid_number_token(token: str) -> bool`
- **Purpose**: Logit mask validator for numeric parameters.
- **Logic**: Returns `True` if `token` consists exclusively of characters `0123456789.-` or is a whitespace / stop delimiter (`","`, `"\n"`). Rejects any tokens containing alphabetic characters or unexpected punctuation.

#### `parse_number_value(raw_text: str, param_type: str = "number") -> int | float`
- **Purpose**: Sanitizes and strongly types an extracted numeric string.
- **Logic**:
  1. Strips trailing syntax characters: `raw_text.strip(" \t\n\r,\"')")`.
  2. Converts to `float(cleaned)`.
  3. If `param_type == "integer"`, returns `int(round(result))`.
  4. If `param_type == "number"`, returns `float(result)`.
- **Production Context**: In Python, `isinstance(2, float)` is `False`. The grading schema strictly asserts `isinstance(a, float)` for `number` schemas and `isinstance(n, int)` for `integer` schemas. This function enforces strong schema typing.

---

### String Cleaning & Information Extraction Primitives

#### `clean_string_delimiters(raw_text: str) -> str`
- **Purpose**: Strips Python call syntax trailing delimiters emitted by the model.
- **Logic**: Slices before `",` or `")` (`raw_text.split('",')[0].split('")')[0]`) and strips outer quotes and whitespace.

#### `extract_quoted_literal(query: str, candidate: str) -> str | None`
- **Purpose**: Preserves exact strings when users wrap inputs in quotes.
- **Logic**: If the prompt contains `'hello'` or `"SELECT * FROM users"`, and `candidate` matches or is contained inside that quoted phrase, returns the verbatim quoted text from the query.

#### `expand_path_or_token(query: str, candidate: str) -> str`
- **Purpose**: Expands file paths or URLs.
- **Logic**: When prompted with `Read /home/user/data.json`, small models often emit only the basename `data.json`. This function scans the query's whitespace tokens. If `candidate` is part of a token containing path separators (`/` or `\`), it recovers the full path `/home/user/data.json`.

#### `align_word_casing(query: str, candidate: str) -> str`
- **Purpose**: Aligns casing with prompt surface forms.
- **Logic**:
  1. Checks if an all-caps form exists in `query` (e.g. `NUMBERS`). If so, prioritizes the all-caps token.
  2. Otherwise, scans for a whole-word match using `\b` and adopts the query's exact casing.

#### `sanitize_string_value(query: str, raw_decoded: str) -> str`
- **Purpose**: Orchestrates the string post-processing pipeline.
- **Pipeline Order**:
  `clean_string_delimiters` $\to$ `extract_quoted_literal` $\to$ `expand_path_or_token` $\to$ `align_word_casing`.

---

### Argument Extraction & Orchestration

#### `extract_single_argument(...) -> Any`
- **Purpose**: Implements **Phase 2** single-parameter extraction.
- **Logic**:
  1. Constructs the **Call-Prefix Prompt**:
     ```text
     User: {query}
     Call: {fn_name}({prev_args}{param_name}={quote}
     ```
  2. Defines stop delimiters:
     - For `string`: `('"', "\n")` (Note: `,` is NOT a stop delimiter for strings, allowing commas inside SQL or sentences).
     - For `number` / `integer`: `(",", ")", "\n")`.
  3. Executes auto-regressive generation loop with type logit masking for numbers.
  4. Decodes generated token IDs via `model.decode(generated_ids)`.
  5. Routes to `parse_number_value` for numeric types, or `sanitize_string_value` for strings.

#### `extract_arguments(model, id_to_token, query, fn_def) -> dict[str, Any]`
- **Purpose**: Iterates through the function's parameter schema.
- **Cumulative Context**: Maintains already-extracted arguments:
  ```python
  parts.append(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}")
  prev_args = ", ".join(parts) + ", "
  ```
  Passing previously extracted values prevents the LLM from suffering from **Entity Reuse Bias**.

#### `generate_function_call(model, id_to_token, trie, functions, query) -> FunctionCallResult`
- **Purpose**: The top-level orchestrator connecting Phase 1 and Phase 2.
- **Workflow**:
  1. Builds semantic prompt listing all available functions and their descriptions.
  2. Calls `select_function_name` with PrefixTrie logit masking.
  3. Looks up the matching `FunctionDefinition`.
  4. Calls `extract_arguments` to populate the parameter dictionary.
  5. Returns a validated `FunctionCallResult(prompt=query, name=fn_name, parameters=parameters)`.

---

## 5. Step-by-Step Execution Traces

### Trace 1: Numeric Function (`fn_add_numbers`)

**Query**: `"What is the sum of 265 and 345?"`

#### Phase 1: Function Selection
1. **Prompt Formatted**:
   ```text
   Instructions: Select the single function from the list that best matches the query.

   Available functions:
   - fn_add_numbers: Add two numbers together and return their sum.
   - fn_greet: Generate a greeting message for a person by name.
   ...
   Query: What is the sum of 265 and 345?
   Selected function: 
   ```
2. **Decoding Step 1**:
   - `current_name = ""`
   - Allowed prefixes in Trie: tokens starting with `"fn_"`. All other 151,000+ tokens set to `-inf`.
   - Highest logit: `"fn"` $\to$ `current_name = "fn"`.
3. **Decoding Step 2**:
   - Allowed prefixes: tokens starting with `"_add"`, `"_greet"`, etc.
   - Highest logit: `"_add_numbers"` $\to$ `current_name = "fn_add_numbers"`.
4. **Completion**:
   - `"fn_add_numbers"` matches a complete word in `trie.words`. Loop terminates.
   - Phase 1 Result: `"fn_add_numbers"`.

#### Phase 2: Argument Extraction
The schema for `fn_add_numbers` defines:
```json
{"a": {"type": "number"}, "b": {"type": "number"}}
```

**Extracting Parameter `a`**:
- Prefix: `User: What is the sum of 265 and 345?\nCall: fn_add_numbers(a=`
- Logit masking: All non-numeric tokens set to `-inf`.
- Model generates: `265` followed by stop delimiter `,`.
- Decoded: `"265,"`.
- `parse_number_value("265,", "number")` $\to$ `265.0` (`float`).
- Parameters map: `{"a": 265.0}`.

**Extracting Parameter `b` (with Cumulative Context)**:
- Prefix: `User: What is the sum of 265 and 345?\nCall: fn_add_numbers(a=265.0, b=`
- Attention mechanism: Because `a=265.0` is already in the prompt, the self-attention heads suppress `265` and attend directly to `345`.
- Model generates: `345` followed by stop delimiter `)`.
- `parse_number_value("345)", "number")` $\to$ `345.0` (`float`).
- Parameters map: `{"a": 265.0, "b": 345.0}`.

**Final Result**:
```json
{
  "prompt": "What is the sum of 265 and 345?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 265.0,
    "b": 345.0
  }
}
```

---

### Trace 2: String & Path Function (`fn_read_file`)

**Query**: `"Read the file at /home/user/data.json with utf-8 encoding"`

#### Phase 1: Function Selection
- PrefixTrie masks invalid function names.
- Model selects `"fn_read_file"`.

#### Phase 2: Parameter Extraction
Schema: `{"path": {"type": "string"}, "encoding": {"type": "string"}}`

**Extracting Parameter `path`**:
- Call Prefix:
  ```text
  User: Read the file at /home/user/data.json with utf-8 encoding
  Call: fn_read_file(path="
  ```
- Stop delimiters: `('"', "\n")`.
- Model generates tokens: `data.json"` (model emitted only the filename).
- Raw decoded: `"data.json"`.
- `clean_string_delimiters("data.json")` $\to$ `"data.json"`.
- `extract_quoted_literal` $\to$ `None` (path was not in quotes in the prompt).
- `expand_path_or_token(query, "data.json")`:
  - Scans query tokens: `["Read", "the", "file", "at", "/home/user/data.json", "with", "utf-8", "encoding"]`.
  - Token `"/home/user/data.json"` contains `"data.json"` and has path separator `/`.
  - Returns `"/home/user/data.json"`.
- Extracted: `path = "/home/user/data.json"`.

**Extracting Parameter `encoding`**:
- Call Prefix with Cumulative Context:
  ```text
  User: Read the file at /home/user/data.json with utf-8 encoding
  Call: fn_read_file(path="/home/user/data.json", encoding="
  ```
- Model generates: `utf-8")`.
- Raw decoded: `"utf-8")"`.
- `clean_string_delimiters` strips `")"` $\to$ `"utf-8"`.
- Extracted: `encoding = "utf-8"`.

**Final Result**:
```json
{
  "prompt": "Read the file at /home/user/data.json with utf-8 encoding",
  "name": "fn_read_file",
  "parameters": {
    "path": "/home/user/data.json",
    "encoding": "utf-8"
  }
}
```

---

## 6. Defensive Design & Tradeoff Analysis

### 1. Zero Hardcoding vs. Test Contamination
- **Rule**: *"The given input files may change during the peer review. Do not hardcode solutions based on the provided examples."*
- **Our Design**: There is not a single `if fn_name == ...` or `if param_name == ...` check in [`src/constrained.py`](file:///d:/call_me_maybe/src/constrained.py).
- **Benefit**: If a peer reviewer tests unseen functions (e.g. `fn_download_data(url: str, timeout: int)`), our pipeline executes identically and correctly.

### 2. Why Not Stop at Commas for Strings?
- For numbers, `,` is a stop delimiter (`a=2, b=3`).
- For strings, `,` is valid syntax:
  - SQL query: `'INSERT INTO logs VALUES (1, 2, 3)'`.
  - Template: `'Hello, world!'`.
- **Engineering Decision**: Strings only stop at closing double quotes `"` or newlines `\n`.

### 3. Open-Closed Principle (OCP)
The engine is **open for extension, closed for modification**:
- To add 50 new functions to the system, you only edit `functions_definition.json`.
- The engine dynamically populates the `PrefixTrie`, builds the Phase 1 prompt, and drives Phase 2 type masking purely from the Pydantic schema without changing one line of Python code.
