"""Prefix Tree (Trie) for constrained decoding over fixed candidate strings.

This module provides the data structure used in Phase 1 to guarantee that
the LLM only selects a valid function name from the available schema.
"""


class TrieNode:
    """A single node within the prefix tree."""

    def __init__(self) -> None:
        """Initialize a Trie node with child mappings and completion flag."""
        self.children: dict[str, TrieNode] = {}
        self.is_complete_word: bool = False


class PrefixTrie:
    """Trie structure storing allowed string targets for generation."""

    def __init__(self, words: list[str] | None = None) -> None:
        """Initialize the Trie, optionally populating it with initial words.

        Args:
            words: Optional list of strings to insert.
        """
        self.root = TrieNode()
        self.words: list[str] = []
        if words:
            for word in words:
                self.insert(word)

    def insert(self, word: str) -> None:
        """Insert a word into the Trie.

        Args:
            word: The string to insert.
        """
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_complete_word = True

        if word not in self.words:
            self.words.append(word)

    def is_valid_prefix(self, prefix: str) -> bool:
        """Check whether a prefix matches any path in the Trie.

        Args:
            prefix: The substring to test.

        Returns:
            True if the prefix can lead to at least one valid word;
            False otherwise.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def can_continue(self, current_prefix: str, candidate_chunk: str) -> bool:
        """Determine if appending candidate_chunk keeps us on a valid path.

        Args:
            current_prefix: The string generated so far.
            candidate_chunk: The proposed next token text.

        Returns:
            True if combined string is a valid prefix; False otherwise.
        """
        combined = current_prefix + candidate_chunk
        return self.is_valid_prefix(combined)
