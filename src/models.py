"""Data models and schemas validated using Pydantic.

This module defines structured data representations for function definitions,
parameters, input test prompts, and function calling results.
"""

from typing import Any
from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):
    """Schema definition for a single parameter of a function."""

    type: str


class ReturnDefinition(BaseModel):
    """Schema definition for the return value of a function."""

    type: str


class FunctionDefinition(BaseModel):
    """Schema definition for a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition | None = None


class PromptItem(BaseModel):
    """Schema definition for an incoming natural language prompt item."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Schema definition for the final structured function call result."""

    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
