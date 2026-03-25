"""Data models used in the project."""

from pydantic import BaseModel


class FunctionParameters(BaseModel):
    """The parameters of a function."""

    type: str
    required: list[str]
    properties: dict[str, dict[str, str | list[str]]]


class FunctionDescription(BaseModel):
    """A description of a function."""

    name: str
    description: str
    parameters: FunctionParameters


class Example(BaseModel):
    """A BFCL example."""

    id: str
    question: list[list[dict[str, str]]]
    function: list[FunctionDescription]
    ground_truth: list[dict[str, dict]]


class TranslationOutput(BaseModel):
    """The output of the translation."""

    new_instruction: str
