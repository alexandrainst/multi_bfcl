"""Translation of tool calling examples."""

from copy import deepcopy
from textwrap import dedent

from .data_models import Example, TranslationOutput
from .languages import Language
from .llm import generate


def translate_example(
    example: Example, language: Language, language_example: str, model: str
) -> Example:
    """Translate a tool calling example to a different language.

    Args:
        example:
            The example to translate.
        language:
            The language code to translate the example to.
        language_example:
            An example of some text written in the target language.
        model:
            The model to use for translation.

    Returns:
        The translated example.
    """
    new_example = deepcopy(example)

    prompt = dedent(f"""
        You are a professional translator from English to {language.name} (language
        code: {language.code!r}).

        Here is an instruction in English:

        <example>
        {example.question[0][0]["content"]}
        </example>

        You need to translate the instruction to {language.name}.

        Here is an example of some text written in {language.name}:

        <{language.code}-example>
        {language_example.replace("\n", " ")}
        </{language.code}-example>

        You should return the translated instruction in JSON format, with the following
        structure:

        - `new_instruction` (str): The translated instruction.
    """).strip()

    generated_example = generate(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        response_format=TranslationOutput,
    )
    new_example.question[0][0]["content"] = generated_example.new_instruction
    return new_example
