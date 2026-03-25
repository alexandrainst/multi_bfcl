"""Loading of data to use in the project."""

import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import DatasetInfo, HfApi

from .data_models import Example
from .languages import Language, get_all_languages


def load_bfcl() -> list[Example]:
    """Load the BFCL-v2 dataset.

    Returns:
        A list of examples.
    """
    examples: list = []
    for subset_name in [
        "live_multiple",
        "live_parallel_multiple",
        "live_parallel",
        "live_simple",
        "multiple",
        "parallel_multiple",
        "parallel",
        "simple_java",
        "simple_javascript",
        "simple_python",
    ]:
        url_prefix = (
            "https://raw.githubusercontent.com/ShishirPatil/gorilla"
            "/refs/heads/main/berkeley-function-call-leaderboard/bfcl_eval/data"
        )
        input_url = f"{url_prefix}/BFCL_v4_{subset_name}.json"
        ground_truth_url = f"{url_prefix}/possible_answer/BFCL_v4_{subset_name}.json"
        print(f"Loading dataset '{subset_name}'")
        inputs = _load_jsonl_from_url(input_url)
        ground_truth = _load_jsonl_from_url(ground_truth_url)

        # Join input and ground_truth by 'id' key
        ground_truth_by_id = {item["id"]: item for item in ground_truth}
        for item in inputs:
            item_id = item["id"]
            gt: dict = ground_truth_by_id.get(item_id, {})
            joined: dict = item | gt
            examples.append(joined)

    return [Example.model_validate(example) for example in examples]


def load_languages() -> list[Language]:
    """Load a list of all the languages in MultiWikiQA.

    Returns:
        A list of languages.
    """
    api = HfApi()
    repo_info = api.repo_info("alexandrainst/multi-wiki-qa", repo_type="dataset")
    assert isinstance(repo_info, DatasetInfo), (
        f"Expected a DatasetInfo object, but got {type(repo_info)}"
    )

    language_code_to_language = get_all_languages()
    languages = [
        language_code_to_language[config["config_name"]]
        for config in repo_info.cardData.configs
        if config["config_name"] in language_code_to_language
    ]

    return languages


def _load_jsonl_from_url(url: str) -> list:
    """Load jsonl from url.

    Args:
        url: url to JSONL

    Returns:
        List of deserialized objects
    """
    with urlopen(url) as r:
        path = r.read().decode()
        if isinstance(path, Path):
            path = path.read_text()
        return [json.loads(line) for line in path.splitlines()]
