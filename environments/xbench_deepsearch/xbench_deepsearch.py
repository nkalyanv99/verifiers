import base64
import csv
from pathlib import Path

from datasets import Dataset
import verifiers as vf


def xor_decrypt(data: bytes, key: str) -> bytes:
    """XOR decrypt data with a cyclic key."""
    key_bytes = key.encode("utf-8")
    return bytes([data[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data))])


def load_deepsearch_dataset(filepath: str | Path) -> Dataset:
    """
    Load and decrypt a DeepSearch CSV file.

    Args:
        filepath: Path to DeepSearch CSV (e.g., DeepSearch-2505.csv)

    Returns:
        HuggingFace Dataset with columns:
        - example_id: int (unique identifier)
        - question: str (decrypted prompt)
        - answer: str (decrypted answer)
        - reference_steps: str | None (decrypted, if present)
        - id: str (original ID from CSV)
    """
    filepath = Path(filepath)

    with open(filepath, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    processed = []
    for i, row in enumerate(rows):
        key = row["canary"]

        # Decrypt prompt
        question = xor_decrypt(base64.b64decode(row["prompt"]), key).decode("utf-8")

        # Decrypt answer
        answer = xor_decrypt(base64.b64decode(row["answer"]), key).decode("utf-8")

        # Decrypt reference_steps if present
        reference_steps = None
        if "reference_steps" in row and row["reference_steps"]:
            reference_steps = xor_decrypt(
                base64.b64decode(row["reference_steps"]), key
            ).decode("utf-8")

        processed.append(
            {
                "example_id": i,
                "id": row["id"],
                "question": question,
                "answer": answer,
                "reference_steps": reference_steps,
            }
        )

    return Dataset.from_list(processed)


def load_environment(max_turns: int = 10, **kwargs) -> vf.Environment:
    # TODO: Load and decrypt DeepSearch-2510 dataset
    dataset = None

    # TODO: Define search tools
    tools = []

    # TODO: Setup rubrics (ToolRubric + JudgeRubric)
    rubric = None

    return vf.ToolEnv(
        dataset=dataset,
        system_prompt="",
        tools=tools,
        rubric=rubric,
        max_turns=max_turns,
    )
