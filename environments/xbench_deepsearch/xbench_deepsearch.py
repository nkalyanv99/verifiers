import base64
import csv
import os
import re
from pathlib import Path

from datasets import Dataset
from openai import OpenAI
import verifiers as vf


###################################################
# Grading Prompt Templates (copied from xbench-evals/eval_grader.py)
###################################################

# English translation of LLM_JUDGE_PROMPT:
# """
# You are a general AI assistant. Based on the [Correct Answer] given below,
# determine whether the following [Response] to the [Original Question] is correct.
#
# [Original Question]: {question}
# [Correct Answer]: {correct_answer}
# [Response]: {response}
#
# Your judgment must follow this format and criteria:
#
# Final Answer: The final precise answer extracted from [Response].
#               If no clear final answer exists, write 'None'.
#
# Explanation: Based on [Correct Answer], explain why [Final Answer] is correct
#              or incorrect. Focus only on substantive differences between
#              [Final Answer] and [Correct Answer]. Do not comment on background,
#              do not re-solve the problem, do not defend any answer different
#              from [Correct Answer]. Only focus on judging answer consistency.
#
# Conclusion: If [Final Answer] matches [Correct Answer], or for numerical
#             questions falls within acceptable margin of error, write 'Correct';
#             otherwise (any inconsistency, ambiguity, inequivalence, or
#             extraction error) write 'Incorrect'.
# """

LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


###################################################
# Grading Functions (copied from xbench-evals/eval_grader.py)
###################################################


def get_judge_response(
    prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
) -> str | None:
    """Call Gemini via OpenAI-compatible API for judge evaluation."""
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Judge API error: {e}")
        return None


def parse_match_result(match):
    """Parse regex match result to extract answer after colon."""
    if match is None:
        return match

    match = match.group(0)

    try:
        target = match.split(":")[1].strip()
        return target
    except Exception:
        return match  # return naive result in case of failure


def grade_question(
    question_text: str,
    correct_answer: str,
    llm_response: str | None,
    judge_model: str = "gemini-2.5-flash",
) -> tuple[int, str, str]:
    """
    Grade a question response using two-stage grading:
    1. Exact pattern match (fast path)
    2. LLM judge fallback (for ambiguous cases)

    Args:
        question_text: The original question
        correct_answer: The expected correct answer
        llm_response: The model's response to grade
        judge_model: Model to use for LLM judge (default: Gemini 2.5 Flash)

    Returns:
        Tuple of (score, extracted_answer, explanation)
        - score: 1 if correct, 0 if incorrect
        - extracted_answer: The answer extracted from response
        - explanation: Grading explanation
    """
    if llm_response is None:
        return 0, "", ""

    # Stage 1: If there's direct match, do not need LLM judge
    simple_match = re.search(r"最终答案:*(.*)", llm_response)
    simple_match = parse_match_result(simple_match)
    if simple_match == correct_answer:
        return 1, simple_match, "答案完全正确, 无需调用LLM Judge"

    # Stage 2: Otherwise, use LLM Judge
    judge_prompt = LLM_JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=llm_response,
    )

    judge_response = get_judge_response(judge_prompt, model=judge_model)
    if judge_response is None:
        return 0, "", "Judge Response error"

    # Extract grader conclusions
    extract_match = re.search(r"最终答案:*(.*)", judge_response)
    extract_match = parse_match_result(extract_match)

    correct_match = re.search(r"结论:*.(正确|错误)", judge_response)
    correct_match = parse_match_result(correct_match)

    explain_match = re.search(r"解释:*(.*)", judge_response)
    explain_match = parse_match_result(explain_match)

    score = 1 if (correct_match == "正确") else 0

    return score, extract_match, explain_match


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
    # Load and decrypt DeepSearch-2510 dataset
    data_path = (
        Path(__file__).parent.parent.parent
        / "xbench-evals"
        / "data"
        / "DeepSearch-2510.csv"
    )
    dataset = load_deepsearch_dataset(data_path)

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
