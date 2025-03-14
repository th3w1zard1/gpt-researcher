from __future__ import annotations

import tiktoken

# Per OpenAI Pricing Page: https://openai.com/api/pricing/
ENCODING_MODEL: str = "o200k_base"
INPUT_COST_PER_TOKEN: float = 0.000005
OUTPUT_COST_PER_TOKEN: float = 0.000015
IMAGE_INFERENCE_COST: float = 0.003825
EMBEDDING_COST: float = 0.02 / 1000000  # Assumes new ada-3-small


# Cost estimation is via OpenAI libraries and models. May vary for other models
def estimate_llm_cost(
    input_content: str,
    output_content: str,
) -> float:
    encoding: tiktoken.Encoding = tiktoken.get_encoding(ENCODING_MODEL)
    input_tokens: list[int] = encoding.encode(input_content)
    output_tokens: list[int] = encoding.encode(output_content)
    input_costs: float = len(input_tokens) * INPUT_COST_PER_TOKEN
    output_costs: float = len(output_tokens) * OUTPUT_COST_PER_TOKEN
    return input_costs + output_costs


def estimate_embedding_cost(
    model: str,
    docs: list[str],
) -> float:
    encoding: tiktoken.Encoding = tiktoken.encoding_for_model(model)
    total_tokens: int = sum(len(encoding.encode(str(doc))) for doc in docs)
    return total_tokens * EMBEDDING_COST
