import lorem
import torch
import transformers
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_message(content, add_system_prompt=False):
    prompt = []
    if add_system_prompt:
        prompt.append(
            {
                "role": "system",
                "content": (
                    "You are a highly skilled text analysis assistant capable of identifying thematic shifts in video transcripts. "
                    "Your task is to segment the given ASR transcript into distinct chapters based on changes in topics or themes. "
                ),
            }
        )
    prompt.append({"role": "user", "content": content})
    return prompt


def get_model(model_id: str):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    return pipeline


def get_output(pipeline, prompt: str):
    prompt = pipeline.tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        # add_special_tokens=True,
    )

    return outputs[0]["generated_text"][len(prompt) :]


def main(model_name):
    if model_name == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "llama31":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    print(f"Loading {model_name} model...")
    pipeline = get_model(model_name)

    for _ in tqdm(range(100)):
        prompt = format_message(lorem.text())
        _ = get_output(pipeline, prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language overlap")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama31",
        choices=[
            "llama3",
            "llama31",
        ],
    )
    args = parser.parse_args()

    main(
        args.model_name,
    )
