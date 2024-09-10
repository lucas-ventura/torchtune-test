import lorem
from tqdm import tqdm

from inference import get_tune_model


def main(
    checkpoint_dir,
    model_name,
    max_new_tokens,
    dtype,
    quantization="",
):
    experiment_name = model_name
    if quantization:
        experiment_name += f"-{quantization}"
    if dtype:
        experiment_name += f"-{dtype}"

    print("Loading model...", end="")
    model = get_tune_model(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        dtype=dtype,
        quantization=quantization,
    )
    print("done")

    for _ in tqdm(range(100)):
        _ = model.generate(prompt=lorem.text(), max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_name",
        default="llama31",
        choices=["llama31", "llama3"],
    )
    parser.add_argument("--checkpoint_dir", default="/tmp/")
    parser.add_argument("--max_new_tokens", default=512)
    parser.add_argument("--dtype", default=None, choices=[None, "bf16", "fp32"])
    parser.add_argument("-Q", "--quantization", default="", choices=["", "4w"])

    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        quantization=args.quantization,
    )
