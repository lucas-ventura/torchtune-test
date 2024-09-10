from pathlib import Path

from omegaconf import OmegaConf


def get_cfg_model(
    model_name="llama31",
):
    ver_u = "3_1" if model_name == "llama31" else "3"

    model_dict = {
        "_component_": f"torchtune.models.llama{ver_u}.llama{ver_u}_8b",
    }
    model_cfg = OmegaConf.create(model_dict)
    return model_cfg


def get_cfg_checkpointer(
    checkpoint_dir,
    model_name="llama31",
    quantization="",
):
    if quantization == "":
        return get_cfg_checkpointer_base(
            model_name=model_name, checkpoint_dir=checkpoint_dir
        )

    return get_cfg_checkpointer_quantization(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        quantization=quantization,
    )


def get_cfg_checkpointer_base(
    checkpoint_dir,
    model_name="llama31",
):
    ver_p = "3.1" if model_name == "llama31" else "3"

    model_dir = f"{checkpoint_dir}/Meta-Llama-{ver_p}-8B-Instruct/"
    assert Path(model_dir).exists(), f"Model directory {model_dir} does not exist."

    checkpointer_dict = {
        "_component_": "torchtune.utils.FullModelMetaCheckpointer",
        "checkpoint_dir": model_dir,
        "checkpoint_files": ["consolidated.00.pth"],
        "output_dir": f"{checkpoint_dir}/finetuned/Meta-Llama-{ver_p}-8B-Instruct/",
        "model_type": "LLAMA3",
    }
    checkpointer_cfg = OmegaConf.create(checkpointer_dict)
    return checkpointer_cfg


def get_cfg_checkpointer_quantization(
    checkpoint_dir,
    model_name="llama31",
    quantization="4w",
):
    ver_p = "3.1" if model_name == "llama31" else "3"

    model_dir = f"{checkpoint_dir}/finetuned/Meta-Llama-{ver_p}-8B-Instruct/"
    assert Path(model_dir).exists(), f"Model directory {model_dir} does not exist."

    checkpointer_dict = {
        "_component_": "torchtune.utils.FullModelTorchTuneCheckpointer",
        "checkpoint_dir": model_dir,
        "checkpoint_files": [f"consolidated-{quantization}.pt"],
        "output_dir": f"{checkpoint_dir}/finetuned/Meta-Llama-{ver_p}-8B-Instruct/",
        "model_type": "LLAMA3",
    }
    checkpointer_cfg = OmegaConf.create(checkpointer_dict)
    return checkpointer_cfg


def get_cfg_tokenizer(
    checkpoint_dir,
    model_name="llama31",
):
    ver_p = "3.1" if model_name == "llama31" else "3"

    model_dir = f"{checkpoint_dir}/Meta-Llama-{ver_p}-8B-Instruct/"
    assert Path(model_dir).exists(), f"Model directory {model_dir} does not exist."

    tokenizer_dict = {
        "_component_": "torchtune.models.llama3.llama3_tokenizer",
        "path": f"{model_dir}/tokenizer.model",
    }
    tokenizer_cfg = OmegaConf.create(tokenizer_dict)
    return tokenizer_cfg


def get_cfg_quantization(quantization=""):
    if quantization == "":
        return None
    elif quantization == "4w":
        quantizer_dict = {
            "_component_": "torchtune.utils.quantization.Int4WeightOnlyQuantizer",
            "groupsize": 256,
        }
        return OmegaConf.create(quantizer_dict)
    elif quantization == "8w":
        quantizer_dict = {
            "_component_": "torchtune.utils.quantization.Int8WeightOnlyQuantizer",
        }
        return OmegaConf.create(quantizer_dict)
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
