# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torchtune import config, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

from cfgs import (
    get_cfg_checkpointer,
    get_cfg_model,
    get_cfg_quantization,
    get_cfg_tokenizer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = utils.get_logger("DEBUG")


class LLaMAInference:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, device, dtype, quantizer=None, seed=1234) -> None:
        self._device = utils.get_device(device=device)
        self._dtype = utils.get_dtype(dtype=dtype)
        self._quantizer = config.instantiate(quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=seed)

    def setup(
        self,
        checkpointer_cfg,
        model_cfg,
        tokenizer_cfg,
        enable_kv_cache=False,
    ) -> None:
        _checkpointer = config.instantiate(checkpointer_cfg)
        if self._quantization_mode is None:
            ckpt_dict = _checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = _checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=model_cfg,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            enable_kv_cache=enable_kv_cache,
        )
        self._tokenizer = config.instantiate(tokenizer_cfg)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        enable_kv_cache: bool = True,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        if enable_kv_cache:
            with self._device:
                model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=add_bos, add_eos=add_eos)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        chat_format=None,
        instruct_template=None,
        temperature=0.6,
        top_k=300,
        max_new_tokens=300,
        use_formatted_prompt=True,
    ) -> None:
        if use_formatted_prompt:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            # prompt = f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

        tokens = self.convert_prompt_to_tokens(prompt, chat_format, instruct_template)
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                utils.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = utils.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=temperature,
                top_k=top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                pad_id=self._tokenizer.pad_id,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            pad_id=self._tokenizer.pad_id,
            custom_generate_next_token=custom_generate_next_token,
        )

        # Decode only the generated tokens
        generated_tokens = generated_tokens[0][len(tokens) :]
        output = self._tokenizer.decode(generated_tokens, truncate_at_eos=True)

        # remove everything after the first special token
        special_tokens = list(self._tokenizer.special_tokens.keys())
        for token in special_tokens:
            if token in output:
                output = output.split(token)[0]

        return output


def get_tune_model(checkpoint_dir, dtype="bf16", model_name="llama31", quantization=""):
    quantization_cfg = get_cfg_quantization(quantization)
    tokenizer_cfg = get_cfg_tokenizer(
        model_name=model_name, checkpoint_dir=checkpoint_dir
    )
    checkpointer_cfg = get_cfg_checkpointer(
        model_name=model_name,
        quantization=quantization,
        checkpoint_dir=checkpoint_dir,
    )
    model_cfg = get_cfg_model(model_name=model_name)

    model = LLaMAInference(device=device, dtype=dtype, quantizer=quantization_cfg)

    model.setup(
        checkpointer_cfg=checkpointer_cfg,
        model_cfg=model_cfg,
        tokenizer_cfg=tokenizer_cfg,
        enable_kv_cache=True,
    )

    return model
