# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
)

SPECIAL_TOKENS = {
    "<|im_start|>": 1,
    "<|im_end|>": 2,
    "<|finetune_right_pad_id|>": 0,
}


def smollm2_tokenizer():
    return SmolLM2Tokenizer()


class SmolLM2Tokenizer(ModelTokenizer, Transform):

    def __init__(
        self,
    ):
        self.special_tokens = SPECIAL_TOKENS

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<|im_start|>"]
        self.eos_id = self.special_tokens["<|im_end|>"]
        self.pad_id = self.special_tokens["<|finetune_right_pad_id|>"]

        # During generation, stop when either eos_id, eot_id, or eom_id is encountered
        self.stop_tokens = [self.eos_id]

        from transformers import AutoTokenizer

        checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.max_seq_len = 8192

    @property
    def base_vocab_size(self) -> int:
        return 49152

    @property
    def vocab_size(self) -> int:
        return 49152

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tokenizer.encode(text, return_tensors="pt").squeeze().tolist()

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        # Flatten messages by joining their content into a single string.
        combined_input = ""
        for message in messages:
            combined_input = (
                combined_input
                + "<|im_start|>"
                + message.role
                + "\n"
                + message.content[0]["content"]
                + "<|im_end|>"
                + "\n"
            )
        # Tokenize the combined string using the Hugging Face tokenizer.
        token_ids = self.tokenizer.encode(combined_input, add_special_tokens=False)

        # Prepend the beginning-of-sequence token.
        tokens = [self.bos_id] + token_ids
        mask = [True] * len(tokens)

        # Append the end-of-sequence token if required.
        if add_end_tokens:
            tokens.append(self.eos_id)
            mask.append(True)

        # Truncate tokens and mask to max_seq_len if needed.
        if self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
            mask = mask[: self.max_seq_len]

        return tokens, mask

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    messages = [{"role": "user", "content": "What is the capital of France."}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    breakpoint()
