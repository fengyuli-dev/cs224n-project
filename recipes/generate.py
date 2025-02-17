import itertools
import random
import sys
import time
import json  # Still available if needed for other purposes
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.data import Message, Role
from torchtune.training import FullModelTorchTuneCheckpointer
from tqdm import tqdm

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    Adapted for evaluating accuracy on the MMLU benchmark using Hugging Face datasets.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)

        if self._quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in self._quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )

        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)
            for k, v in model_state_dict.items():
                model_state_dict[k] = v.to(self._device)
            model.load_state_dict(model_state_dict, assign=True)
        else:
            model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Dict[Role, str],
    ) -> List[int]:
        """
        Convert the prompt string to a user message with optional system messages
        and tokenize using the prompt template defined on the tokenizer.
        """
        messages = []
        if "system" in prompt and prompt["system"] is not None:
            messages.append(Message(role="system", content=prompt["system"]))
        messages.extend(
            [
                Message(role="user", content=prompt["user"]),
                # Message(role="assistant", content=prompt.get("assistant", "")),
            ]
        )
        tokenized = self._tokenizer({"messages": messages}, inference=True)["tokens"]
        tokenized += self._tokenizer.encode("<|im_start|>assistant\n>")
        return tokenized

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        tokens = self.convert_prompt_to_tokens(cfg.prompt)
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        if cfg.enable_kv_cache:
            with self._device:
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens,
                )

        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                generation.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = generation.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")
            self._model.reset_caches()

        t0 = time.perf_counter()
        generated_tokens, _ = generation.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            pad_id=self._tokenizer.pad_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
        generated_tokens = generated_tokens.tolist()
        t = time.perf_counter() - t0

        output_text = self._tokenizer.decode(generated_tokens[0])
        logger.info(output_text)

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            logger.info(
                f"Memory used: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )

    @torch.inference_mode()
    def evaluate_mmlu(self, cfg: DictConfig):
        """
        Evaluate the model's accuracy on the MMLU benchmark accessed from Hugging Face.
        Assumes that the dataset returns examples with the keys:
            - "question": The question text.
            - "choices": Either a dict mapping choice labels to option texts or a list of options.
            - "answer": The correct choice label.
        Optionally, if "choices" is a list, it is converted to a dict using A, B, C, etc.
        Incorporates 3-shot demonstration examples.
        """

        if cfg.enable_kv_cache:
            with self._device:
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=8192,
                )

        from datasets import load_dataset

        # Load the MMLU dataset from Hugging Face.
        dataset = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
        # dataset = load_dataset("cais/mmlu", "all", split="validation")
        total_questions = len(dataset)
        correct = 0

        cot_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <think> {thought with steps separated with '\n\n'} <think/> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <answer> {final formatted, precise, and clear solution} <answer/> Now, try to solve the following question through the above guidelines:"

        sys_prompt = "You are an expert who knows everything, you are tasked to answer the following multiple-choice question. Give your final answer in the format of 'The answer is (chosen multiple-choice option)'."

        custom_generate_next_token = None
        if self._quantization_mode is not None:
            custom_generate_next_token = torch.compile(
                generation.generate_next_token, mode="max-autotune", fullgraph=True
            )
            dummy_prompt = torch.tensor([0, 1, 2], dtype=torch.int, device=self._device)
            _ = generation.generate(
                model=self._model,
                prompt=dummy_prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            self._model.reset_caches()

        logger.info(f"Starting MMLU evaluation over {total_questions} questions.")
        start_time = time.perf_counter()

        for idx, example in tqdm(enumerate(dataset)):
            self._model.reset_caches()

            question = example["question"]
            choices = example["choices"]
            answer = example["answer"]
            answer = chr(ord("A") + int(answer))

            # If choices is a list, convert it to a dict with keys A, B, C, ...
            if isinstance(choices, list):
                labels = list("ABCD")
                choices = {labels[i]: choice for i, choice in enumerate(choices)}

            # Build the prompt with 3-shot demonstration.
            # prompt_text = few_shot_prompt
            prompt_text = ""
            prompt_text += f"Question: {question}\nOptions:\n"
            for label, option in choices.items():
                prompt_text += f"{label}. {option}\n"
            prompt_text += "Answer: "
            prompt_dict = {
                "user": prompt_text,
            }

            prompt_dict["system"] = cot_prompt
            tokens = self.convert_prompt_to_tokens(prompt_dict)
            prompt_tensor = torch.tensor(tokens, dtype=torch.int, device=self._device)

            generated_tokens, generated_logits = generation.generate(
                model=self._model,
                prompt=prompt_tensor,
                max_generated_tokens=cfg.max_new_tokens,
                pad_id=self._tokenizer.pad_id,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            generated_tokens = generated_tokens.tolist()[0]
            output_text = self._tokenizer.decode(generated_tokens)

            logits = generated_logits[0, :, :]
            choices_token_ids = [
                self._tokenizer.encode(label) for label in choices.keys()
            ]
            choice_logits = logits[:, choices_token_ids]
            choice_logits = choice_logits.amax(dim=0)
            predicted = list(choices.keys())[torch.argmax(choice_logits).item()]

            is_correct = predicted.strip().upper() == str(answer).strip().upper()
            if is_correct:
                correct += 1

            logger.info(
                f"Q{idx+1}: True Answer: {answer} | Model Answer: {predicted} | {'Correct' if is_correct else 'Incorrect'}"
            )

        total_time = time.perf_counter() - start_time
        accuracy = correct / total_questions * 100
        logger.info(
            f"MMLU Evaluation completed: {accuracy:.2f}% accuracy over {total_questions} questions in {total_time:.2f} sec."
        )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    # If evaluation mode is enabled, run MMLU evaluation; otherwise, do standard generation.
    if cfg.get("mmlu", False):
        recipe.evaluate_mmlu(cfg=cfg)
    else:
        recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
