import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.mage_flow.model.mage_flow_text_encoder import (
    CONTENT_FILTER_EDIT_SYSTEM,
    CONTENT_FILTER_SYSTEM,
    MageFlowContentPolicy,
    make_refusal_image,
)


class _PolicyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, decoded: str):
        self.decoded = decoded
        self.messages = []
        self.tokenize_calls = []
        self.decoded_ids = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|image_pad|>": 3,
            "<|video_pad|>": 4,
        }[token]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages.append(messages)
        rendered = [f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"]
        rendered.append("<|im_start|>user\n")
        user_content = messages[1]["content"]
        if isinstance(user_content, str):
            rendered.append(user_content)
        else:
            for item in user_content:
                if item["type"] == "image":
                    rendered.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item["type"] == "text":
                    rendered.append(item["text"])
        rendered.append("<|im_end|>\n<|im_start|>assistant\n")
        return "".join(rendered)

    def __call__(
        self,
        texts,
        *,
        padding: bool,
        return_tensors: str,
        max_length: int | None = None,
        truncation: bool = False,
    ):
        self.tokenize_calls.append(
            {
                "texts": texts,
                "padding": padding,
                "return_tensors": return_tensors,
                "max_length": max_length,
                "truncation": truncation,
            }
        )
        sequences = []
        for text in texts:
            image_tokens = text.count("<|image_pad|>")
            sequences.append([1, *([3] * image_tokens), 6])
        maximum = max(len(sequence) for sequence in sequences)
        input_ids = np.array(
            [sequence + [self.pad_token_id] * (maximum - len(sequence)) for sequence in sequences],
            dtype=np.int32,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).astype(np.int32),
        }

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids
        return self.decoded


class _TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _PolicyTextEncoder:
    def __init__(self, generated_ids=(7, 2), error: Exception | None = None):
        self.generated_ids = generated_ids
        self.error = error
        self.generate_calls = []

    def generate_greedy(self, input_ids, **kwargs):
        self.generate_calls.append({"input_ids": input_ids, **kwargs})
        if self.error is not None:
            raise self.error
        return mx.array([self.generated_ids], dtype=mx.int32)


def test_mage_flow_text_policy_uses_exact_messages_untruncated_greedy_decode() -> None:
    tokenizer = _PolicyTokenizer(
        'preamble {"violates": true, "categories": ["copyright"], "reason": "named IP"} trailing'
    )
    text_encoder = _PolicyTextEncoder()

    verdict = MageFlowContentPolicy.screen_text(
        text_encoder=text_encoder,
        tokenizer=_TokenizerWrapper(tokenizer),
        prompt="Pikachu on a beach",
    )

    assert verdict.violates is True
    assert verdict.categories == ["copyright"]
    assert verdict.reason == "named IP"
    assert verdict.raw.startswith("preamble")
    assert tokenizer.messages == [
        [
            {"role": "system", "content": CONTENT_FILTER_SYSTEM},
            {"role": "user", "content": "Prompt to classify:\nPikachu on a beach"},
        ]
    ]
    assert tokenizer.tokenize_calls[0]["max_length"] is None
    assert tokenizer.tokenize_calls[0]["truncation"] is False
    assert text_encoder.generate_calls[0]["max_new_tokens"] == 160
    assert text_encoder.generate_calls[0]["eos_token_id"] == tokenizer.eos_token_id
    assert tokenizer.decoded_ids == [7, 2]


def test_mage_flow_edit_policy_uses_original_resolution_and_exact_multimodal_prompt() -> None:
    tokenizer = _PolicyTokenizer('{"violates": false, "categories": [], "reason": "ordinary image"}')
    text_encoder = _PolicyTextEncoder()
    reference = Image.new("RGB", (512, 256), color=(128, 128, 128))

    verdict = MageFlowContentPolicy.screen_edit(
        text_encoder=text_encoder,
        tokenizer=_TokenizerWrapper(tokenizer),
        prompt="change the background",
        ref_images=[reference],
    )

    assert verdict.violates is False
    assert verdict.reason == "ordinary image"
    messages = tokenizer.messages[0]
    assert messages[0] == {"role": "system", "content": CONTENT_FILTER_EDIT_SYSTEM}
    assert messages[1]["content"] == [
        {"type": "image"},
        {
            "type": "text",
            "text": (
                "There is 1 source image(s) above. Edit instruction: change the background\nClassify this edit request."
            ),
        },
    ]
    assert tokenizer.tokenize_calls[0]["max_length"] is None
    assert tokenizer.tokenize_calls[0]["truncation"] is False
    generate_call = text_encoder.generate_calls[0]
    np.testing.assert_array_equal(
        np.asarray(generate_call["image_grid_thw"]),
        np.array([[1, 16, 32]], dtype=np.int32),
    )
    assert int(mx.sum(generate_call["input_ids"] == 3).item()) == 128
    assert generate_call["max_new_tokens"] == 192


def test_mage_flow_policy_is_fail_closed_for_malformed_or_failed_generation() -> None:
    malformed = MageFlowContentPolicy.screen_text(
        text_encoder=_PolicyTextEncoder(),
        tokenizer=_PolicyTokenizer("not JSON"),
        prompt="a cat",
    )
    failed = MageFlowContentPolicy.screen_edit(
        text_encoder=_PolicyTextEncoder(error=RuntimeError("decode failed")),
        tokenizer=_PolicyTokenizer(""),
        prompt="change it",
        ref_images=[Image.new("RGB", (32, 32))],
    )

    assert malformed.violates is True
    assert malformed.categories == ["policy"]
    assert "no JSON object" in malformed.reason
    assert failed.violates is True
    assert failed.categories == ["policy"]
    assert failed.reason == "edit filter error (blocked): RuntimeError: decode failed"


def test_mage_flow_empty_text_policy_bypasses_generation_and_refusal_is_plain_white() -> None:
    text_encoder = _PolicyTextEncoder(error=AssertionError("must not generate"))

    verdict = MageFlowContentPolicy.screen_text(
        text_encoder=text_encoder,
        tokenizer=_PolicyTokenizer(""),
        prompt="  ",
    )
    refusal = make_refusal_image(verdict, height=32, width=48)

    assert verdict.violates is False
    assert verdict.reason == "empty prompt"
    assert text_encoder.generate_calls == []
    assert refusal.mode == "RGB"
    assert refusal.size == (48, 32)
    np.testing.assert_array_equal(np.asarray(refusal), np.full((32, 48, 3), 255, dtype=np.uint8))
