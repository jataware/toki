import json
from threading import Thread
from typing import Generator, overload
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ..model import BaseModel, TokiMessage, TokiToolCall, TokiToolResponse, TokiUsageMetadata


class LocalModel(BaseModel):
    """Local `transformers`-backed model with the same interface as other toki backends."""

    def __init__(self, model: str, allow_parallel_tool_calls: bool = False):
        super().__init__()
        self.model = model
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"'{model}' has no chat template; toki targets instruction-tuned chat models. "
                f"Try the '-it' / '-Instruct' variant of this model if one exists."
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self._model.to(self.device)
        self._model.eval()

    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, **kwargs) -> str | TokiToolResponse: ...
    def _blocking_complete(
        self,
        messages: list[TokiMessage],
        tools: list | None = None,
        **kwargs,
    ) -> str | TokiToolResponse:
        prompt = self._build_prompt(messages, tools=tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = self._build_generate_kwargs(
            input_token_count=inputs["input_ids"].shape[-1],
            **kwargs,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self._set_usage_metadata(prompt_tokens=inputs["input_ids"].shape[-1], completion_text=response)
        return self._parse_response(response, tools)

    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    def _streaming_complete(
        self,
        messages: list[TokiMessage],
        tools: list | None = None,
        **kwargs,
    ) -> Generator[str | TokiToolResponse, None, None]:
        prompt = self._build_prompt(messages, tools=tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = self._build_generate_kwargs(
            input_token_count=inputs["input_ids"].shape[-1],
            **kwargs,
        )
        errors: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        **inputs,
                        **generation_kwargs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        streamer=streamer,
                    )
            except BaseException as exc:
                errors.append(exc)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()

        chunks: list[str] = []
        for chunk in streamer:
            chunks.append(chunk)
            yield chunk

        worker.join()
        if errors:
            raise errors[0]

        response = "".join(chunks)
        self._set_usage_metadata(prompt_tokens=inputs["input_ids"].shape[-1], completion_text=response)
        if tools:
            parsed_response = self._parse_response(response, tools)
            if isinstance(parsed_response, dict):
                yield parsed_response

    def _build_prompt(self, messages: list[TokiMessage], *, tools: list | None = None) -> str:
        template_kwargs = {}
        if tools:
            template_kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def _build_generate_kwargs(self, *, input_token_count: int, **kwargs) -> dict:
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.95)
        max_new_tokens = kwargs.pop("max_new_tokens", None)

        generate_kwargs = {
            "do_sample": temperature > 0,
            **kwargs,
        }
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        else:
            generate_kwargs["max_new_tokens"] = max(1, self.tokenizer.model_max_length - input_token_count)
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        return generate_kwargs

    def _set_usage_metadata(self, *, prompt_tokens: int, completion_text: str) -> None:
        completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))
        self._usage_metadata = TokiUsageMetadata(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _parse_response(self, response: str, tools: list | None) -> str | TokiToolResponse:
        if not tools:
            return response

        tool_calls = self._extract_tool_calls(response, tools)
        if not tool_calls:
            return response

        thought = response.split("{", 1)[0].strip()
        return TokiToolResponse(thought=thought, tool_calls=tool_calls)

    def _extract_tool_calls(self, response: str, tools: list) -> list[TokiToolCall]:
        allowed_names = {tool["function"]["name"] for tool in tools}
        decoder = json.JSONDecoder()
        tool_calls: list[TokiToolCall] = []
        idx = 0

        while idx < len(response):
            next_open = response.find("{", idx)
            if next_open == -1:
                break

            try:
                payload, end = decoder.raw_decode(response[next_open:])
            except json.JSONDecodeError:
                idx = next_open + 1
                continue

            tool_calls.extend(self._coerce_tool_calls(payload, allowed_names))

            idx = next_open + end

        return tool_calls

    def _coerce_tool_calls(self, payload: object, allowed_names: set[str]) -> list[TokiToolCall]:
        if isinstance(payload, list):
            tool_calls: list[TokiToolCall] = []
            for item in payload:
                tool_calls.extend(self._coerce_tool_calls(item, allowed_names))
            return tool_calls

        if not isinstance(payload, dict):
            return []

        if "function_call" in payload:
            return self._coerce_tool_calls(payload["function_call"], allowed_names)

        if "tool_calls" in payload:
            return self._coerce_tool_calls(payload["tool_calls"], allowed_names)

        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or name not in allowed_names or arguments is None:
            return []

        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments)

        return [
            TokiToolCall(
                id=f"local-tool-{uuid4().hex}",
                type="function",
                function={"name": name, "arguments": arguments_json},
            )
        ]
