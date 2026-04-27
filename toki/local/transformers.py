import json
from dataclasses import asdict
from threading import Thread
from typing import Generator, Iterator, Literal, overload
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ..model import (
    BaseModel,
    TokiChatResponse,
    TokiMessage,
    TokiThinking,
    TokiToolCall,
    TokiToolFunction,
    TokiToolResponse,
    TokiUsageMetadata,
)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _ThinkingSplitter:
    """Streaming state machine: feed text chunks, yields ('content', s) or ('thought', s) tuples.

    Buffers boundary-straddling tokens like '<thi' until enough characters arrive to disambiguate
    a tag from regular text.
    """

    def __init__(self) -> None:
        self.buf: str = ""
        self.in_thought: bool = False

    def feed(self, chunk: str) -> Iterator[tuple[Literal['content', 'thought'], str]]:
        self.buf += chunk
        while self.buf:
            target, kind_after = (_THINK_OPEN, True) if not self.in_thought else (_THINK_CLOSE, False)
            kind: Literal['content', 'thought'] = 'thought' if self.in_thought else 'content'

            idx = self.buf.find(target)
            if idx != -1:
                emit = self.buf[:idx]
                if emit:
                    yield (kind, emit)
                self.buf = self.buf[idx + len(target):]
                self.in_thought = kind_after
                continue

            # no full tag in buf; emit everything except the trailing chars that could be the start
            # of a tag (we hold back up to len(target)-1 chars).
            hold = len(target) - 1
            safe_end = self._safe_emit_end(self.buf, target, hold)
            if safe_end > 0:
                emit = self.buf[:safe_end]
                self.buf = self.buf[safe_end:]
                yield (kind, emit)
            return

    def flush(self) -> Iterator[tuple[Literal['content', 'thought'], str]]:
        if self.buf:
            kind: Literal['content', 'thought'] = 'thought' if self.in_thought else 'content'
            yield (kind, self.buf)
            self.buf = ""

    @staticmethod
    def _safe_emit_end(buf: str, target: str, hold: int) -> int:
        # find the largest k <= hold such that the last k chars of buf are a prefix of target
        max_check = min(hold, len(buf))
        for k in range(max_check, 0, -1):
            if target.startswith(buf[-k:]):
                return len(buf) - k
        return len(buf)


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
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[False] = False, **kwargs) -> str: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolResponse: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse: ...
    @overload
    def _blocking_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[True], **kwargs) -> TokiChatResponse | TokiToolResponse: ...
    def _blocking_complete(
        self,
        messages: list[TokiMessage],
        tools: list | None = None,
        *,
        capture_thinking: bool = False,
        **kwargs,
    ) -> str | TokiChatResponse | TokiToolResponse:
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

        content_text, thought_text = self._split_thinking(response)

        if tools:
            tool_calls = self._extract_tool_calls(content_text, tools)
            if tool_calls:
                return TokiToolResponse(thought=thought_text if capture_thinking else content_text.split("{", 1)[0].strip(), tool_calls=tool_calls)

        if capture_thinking:
            return TokiChatResponse(content=content_text, thought=thought_text)
        return content_text

    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolResponse, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: None = None, *, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking, None, None]: ...
    @overload
    def _streaming_complete(self, messages: list[TokiMessage], tools: list, *, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolResponse, None, None]: ...
    def _streaming_complete(
        self,
        messages: list[TokiMessage],
        tools: list | None = None,
        *,
        capture_thinking: bool = False,
        **kwargs,
    ) -> Generator[str | TokiThinking | TokiToolResponse, None, None]:
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

        splitter = _ThinkingSplitter()
        content_chunks: list[str] = []
        for raw in streamer:
            for kind, s in splitter.feed(raw):
                if kind == 'thought':
                    if capture_thinking:
                        yield TokiThinking(text=s)
                else:
                    content_chunks.append(s)
                    yield s
        for kind, s in splitter.flush():
            if kind == 'thought':
                if capture_thinking:
                    yield TokiThinking(text=s)
            else:
                content_chunks.append(s)
                yield s

        worker.join()
        if errors:
            raise errors[0]

        full_content = "".join(content_chunks)
        # also account for stripped thinking in the usage tally so callers see total work done
        self._set_usage_metadata(prompt_tokens=inputs["input_ids"].shape[-1], completion_text=full_content)
        if tools:
            tool_calls = self._extract_tool_calls(full_content, tools)
            if tool_calls:
                thought = full_content.split("{", 1)[0].strip()
                yield TokiToolResponse(thought=thought, tool_calls=tool_calls)

    def _build_prompt(self, messages: list[TokiMessage], *, tools: list | None = None) -> str:
        template_kwargs: dict = {}
        if tools:
            template_kwargs["tools"] = tools
        wire_messages = [self._msg_to_wire(m) for m in messages]
        return self.tokenizer.apply_chat_template(
            wire_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    @staticmethod
    def _msg_to_wire(m: TokiMessage) -> dict:
        out: dict = {"role": m.role, "content": m.content}
        if m.tool_calls is not None:
            out["tool_calls"] = [asdict(tc) for tc in m.tool_calls]
        if m.tool_call_id is not None:
            out["tool_call_id"] = m.tool_call_id
        return out

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """Run the splitter eagerly over a complete string. Returns (content, thought)."""
        splitter = _ThinkingSplitter()
        content_parts: list[str] = []
        thought_parts: list[str] = []
        for kind, s in splitter.feed(text):
            (thought_parts if kind == 'thought' else content_parts).append(s)
        for kind, s in splitter.flush():
            (thought_parts if kind == 'thought' else content_parts).append(s)
        return "".join(content_parts), "".join(thought_parts)

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

            tool_calls.extend(self._collect_tool_calls(payload, allowed_names))

            idx = next_open + end

        return tool_calls

    def _collect_tool_calls(self, payload: object, allowed_names: set[str]) -> list[TokiToolCall]:
        if isinstance(payload, list):
            tool_calls: list[TokiToolCall] = []
            for item in payload:
                tool_calls.extend(self._collect_tool_calls(item, allowed_names))
            return tool_calls

        if not isinstance(payload, dict):
            return []

        if "function_call" in payload:
            return self._collect_tool_calls(payload["function_call"], allowed_names)

        if "tool_calls" in payload:
            return self._collect_tool_calls(payload["tool_calls"], allowed_names)

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
                function=TokiToolFunction(name=name, arguments=arguments_json),
            )
        ]
