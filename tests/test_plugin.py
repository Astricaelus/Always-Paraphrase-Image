from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot.core.agent.message import ImageURLPart, Message, TextPart
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.utils import image_input
from astrbot.core.utils.image_input import prepare_request_images


@pytest.fixture(scope="module")
def plugin_module():
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location(
        "always_paraphrase_image_main",
        path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CaptionProvider(Provider):
    def __init__(self, captions=None, error=None, modalities=None):
        super().__init__(
            {
                "type": "openai_chat_completion",
                "id": "caption",
                "modalities": modalities or ["text", "image"],
            },
            {},
        )
        self.captions = list(captions or [])
        self.error = error
        self.calls = []

    def get_current_key(self):
        return "key"

    def set_key(self, key):
        self.key = key

    async def get_models(self):
        return ["caption-model"]

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        caption = (
            self.captions.pop(0) if self.captions else f"caption {len(self.calls)}"
        )
        return LLMResponse(role="assistant", completion_text=caption)


class FakeContext:
    def __init__(
        self,
        *,
        provider=None,
        current_provider=None,
        provider_id="caption",
        core_provider_id=None,
        prompt="Describe.",
    ):
        self.provider = provider
        self.current_provider = current_provider
        self.provider_id = provider_id
        self.core_provider_id = (
            provider_id if core_provider_id is None else core_provider_id
        )
        self.prompt = prompt
        self.requested_ids = []
        self.providers = {}
        if provider is not None and self.core_provider_id:
            self.providers[self.core_provider_id] = provider

    def get_provider_by_id(self, provider_id):
        self.requested_ids.append(provider_id)
        return self.providers.get(provider_id)

    def get_config(self, umo=None):
        return {
            "provider_settings": {
                "default_image_caption_provider_id": self.core_provider_id,
                "image_caption_prompt": self.prompt,
            }
        }

    async def get_using_provider_async(self, umo=None):
        return self.current_provider


def make_plugin(
    plugin_module,
    *,
    provider=None,
    current_provider=None,
    provider_id="caption",
    core_provider_id=None,
    prompt="Describe.",
    config=None,
):
    context = FakeContext(
        provider=provider,
        current_provider=current_provider,
        provider_id=provider_id,
        core_provider_id=core_provider_id,
        prompt=prompt,
    )
    plugin = plugin_module.AlwaysParaphraseImage(context, config or {})
    return plugin, context


def make_event():
    return SimpleNamespace(unified_msg_origin="test:friend:1")


def image_refs_in(value):
    """Return all image_url blocks nested in an OpenAI-style structure."""
    found = []
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            found.append(value)
        for child in value.values():
            found.extend(image_refs_in(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(image_refs_in(child))
    return found


@pytest.mark.asyncio
async def test_current_images_become_caption_text(plugin_module):
    refs = ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]
    provider = CaptionProvider(["a cat", "a dog"])
    plugin, _ = make_plugin(plugin_module, provider=provider)
    req = ProviderRequest(prompt="what are these?", image_urls=refs)

    await plugin.paraphrase_images(make_event(), req)

    assert req.image_urls == []
    assert req.prompt.startswith("what are these?")
    assert "Image 1: a cat" in req.prompt
    assert "Image 2: a dog" in req.prompt
    assert len(provider.calls) == 2
    assert all(call["prompt"] == "Describe." for call in provider.calls)


@pytest.mark.asyncio
async def test_history_images_are_replaced_and_text_only(plugin_module):
    first = "data:image/png;base64,AAAA"
    second = "data:image/png;base64,BBBB"
    provider = CaptionProvider(["first caption", "second caption"])
    plugin, _ = make_plugin(plugin_module, provider=provider)
    req = ProviderRequest(
        contexts=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": first}},
                    {"type": "text", "text": "and this"},
                    {"type": "image_url", "image_url": {"url": second}},
                ],
            },
            {"role": "assistant", "content": "ok"},
        ]
    )

    await plugin.paraphrase_images(make_event(), req)

    assert image_refs_in(req.contexts) == []
    text = json.dumps(req.contexts, ensure_ascii=False)
    assert "first caption" in text
    assert "second caption" in text
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_caption_cache_avoids_duplicate_provider_calls(plugin_module):
    image = "data:image/png;base64,AAAA"
    provider = CaptionProvider(["cached caption"])
    plugin, _ = make_plugin(plugin_module, provider=provider)

    first_req = ProviderRequest(image_urls=[image])
    await plugin.paraphrase_images(make_event(), first_req)

    second_req = ProviderRequest(
        contexts=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ]
    )
    await plugin.paraphrase_images(make_event(), second_req)

    assert len(provider.calls) == 1
    assert "cached caption" in json.dumps(second_req.contexts)


@pytest.mark.asyncio
async def test_caption_failure_fails_closed(plugin_module):
    provider = CaptionProvider(error=RuntimeError("boom"))
    plugin, _ = make_plugin(plugin_module, provider=provider)
    req = ProviderRequest(image_urls=["data:image/png;base64,AAAA"])

    await plugin.paraphrase_images(make_event(), req)

    assert req.image_urls == []
    assert plugin_module.FAILED_CAPTION_TEXT in (req.prompt or "")


@pytest.mark.asyncio
async def test_missing_provider_fails_closed(plugin_module):
    plugin, context = make_plugin(
        plugin_module,
        provider=None,
        current_provider=None,
        provider_id="",
    )
    req = ProviderRequest(image_urls=["data:image/png;base64,AAAA"])

    await plugin.paraphrase_images(make_event(), req)

    assert req.image_urls == []
    assert plugin_module.MISSING_PROVIDER_TEXT in (req.prompt or "")
    assert context.requested_ids == []


@pytest.mark.asyncio
async def test_falls_back_to_image_capable_session_provider(plugin_module):
    session_provider = CaptionProvider(["session caption"])
    plugin, _ = make_plugin(
        plugin_module,
        provider=None,
        current_provider=session_provider,
        provider_id="",
    )
    req = ProviderRequest(image_urls=["data:image/png;base64,AAAA"])

    await plugin.paraphrase_images(make_event(), req)

    assert len(session_provider.calls) == 1
    assert "session caption" in (req.prompt or "")


@pytest.mark.asyncio
async def test_falls_back_to_core_caption_provider(plugin_module):
    provider = CaptionProvider(["core caption"])
    plugin, context = make_plugin(
        plugin_module,
        provider=provider,
        provider_id="stale-plugin-setting",
        core_provider_id="caption",
        config={"caption_provider_id": "stale-plugin-setting"},
    )
    req = ProviderRequest(image_urls=["data:image/png;base64,AAAA"])

    await plugin.paraphrase_images(make_event(), req)

    assert len(provider.calls) == 1
    assert context.requested_ids == ["stale-plugin-setting", "caption"]
    assert "core caption" in (req.prompt or "")


@pytest.mark.asyncio
async def test_blank_plugin_prompt_falls_back_to_core_prompt(plugin_module):
    provider = CaptionProvider(["prompt caption"])
    plugin, _ = make_plugin(
        plugin_module,
        provider=provider,
        config={"caption_prompt": "   "},
    )
    req = ProviderRequest(image_urls=["data:image/png;base64,AAAA"])

    await plugin.paraphrase_images(make_event(), req)

    assert provider.calls[0]["prompt"] == "Describe."


@pytest.mark.asyncio
async def test_extra_image_part_preserves_temporary_flag(plugin_module):
    provider = CaptionProvider(["extra caption"])
    plugin, _ = make_plugin(plugin_module, provider=provider)
    image_part = ImageURLPart(
        image_url=ImageURLPart.ImageURL(url="data:image/png;base64,AAAA")
    ).mark_as_temp()
    req = ProviderRequest(extra_user_content_parts=[image_part])

    await plugin.paraphrase_images(make_event(), req)

    assert len(req.extra_user_content_parts) == 1
    new_part = req.extra_user_content_parts[0]
    assert isinstance(new_part, TextPart)
    assert new_part.model_dump_for_context()["_no_save"] is True
    assert "extra caption" in new_part.text


@pytest.mark.asyncio
async def test_no_images_does_not_call_provider(plugin_module):
    provider = CaptionProvider()
    plugin, _ = make_plugin(plugin_module, provider=provider)
    req = ProviderRequest(prompt="hello")

    await plugin.paraphrase_images(make_event(), req)

    assert provider.calls == []
    assert req.prompt == "hello"


@pytest.mark.asyncio
async def test_malformed_image_part_is_removed(plugin_module):
    provider = CaptionProvider()
    plugin, _ = make_plugin(plugin_module, provider=provider)
    req = ProviderRequest(extra_user_content_parts=[{"type": "image_url"}])

    await plugin.paraphrase_images(make_event(), req)

    assert image_refs_in(req.extra_user_content_parts) == []
    assert isinstance(req.extra_user_content_parts[0], TextPart)
    assert plugin_module.FAILED_CAPTION_TEXT in req.extra_user_content_parts[0].text


@pytest.mark.asyncio
async def test_message_objects_in_context_are_rewritten(plugin_module):
    provider = CaptionProvider(["message caption"])
    plugin, _ = make_plugin(plugin_module, provider=provider)
    message = Message(
        role="user",
        content=[
            TextPart(text="before"),
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,AAAA")
            ),
        ],
    )
    req = ProviderRequest(contexts=[message])

    await plugin.paraphrase_images(make_event(), req)

    assert all(part.type != "image_url" for part in message.content)
    assert any(
        isinstance(part, TextPart) and "message caption" in part.text
        for part in message.content
    )


@pytest.mark.asyncio
async def test_request_stays_text_only_after_pipeline_prepare(
    plugin_module,
    tmp_path,
    monkeypatch,
):
    async def fake_prepare_model_image(image_ref, *, max_size, output_dir):
        return image_ref, False, False, image_ref

    # Keep the test focused on request rewriting. Some sandboxes deadlock when
    # AstrBot's image preparation hands Pillow work to an executor thread.
    monkeypatch.setattr(
        image_input,
        "prepare_model_image",
        fake_prepare_model_image,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"not-decoded-in-this-test")

    class EventStub:
        unified_msg_origin = "test:friend:1"

        def __init__(self):
            self.temporary_files = []

        def track_temporary_local_file(self, path):
            self.temporary_files.append(path)

    event = EventStub()
    req = ProviderRequest(prompt="look", image_urls=[str(source)])
    prepared = {}
    await prepare_request_images(
        req,
        event,
        max_size=1024,
        output_dir=tmp_path,
        prepared=prepared,
        finalize=True,
        supports_image=True,
    )
    assert req.image_urls

    provider = CaptionProvider(["pipeline caption"])
    plugin, _ = make_plugin(plugin_module, provider=provider)
    await plugin.paraphrase_images(event, req)

    # The internal agent pipeline prepares images once more after the hook.
    await prepare_request_images(
        req,
        event,
        max_size=1024,
        output_dir=tmp_path,
        prepared=prepared,
        finalize=True,
        supports_image=True,
    )

    assert req.image_urls == []
    assert image_refs_in(req.extra_user_content_parts) == []
    assert "pipeline caption" in (req.prompt or "")
    assembled = await req.assemble_context()
    assert image_refs_in(assembled) == []
    assert "pipeline caption" in json.dumps(assembled, ensure_ascii=False)
