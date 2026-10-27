from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from astrbot.core import astr_main_agent
from astrbot.core.config.default import (
    CONFIG_METADATA_2,
    CONFIG_METADATA_3,
    DEFAULT_CONFIG,
)
from astrbot.core.config.i18n_utils import ConfigMetadataI18n
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider


@pytest.fixture(scope="module")
def plugin_modules():
    root = Path(__file__).resolve().parents[1]
    package_name = "always_paraphrase_image_test"
    package = ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    return (
        importlib.import_module(f"{package_name}.main"),
        importlib.import_module(f"{package_name}.integration"),
    )


class FakeConfigManager:
    def __init__(self) -> None:
        self.confs = {
            "default": {"provider_settings": {}},
            "other": {
                "provider_settings": {
                    "always_paraphrase_image": True,
                }
            },
        }


class FakeContext:
    def __init__(self) -> None:
        self.astrbot_config_mgr = FakeConfigManager()

    def get_config(self, umo=None):
        del umo
        return self.astrbot_config_mgr.confs["default"]


@pytest.fixture
def bridge(plugin_modules):
    _, integration = plugin_modules
    context = FakeContext()
    original_converter = ConfigMetadataI18n.convert_to_i18n_keys
    profile_bridge = integration.ProfileSchemaBridge(context)
    profile_bridge.install()
    try:
        yield profile_bridge, context, original_converter
    finally:
        profile_bridge.remove()


def ai_items():
    return CONFIG_METADATA_3["ai_group"]["metadata"]["ai"]["items"]


def validation_items():
    return CONFIG_METADATA_2["provider_group"]["metadata"]["provider_settings"]["items"]


def test_profile_schema_adds_toggle(bridge, plugin_modules):
    _, integration = plugin_modules
    _, context, _ = bridge

    keys = list(ai_items())
    assert integration.PROFILE_PATH in keys
    image_model_index = keys.index(
        "provider_settings.default_image_caption_provider_id"
    )
    assert keys[image_model_index + 1] == integration.PROFILE_PATH

    assert DEFAULT_CONFIG["provider_settings"][integration.PROFILE_KEY] is False
    assert (
        context.astrbot_config_mgr.confs["default"]["provider_settings"][
            integration.PROFILE_KEY
        ]
        is False
    )
    assert (
        context.astrbot_config_mgr.confs["other"]["provider_settings"][
            integration.PROFILE_KEY
        ]
        is True
    )
    assert validation_items()[integration.PROFILE_KEY] == {"type": "bool"}


def test_profile_schema_uses_raw_label_text(bridge, plugin_modules):
    _, integration = plugin_modules

    converted = ConfigMetadataI18n.convert_to_i18n_keys(CONFIG_METADATA_3)
    field = converted["ai_group"]["metadata"]["ai"]["items"][integration.PROFILE_PATH]

    assert field["description"] == integration.PROFILE_DESCRIPTION
    assert field["hint"] == integration.PROFILE_HINT


def test_profile_schema_remove_restores_core(bridge, plugin_modules):
    _, integration = plugin_modules
    profile_bridge, _, original_converter = bridge

    profile_bridge.remove()

    assert integration.PROFILE_PATH not in ai_items()
    assert integration.PROFILE_KEY not in validation_items()
    assert integration.PROFILE_KEY not in DEFAULT_CONFIG["provider_settings"]
    assert ConfigMetadataI18n.convert_to_i18n_keys is original_converter


def test_profile_toggle_requires_caption_provider(plugin_modules):
    _, integration = plugin_modules
    event = SimpleNamespace(unified_msg_origin="test:friend:1")
    context = FakeContext()

    config = SimpleNamespace(
        provider_settings={
            integration.PROFILE_KEY: True,
            "default_image_caption_provider_id": "caption",
        }
    )
    assert integration.profile_requests_image_caption(config, context, event) is True

    config.provider_settings[integration.PROFILE_KEY] = False
    assert integration.profile_requests_image_caption(config, context, event) is False

    config.provider_settings[integration.PROFILE_KEY] = True
    config.provider_settings["default_image_caption_provider_id"] = ""
    assert integration.profile_requests_image_caption(config, context, event) is False


def test_profile_toggle_falls_back_to_context_config(plugin_modules):
    _, integration = plugin_modules
    event = SimpleNamespace(unified_msg_origin="test:friend:1")
    context = FakeContext()
    context.astrbot_config_mgr.confs["default"]["provider_settings"].update(
        {
            integration.PROFILE_KEY: True,
            "default_image_caption_provider_id": "caption",
        }
    )
    config = SimpleNamespace(
        provider_settings={"default_image_caption_provider_id": "caption"}
    )

    assert integration.profile_requests_image_caption(config, context, event) is True


def test_provider_proxy_removes_only_image_modality(plugin_modules):
    _, integration = plugin_modules

    class Provider:
        provider_config: ClassVar[dict] = {
            "id": "main",
            "modalities": ["text", "image", "audio", "tool_use"],
        }

        def method(self):
            return "delegated"

    provider = Provider()
    proxy = integration._ProviderWithoutImage(provider)

    assert proxy.provider_config["modalities"] == ["text", "audio", "tool_use"]
    assert provider.provider_config["modalities"] == [
        "text",
        "image",
        "audio",
        "tool_use",
    ]
    assert proxy.provider_config is not provider.provider_config
    assert proxy.method() == "delegated"


@pytest.mark.parametrize(
    "modalities,expected",
    [
        ([], ["text"]),
        (None, ["text"]),
        (["image"], ["text"]),
    ],
)
def test_provider_proxy_handles_unconfigured_modalities(
    plugin_modules,
    modalities,
    expected,
):
    _, integration = plugin_modules
    provider = SimpleNamespace(provider_config={"modalities": modalities})

    proxy = integration._ProviderWithoutImage(provider)

    assert proxy.provider_config["modalities"] == expected


@pytest.mark.asyncio
async def test_patch_only_uses_proxy_when_profile_toggle_is_on(
    plugin_modules,
    monkeypatch,
):
    _, integration = plugin_modules
    calls = []

    async def fake_decorate(
        event,
        req,
        plugin_context,
        config,
        provider=None,
        montage_refs=None,
    ):
        del event, req, plugin_context, config, montage_refs
        calls.append(provider)
        return {"captioned"}

    monkeypatch.setattr(astr_main_agent, "_decorate_llm_request", fake_decorate)
    patch = integration.ForceImageCaptionPatch()
    patch.install()

    real_provider = SimpleNamespace(
        provider_config={"id": "main", "modalities": ["text", "image"]}
    )
    plugin_context = FakeContext()
    event = SimpleNamespace(unified_msg_origin="test:friend:1")
    req = SimpleNamespace()

    try:
        enabled_config = SimpleNamespace(
            provider_settings={
                integration.PROFILE_KEY: True,
                "default_image_caption_provider_id": "caption",
            }
        )
        result = await astr_main_agent._decorate_llm_request(
            event,
            req,
            plugin_context,
            enabled_config,
            provider=real_provider,
        )
        assert result == {"captioned"}
        assert calls[-1] is not real_provider
        assert calls[-1].provider_config["modalities"] == ["text"]

        disabled_config = SimpleNamespace(
            provider_settings={
                integration.PROFILE_KEY: False,
                "default_image_caption_provider_id": "caption",
            }
        )
        await astr_main_agent._decorate_llm_request(
            event,
            req,
            plugin_context,
            disabled_config,
            provider=real_provider,
        )
        assert calls[-1] is real_provider

        no_caption_provider = SimpleNamespace(
            provider_settings={
                integration.PROFILE_KEY: True,
                "default_image_caption_provider_id": "",
            }
        )
        await astr_main_agent._decorate_llm_request(
            event,
            req,
            plugin_context,
            no_caption_provider,
            provider=real_provider,
        )
        assert calls[-1] is real_provider
    finally:
        patch.remove()

    assert astr_main_agent._decorate_llm_request is fake_decorate


@pytest.mark.asyncio
async def test_enabling_plugin_does_not_enable_profile_behavior(
    plugin_modules,
    monkeypatch,
):
    main, integration = plugin_modules
    calls = []

    async def fake_decorate(
        event,
        req,
        plugin_context,
        config,
        provider=None,
        montage_refs=None,
    ):
        del event, req, plugin_context, config, montage_refs
        calls.append(provider)
        return set()

    monkeypatch.setattr(astr_main_agent, "_decorate_llm_request", fake_decorate)
    context = FakeContext()
    plugin = main.AlwaysParaphraseImage(context)
    await plugin.initialize()

    real_provider = SimpleNamespace(
        provider_config={"id": "main", "modalities": ["text", "image"]}
    )
    config = SimpleNamespace(
        provider_settings=context.astrbot_config_mgr.confs["default"][
            "provider_settings"
        ]
    )

    try:
        assert config.provider_settings[integration.PROFILE_KEY] is False
        await astr_main_agent._decorate_llm_request(
            SimpleNamespace(unified_msg_origin="test:friend:1"),
            SimpleNamespace(),
            context,
            config,
            provider=real_provider,
        )
        assert calls[-1] is real_provider
    finally:
        await plugin.terminate()

    assert astr_main_agent._decorate_llm_request is fake_decorate
    assert integration.PROFILE_PATH not in ai_items()


@pytest.mark.asyncio
async def test_profile_toggle_forces_builtin_captioning_path(
    plugin_modules,
    monkeypatch,
):
    main, integration = plugin_modules
    calls = []

    async def fake_decorate(
        event,
        req,
        plugin_context,
        config,
        provider=None,
        montage_refs=None,
    ):
        del event, req, plugin_context, config, montage_refs
        calls.append(provider)
        return set()

    monkeypatch.setattr(astr_main_agent, "_decorate_llm_request", fake_decorate)
    context = FakeContext()
    context.astrbot_config_mgr.confs["default"]["provider_settings"].update(
        {
            integration.PROFILE_KEY: True,
            "default_image_caption_provider_id": "caption",
        }
    )
    plugin = main.AlwaysParaphraseImage(context)
    await plugin.initialize()

    real_provider = SimpleNamespace(
        provider_config={"id": "main", "modalities": ["text", "image"]}
    )
    config = SimpleNamespace(
        provider_settings=context.astrbot_config_mgr.confs["default"][
            "provider_settings"
        ]
    )

    try:
        await astr_main_agent._decorate_llm_request(
            SimpleNamespace(unified_msg_origin="test:friend:1"),
            SimpleNamespace(),
            context,
            config,
            provider=real_provider,
        )
        assert calls[-1] is not real_provider
        assert "image" not in calls[-1].provider_config["modalities"]
    finally:
        await plugin.terminate()


def test_plugin_has_no_llm_configuration_schema():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "_conf_schema.json").exists()


@pytest.mark.asyncio
async def test_proxy_triggers_real_builtin_captioning(
    plugin_modules,
    monkeypatch,
):
    _, integration = plugin_modules
    image_ref = "data:image/png;base64,AAAA"

    class CaptionProvider(Provider):
        def __init__(self):
            super().__init__(
                {
                    "type": "openai_chat_completion",
                    "id": "caption",
                    "modalities": ["text", "image"],
                },
                {},
            )
            self.calls = []

        def get_current_key(self):
            return "key"

        def set_key(self, key):
            self.key = key

        async def get_models(self):
            return ["caption-model"]

        async def text_chat(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(role="assistant", completion_text="a cat")

    caption_provider = CaptionProvider()
    context = FakeContext()
    context.astrbot_config_mgr.confs["default"]["provider_settings"].update(
        {
            integration.PROFILE_KEY: True,
            "default_image_caption_provider_id": "caption",
            "image_caption_prompt": "describe",
        }
    )
    context.get_provider_by_id = lambda provider_id: (
        caption_provider if provider_id == "caption" else None
    )
    context.get_config = lambda umo=None: context.astrbot_config_mgr.confs["default"]

    async def noop(*args, **kwargs):
        del args, kwargs

    def sync_noop(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr(astr_main_agent, "_ensure_persona_and_skills", noop)
    monkeypatch.setattr(astr_main_agent, "_append_system_reminders", sync_noop)
    monkeypatch.setattr(astr_main_agent, "_apply_workspace_extra_prompt", noop)

    provider = SimpleNamespace(
        provider_config={"id": "main", "modalities": ["text", "image"]}
    )
    config = SimpleNamespace(
        provider_settings=context.astrbot_config_mgr.confs["default"][
            "provider_settings"
        ],
        timezone=None,
    )
    req = ProviderRequest(
        prompt="look",
        image_urls=[image_ref],
        conversation=SimpleNamespace(cid="conversation"),
    )
    patch = integration.ForceImageCaptionPatch()
    patch.install()

    try:
        captioned_refs = await astr_main_agent._decorate_llm_request(
            SimpleNamespace(unified_msg_origin="test:friend:1"),
            req,
            context,
            config,
            provider=provider,
        )
    finally:
        patch.remove()

    assert captioned_refs == {image_ref}
    assert caption_provider.calls[0]["image_urls"] == [image_ref]
    assert caption_provider.calls[0]["prompt"] == "describe"
    assert req.image_urls == []
    assert any(
        "<image_caption>a cat</image_caption>" in getattr(part, "text", "")
        for part in req.extra_user_content_parts
    )
