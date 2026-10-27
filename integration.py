from __future__ import annotations

import copy
from typing import Any

from astrbot.core import astr_main_agent
from astrbot.core.config.default import (
    CONFIG_METADATA_2,
    CONFIG_METADATA_3,
    DEFAULT_CONFIG,
)
from astrbot.core.config.i18n_utils import ConfigMetadataI18n

PROFILE_KEY = "always_paraphrase_image"
PROFILE_PATH = f"provider_settings.{PROFILE_KEY}"
PROFILE_DESCRIPTION = "始终转述图片 / Always paraphrase images"
PROFILE_HINT = (
    "即使对话模型支持图片输入，也先调用图片转述模型，只把文字描述发送给对话模型。"
    "需要在上方配置图片转述模型。 / Even when the chat model supports image input, "
    "use the image-captioning model and send only the text description. "
    "Requires an image-captioning model above."
)
PROFILE_ITEM_META = {
    "description": PROFILE_DESCRIPTION,
    "hint": PROFILE_HINT,
    "type": "bool",
}


def profile_requests_image_caption(
    config: Any,
    plugin_context: Any,
    event: Any,
) -> bool:
    """Return whether the current profile forces image captioning.

    Args:
        config: Main-agent build configuration for the current request.
        plugin_context: AstrBot plugin context used to resolve profile settings.
        event: Event that identifies the current session.

    Returns:
        True when the profile toggle is enabled and an image-captioning
        provider is configured.
    """
    settings = getattr(config, "provider_settings", None)
    if not isinstance(settings, dict) or PROFILE_KEY not in settings:
        try:
            contextual_config = plugin_context.get_config(
                umo=getattr(event, "unified_msg_origin", None)
            )
            settings = contextual_config.get("provider_settings", {})
        except Exception:  # noqa: BLE001
            return False

    if not isinstance(settings, dict) or not settings.get(PROFILE_KEY, False):
        return False
    return bool(str(settings.get("default_image_caption_provider_id") or "").strip())


class _ProviderWithoutImage:
    """Provider view that declares no image modality.

    AstrBot's built-in request decorator only uses the provider to decide
    whether the main model can consume images. This proxy makes that decision
    false for one call without mutating the real provider or racing other
    sessions.
    """

    def __init__(self, provider: Any) -> None:
        source_config = getattr(provider, "provider_config", None)
        provider_config = dict(source_config) if isinstance(source_config, dict) else {}
        modalities = provider_config.get("modalities")
        if isinstance(modalities, list):
            modalities = [item for item in modalities if item != "image"]
        else:
            modalities = []
        provider_config["modalities"] = modalities or ["text"]

        self.__dict__["_provider"] = provider
        self.__dict__["provider_config"] = provider_config

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_provider"], name)


class ForceImageCaptionPatch:
    """Make AstrBot's built-in image-captioning branch run on demand."""

    def __init__(self) -> None:
        self._installed = False
        self._original = None
        self._patched = None

    def install(self) -> None:
        """Install the request-decorator wrapper."""
        if self._installed:
            return

        original = astr_main_agent._decorate_llm_request
        if getattr(original, "_always_paraphrase_patched", False):
            raise RuntimeError("AlwaysParaphraseImage is already installed.")

        async def patched_decorate(
            event: Any,
            req: Any,
            plugin_context: Any,
            config: Any,
            provider: Any = None,
            montage_refs: Any = None,
        ) -> Any:
            if (
                not profile_requests_image_caption(
                    config,
                    plugin_context,
                    event,
                )
                or provider is None
            ):
                return await original(
                    event,
                    req,
                    plugin_context,
                    config,
                    provider=provider,
                    montage_refs=montage_refs,
                )
            return await original(
                event,
                req,
                plugin_context,
                config,
                provider=_ProviderWithoutImage(provider),
                montage_refs=montage_refs,
            )

        patched_decorate._always_paraphrase_patched = True
        patched_decorate._always_paraphrase_original = original
        astr_main_agent._decorate_llm_request = patched_decorate
        self._original = original
        self._patched = patched_decorate
        self._installed = True

    def remove(self) -> None:
        """Restore AstrBot's original request decorator."""
        if not self._installed:
            return
        if astr_main_agent._decorate_llm_request is self._patched:
            astr_main_agent._decorate_llm_request = self._original
        self._original = None
        self._patched = None
        self._installed = False


class ProfileSchemaBridge:
    """Expose the force-paraphrase toggle in AstrBot's profile settings."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._installed = False
        self._added_default = False
        self._added_profile_item = False
        self._added_validation = False
        self._ai_items: dict[str, Any] | None = None
        self._validation_items: dict[str, Any] | None = None
        self._original_converter = None
        self._patched_converter = None

    def install(self) -> None:
        """Add defaults, profile UI metadata, and validation metadata."""
        if self._installed:
            return

        self._installed = True
        try:
            ai_items = self._find_ai_items()
            validation_items = self._find_validation_items()

            provider_defaults = DEFAULT_CONFIG.setdefault("provider_settings", {})
            self._added_default = PROFILE_KEY not in provider_defaults
            provider_defaults.setdefault(PROFILE_KEY, False)

            self._ai_items = ai_items
            self._added_profile_item = PROFILE_PATH not in ai_items
            if self._added_profile_item:
                ordered_items: dict[str, Any] = {}
                for key, value in ai_items.items():
                    ordered_items[key] = value
                    if key == "provider_settings.default_image_caption_provider_id":
                        ordered_items[PROFILE_PATH] = copy.deepcopy(PROFILE_ITEM_META)
                ordered_items.setdefault(PROFILE_PATH, copy.deepcopy(PROFILE_ITEM_META))
                ai_items.clear()
                ai_items.update(ordered_items)

            self._validation_items = validation_items
            if validation_items is not None:
                self._added_validation = PROFILE_KEY not in validation_items
                validation_items.setdefault(PROFILE_KEY, {"type": "bool"})

            config_manager = getattr(self.context, "astrbot_config_mgr", None)
            for config in (getattr(config_manager, "confs", {}) or {}).values():
                if not isinstance(config, dict):
                    continue
                provider_settings = config.get("provider_settings")
                if not isinstance(provider_settings, dict):
                    provider_settings = {}
                    config["provider_settings"] = provider_settings
                provider_settings.setdefault(PROFILE_KEY, False)

            self._install_converter()
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        """Remove the profile UI and validation additions."""
        if not self._installed:
            return

        if self._ai_items is not None and self._added_profile_item:
            self._ai_items.pop(PROFILE_PATH, None)
        if self._validation_items is not None and self._added_validation:
            self._validation_items.pop(PROFILE_KEY, None)
        if self._added_default:
            provider_defaults = DEFAULT_CONFIG.get("provider_settings")
            if isinstance(provider_defaults, dict):
                provider_defaults.pop(PROFILE_KEY, None)
        if (
            self._patched_converter is not None
            and ConfigMetadataI18n.convert_to_i18n_keys is self._patched_converter
        ):
            ConfigMetadataI18n.convert_to_i18n_keys = staticmethod(
                self._original_converter
            )

        self._ai_items = None
        self._validation_items = None
        self._added_default = False
        self._added_profile_item = False
        self._added_validation = False
        self._original_converter = None
        self._patched_converter = None
        self._installed = False

    @staticmethod
    def _find_ai_items() -> dict[str, Any]:
        section = CONFIG_METADATA_3.get("ai_group", {}).get("metadata", {}).get("ai")
        items = section.get("items") if isinstance(section, dict) else None
        if not isinstance(items, dict):
            raise TypeError(
                "Unsupported AstrBot version: AI profile metadata structure is invalid."
            )
        return items

    @staticmethod
    def _find_validation_items() -> dict[str, Any] | None:
        metadata = CONFIG_METADATA_2.get("provider_group", {}).get("metadata", {})
        provider_settings = (
            metadata.get("provider_settings") if isinstance(metadata, dict) else None
        )
        items = (
            provider_settings.get("items")
            if isinstance(provider_settings, dict)
            else None
        )
        return items if isinstance(items, dict) else None

    def _install_converter(self) -> None:
        original = ConfigMetadataI18n.convert_to_i18n_keys
        if getattr(original, "_always_paraphrase_patched", False):
            raise RuntimeError("AlwaysParaphraseImage profile UI is already installed.")

        def patched(metadata: dict[str, Any]) -> dict[str, Any]:
            result = original(metadata)
            try:
                field = result["ai_group"]["metadata"]["ai"]["items"][PROFILE_PATH]
            except (KeyError, TypeError):
                return result
            field["description"] = PROFILE_DESCRIPTION
            field["hint"] = PROFILE_HINT
            return result

        patched._always_paraphrase_patched = True
        patched._always_paraphrase_original = original
        ConfigMetadataI18n.convert_to_i18n_keys = staticmethod(patched)
        self._original_converter = original
        self._patched_converter = patched
