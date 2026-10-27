from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import Provider, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import ImageURLPart, Message, TextPart

DEFAULT_CAPTION_PROMPT = "Please describe the image."
DEFAULT_CAPTION_TIMEOUT_SECONDS = 60
CAPTION_CACHE_LIMIT = 512
CAPTION_CONCURRENCY = 4

FAILED_CAPTION_TEXT = "[Image caption unavailable]"
MISSING_PROVIDER_TEXT = "[Image omitted: no image-captioning provider is configured]"


def _image_ref(part: Any) -> str | None:
    """Return the image reference of a content part, if it has one.

    Args:
        part: A pydantic content part or an OpenAI-style content dictionary.

    Returns:
        The image URL, data URI, or local path, or None when the part has no
        usable image reference.
    """
    if isinstance(part, ImageURLPart):
        url = part.image_url.url
        return url if isinstance(url, str) and url else None

    if isinstance(part, dict) and part.get("type") == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            return url if isinstance(url, str) and url else None
        if isinstance(image_url, str):
            return image_url or None
    return None


def _is_image_part(part: Any) -> bool:
    """Return whether a content part represents an image.

    Args:
        part: A pydantic content part or an OpenAI-style content dictionary.

    Returns:
        True when the part declares the ``image_url`` content type.
    """
    return isinstance(part, ImageURLPart) or (
        isinstance(part, dict) and part.get("type") == "image_url"
    )


def _is_temporary(part: Any) -> bool:
    """Return whether a content part is marked as provider-only.

    Args:
        part: A pydantic content part or an OpenAI-style content dictionary.

    Returns:
        True when the part should not be persisted to conversation history.
    """
    if isinstance(part, dict):
        return bool(part.get("_no_save"))
    return bool(getattr(part, "_no_save", False))


def _text_part(text: str, *, temporary: bool = False) -> TextPart:
    """Build a text part, optionally marking it provider-only.

    Args:
        text: The text content.
        temporary: Whether the part must not be persisted.

    Returns:
        The constructed text part.
    """
    part = TextPart(text=text)
    return part.mark_as_temp() if temporary else part


def _caption_block(caption: str) -> str:
    """Wrap one image caption in AstrBot's caption marker.

    Args:
        caption: The image description.

    Returns:
        The caption wrapped in ``<image_caption>`` tags.
    """
    safe_caption = caption.replace("</image_caption>", "&lt;/image_caption&gt;")
    return f"<image_caption>{safe_caption}</image_caption>"


def _cache_key(prompt: str, image_ref: str) -> str:
    """Build a stable cache key for one image and caption prompt.

    Args:
        prompt: Prompt sent to the captioning provider.
        image_ref: Image URL, data URI, or local path.

    Returns:
        Hex-encoded SHA-256 cache key.
    """
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8", "surrogatepass"))
    digest.update(b"\0")
    digest.update(image_ref.encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def _merge_adjacent_text_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent text blocks in an OpenAI-style content list.

    Args:
        parts: Content blocks belonging to one message.

    Returns:
        A new list with adjacent text blocks merged.
    """
    merged: list[dict[str, Any]] = []
    for part in parts:
        if merged and part.get("type") == "text" and merged[-1].get("type") == "text":
            previous_text = merged[-1].get("text", "")
            current_text = part.get("text", "")
            merged[-1]["text"] = (
                f"{previous_text}\n{current_text}" if previous_text else current_text
            )
            continue
        merged.append(part)
    return merged


class AlwaysParaphraseImage(Star):
    """Force all main-model requests to be text-only.

    The plugin paraphrases current message images, image content parts added by
    other plugins, and images already stored in the conversation history. The
    main model therefore never receives an ``image_url`` content block, even
    when its provider declares image-input support.
    """

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.plugin_config = config or {}
        self._caption_cache: OrderedDict[str, str] = OrderedDict()
        self._warned_missing_provider_ids: set[str] = set()

    async def initialize(self) -> None:
        """Log that the plugin is ready."""
        logger.info(
            "[AlwaysParaphraseImage] Active: main-model requests will contain "
            "text captions instead of images."
        )

    async def terminate(self) -> None:
        """Release the in-memory caption cache."""
        self._caption_cache.clear()
        self._warned_missing_provider_ids.clear()

    @filter.on_llm_request()
    async def paraphrase_images(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Replace every image in a main-model request with a text caption.

        The hook fails closed: if captioning is unavailable, images are still
        removed and replaced with a text placeholder.

        Args:
            event: The message event that triggered the LLM request.
            req: The request that will be sent to the main chat provider.
        """
        contexts = req.contexts
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except (TypeError, ValueError):
                logger.warning(
                    "[AlwaysParaphraseImage] Could not parse conversation history; "
                    "images in the history will be removed without captions."
                )
                contexts = []
            req.contexts = contexts
        if isinstance(contexts, (dict, Message)):
            contexts = [contexts]
            req.contexts = contexts
        elif not isinstance(contexts, list):
            contexts = []

        current_refs = [
            ref for ref in (req.image_urls or []) if isinstance(ref, str) and ref
        ]
        extra_parts = list(req.extra_user_content_parts or [])
        extra_refs = [
            ref
            for part in extra_parts
            if _is_image_part(part) and (ref := _image_ref(part))
        ]
        extra_has_image = any(_is_image_part(part) for part in extra_parts)
        context_has_image = False
        context_refs: list[str] = []
        for message in contexts:
            if isinstance(message, Message):
                content = message.content
            elif isinstance(message, dict):
                content = message.get("content")
            else:
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not _is_image_part(part):
                    continue
                context_has_image = True
                if ref := _image_ref(part):
                    context_refs.append(ref)

        all_refs = list(dict.fromkeys(current_refs + extra_refs + context_refs))
        if not (req.image_urls or extra_has_image or context_has_image):
            return

        provider_ids, prompt = self._resolve_caption_settings(event)
        provider = await self._resolve_caption_provider(event, provider_ids)
        timeout = self._caption_timeout()
        semaphore = asyncio.Semaphore(CAPTION_CONCURRENCY)

        captions = await asyncio.gather(
            *[
                self._caption_image(
                    provider=provider,
                    prompt=prompt,
                    image_ref=ref,
                    timeout=timeout,
                    semaphore=semaphore,
                )
                for ref in all_refs
            ]
        )
        caption_by_ref = dict(zip(all_refs, captions, strict=True))

        req.image_urls = []
        if current_refs:
            current_captions = [caption_by_ref[ref] for ref in current_refs]
            if len(current_captions) == 1:
                current_text = _caption_block(current_captions[0])
            else:
                lines = [
                    f"Image {index}: {caption}"
                    for index, caption in enumerate(current_captions, start=1)
                ]
                current_text = (
                    "<image_caption>\n" + "\n".join(lines) + "\n</image_caption>"
                )
            if req.prompt and req.prompt.strip():
                req.prompt = f"{req.prompt}\n\n{current_text}"
            else:
                req.prompt = current_text

        rewritten_parts: list[Any] = []
        for part in extra_parts:
            if not _is_image_part(part):
                rewritten_parts.append(part)
                continue
            ref = _image_ref(part)
            caption = caption_by_ref.get(ref, FAILED_CAPTION_TEXT)
            rewritten_parts.append(
                _text_part(_caption_block(caption), temporary=_is_temporary(part))
            )
        req.extra_user_content_parts = rewritten_parts

        for message in contexts:
            if isinstance(message, Message):
                content = message.content
                if not isinstance(content, list):
                    continue
                new_content = []
                for part in content:
                    if not _is_image_part(part):
                        new_content.append(part)
                        continue
                    ref = _image_ref(part)
                    caption = caption_by_ref.get(ref, FAILED_CAPTION_TEXT)
                    new_content.append(
                        _text_part(
                            _caption_block(caption), temporary=_is_temporary(part)
                        )
                    )
                message.content = new_content
                continue

            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            new_content: list[dict[str, Any]] = []
            for part in content:
                if not _is_image_part(part):
                    new_content.append(part)
                    continue
                ref = _image_ref(part)
                caption = caption_by_ref.get(ref, FAILED_CAPTION_TEXT)
                new_content.append({"type": "text", "text": _caption_block(caption)})
            if all(isinstance(part, dict) for part in new_content):
                new_content = _merge_adjacent_text_parts(new_content)
            message["content"] = new_content

        logger.debug(
            "[AlwaysParaphraseImage] Paraphrased %d unique image(s) for session %s.",
            len(all_refs),
            getattr(event, "unified_msg_origin", "<unknown>"),
        )

    def _resolve_caption_settings(
        self,
        event: AstrMessageEvent,
    ) -> tuple[list[str], str]:
        """Resolve caption provider candidates and the caption prompt.

        Plugin values take precedence over AstrBot's built-in image-captioning
        settings, which remain the default so the plugin works with the existing
        WebUI configuration.

        Args:
            event: The event used to resolve session-specific AstrBot settings.

        Returns:
            A tuple of deduplicated provider IDs to try and the caption prompt.
        """
        core_config = (
            self.context.get_config(umo=getattr(event, "unified_msg_origin", None))
            or {}
        )
        provider_settings = core_config.get("provider_settings", {}) or {}
        if not isinstance(provider_settings, dict):
            provider_settings = {}

        provider_ids: list[str] = []
        for value in (
            self.plugin_config.get("caption_provider_id"),
            provider_settings.get("default_image_caption_provider_id"),
        ):
            candidate = str(value or "").strip()
            if candidate and candidate not in provider_ids:
                provider_ids.append(candidate)

        for value in (
            self.plugin_config.get("caption_prompt"),
            provider_settings.get("image_caption_prompt"),
            DEFAULT_CAPTION_PROMPT,
        ):
            prompt = str(value or "").strip()
            if prompt:
                return provider_ids, prompt
        return provider_ids, DEFAULT_CAPTION_PROMPT

    async def _resolve_caption_provider(
        self,
        event: AstrMessageEvent,
        provider_ids: list[str],
    ) -> Provider | None:
        """Find a usable chat provider for image captioning.

        Args:
            event: The event that identifies the current session.
            provider_ids: Caption provider IDs to try in order.

        Returns:
            A chat provider, or None when no image-capable provider is available.
        """
        for provider_id in provider_ids:
            provider = self.context.get_provider_by_id(provider_id)
            if isinstance(provider, Provider):
                return provider
            if provider_id not in self._warned_missing_provider_ids:
                self._warned_missing_provider_ids.add(provider_id)
                logger.warning(
                    "[AlwaysParaphraseImage] Image-captioning provider %r was not "
                    "found; trying the next available provider.",
                    provider_id,
                )

        try:
            provider = await self.context.get_using_provider_async(
                umo=getattr(event, "unified_msg_origin", None)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[AlwaysParaphraseImage] Could not resolve the session's chat "
                "provider for image captioning (%s).",
                type(exc).__name__,
            )
            return None

        if isinstance(provider, Provider) and self._supports_image(provider):
            logger.debug(
                "[AlwaysParaphraseImage] No caption provider configured; using "
                "the session's image-capable chat provider."
            )
            return provider
        return None

    @staticmethod
    def _supports_image(provider: Provider) -> bool:
        """Return whether a provider explicitly supports image input.

        Args:
            provider: The provider to inspect.

        Returns:
            True when the provider's modality list contains ``image``.
        """
        modalities = provider.provider_config.get("modalities", None)
        return isinstance(modalities, list) and "image" in modalities

    def _caption_timeout(self) -> float:
        """Return the configured per-image caption timeout.

        Returns:
            Timeout in seconds, or 0 to disable the timeout.
        """
        raw = self.plugin_config.get(
            "caption_timeout_seconds",
            DEFAULT_CAPTION_TIMEOUT_SECONDS,
        )
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return float(DEFAULT_CAPTION_TIMEOUT_SECONDS)

    async def _caption_image(
        self,
        *,
        provider: Provider | None,
        prompt: str,
        image_ref: str,
        timeout: float,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Caption one image, using the in-memory cache when possible.

        Args:
            provider: Captioning provider, or None when none is configured.
            prompt: Prompt sent to the captioning provider.
            image_ref: Image URL, data URI, or local path.
            timeout: Per-image timeout in seconds; 0 disables the timeout.
            semaphore: Limits concurrent caption provider calls.

        Returns:
            The caption, or a text placeholder when captioning fails.
        """
        key = _cache_key(prompt, image_ref)
        cached = self._caption_cache.get(key)
        if cached is not None:
            self._caption_cache.move_to_end(key)
            return cached

        if provider is None:
            return MISSING_PROVIDER_TEXT

        async with semaphore:
            cached = self._caption_cache.get(key)
            if cached is not None:
                self._caption_cache.move_to_end(key)
                return cached

            try:
                request = provider.text_chat(prompt=prompt, image_urls=[image_ref])
                if timeout > 0:
                    response = await asyncio.wait_for(request, timeout=timeout)
                else:
                    response = await request
                caption = (response.completion_text or "").strip()
                if not caption:
                    raise ValueError("empty caption")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[AlwaysParaphraseImage] Failed to caption image %s (%s); "
                    "removing it from the request.",
                    key[:12],
                    type(exc).__name__,
                )
                return FAILED_CAPTION_TEXT

        self._caption_cache[key] = caption
        self._caption_cache.move_to_end(key)
        while len(self._caption_cache) > CAPTION_CACHE_LIMIT:
            self._caption_cache.popitem(last=False)
        return caption
