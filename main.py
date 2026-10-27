from __future__ import annotations

from astrbot.api import logger
from astrbot.api.star import Context, Star

from .integration import ForceImageCaptionPatch, ProfileSchemaBridge

RUNTIME_VERSION = "2.0.0"


class AlwaysParaphraseImage(Star):
    """Add a per-profile switch that forces built-in image captioning."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self._profile_bridge = ProfileSchemaBridge(context)
        self._caption_patch = ForceImageCaptionPatch()
        self._installed = False

    async def initialize(self) -> None:
        """Install the profile toggle and the built-in captioning trigger."""
        if self._installed:
            return

        try:
            self._profile_bridge.install()
            self._caption_patch.install()
        except Exception:
            self._caption_patch.remove()
            self._profile_bridge.remove()
            raise

        self._installed = True
        logger.info(
            "[AlwaysParaphraseImage] v%s installed profile option "
            "provider_settings.always_paraphrase_image. It is disabled by default.",
            RUNTIME_VERSION,
        )

    async def terminate(self) -> None:
        """Restore AstrBot's original profile metadata and request flow."""
        self._caption_patch.remove()
        self._profile_bridge.remove()
        self._installed = False
