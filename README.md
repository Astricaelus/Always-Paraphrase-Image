# Always Paraphrase Image

An AstrBot plugin that keeps the main chat model text-only. It paraphrases
images into captions before every main-model request, even when the main model
supports image input.

This is useful when a session accumulates many images: the main model API body
stays small because images are never sent to it, and only their text
descriptions are kept in the conversation history.

## What it does

- Captions images from the current user message.
- Captions image content parts added by other plugins.
- Replaces images already stored in the conversation history with captions, so
  old images do not keep inflating later requests.
- Caches captions in memory to avoid describing the same image repeatedly.
- Fails closed: if the captioning provider is missing, times out, or errors, the
  image is still removed and replaced with a text placeholder.

## Install

1. Put this repository in `data/plugins/astrbot_plugin_always_paraphrase_image`.
2. Restart AstrBot or reload plugins.
3. Configure `Image captioning provider` in the plugin settings, or keep it
   empty to use AstrBot's existing `default_image_caption_provider_id` setting.

The plugin uses AstrBot's normal image-captioning prompt by default. You can
override the prompt in the plugin settings.

## Configuration

| Option | Description |
| --- | --- |
| `caption_provider_id` | Chat provider used to describe images. Empty falls back to AstrBot's image-captioning provider setting, then to the session's image-capable chat provider when available. |
| `caption_prompt` | Prompt used for each image. Empty falls back to AstrBot's `image_caption_prompt`. |
| `caption_timeout_seconds` | Per-image timeout. `0` disables the timeout. |

## Notes

- The plugin only affects requests made through AstrBot's main agent pipeline.
- Each image gets its own caption request, with bounded concurrency. This makes
  the result easy to map back to each image and lets identical images reuse the
  caption cache.
- Captions are text. They are persisted with the conversation so future turns
  remain text-only.
