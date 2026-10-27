# Always Paraphrase Image

An AstrBot plugin that adds a **per-profile** switch for forcing AstrBot's
built-in image paraphrasing even when the chat model supports image input.

Installing or enabling this plugin does **not** change behavior by itself. The
switch is disabled by default in every profile.

## Enable it

1. Install and enable the plugin.
2. Refresh the AstrBot dashboard.
3. Open your profile: **配置文件 → AI → 通用设置 → 图像与语音**.
4. Turn on **始终转述图片 / Always paraphrase images**.

The switch is per profile. Sessions using another profile keep their own
setting.

If the option is missing, click the circular-arrow **reload** button on the
plugin card, then hard-refresh the dashboard. The version label is read from
`metadata.yaml`, so it can show `2.0.0` while an older plugin instance is still
running. After a successful reload, the log contains:

```text
[AlwaysParaphraseImage] Registered profile option in N profile(s).
```

## Built-in paraphrasing only

The plugin does not define a captioning provider, prompt, timeout, or any other
LLM configuration. It reuses the profile's existing **图片转述模型** and
**图片转述提示词** and only changes the decision that normally disables that
built-in path for image-capable chat models.

An image-captioning model must be configured in the same profile. If it is
empty, the switch is inert.

## Scope

- Applies to AstrBot's built-in Agent request path.
- Third-party runners such as Dify, Coze, DashScope, and DeerFlow manage their
  own image handling and are not affected.
- Disabling the plugin removes the profile option and restores AstrBot's normal
  behavior. The stored per-profile value is preserved for the next enable.
