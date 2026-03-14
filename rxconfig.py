import reflex as rx


config = rx.Config(
    app_name="automlops_reflex",
    state_auto_setters=True,
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)