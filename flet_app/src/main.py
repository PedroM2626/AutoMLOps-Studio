import flet as ft
from components.app_bar import AppBar
from components.navigation import Navigation
from contexts.route import RouteContext, RouteContextValue
from contexts.theme import ThemeContext, ThemeContextValue
from models.app_state import AppState

# Views
from views.data_view import DataView
from views.train_view import TrainView
from views.experiments_view import ExperimentsView
from views.cv_view import CVView
from views.registry_view import RegistryView

@ft.component
def App():
    # State management
    state, _ = ft.use_state(AppState())

    # Subscribe to page events
    ft.context.page.on_route_change = state.route_change
    ft.context.page.on_view_pop = state.view_popped

    # Callbacks for theme
    def toggle_mode():
        state.toggle_theme()
        ft.context.page.update()

    def set_theme_color(color: str):
        state.set_theme_color(color)
        ft.context.page.update()

    theme_value = ft.use_memo(
        lambda: ThemeContextValue(
            mode=state.theme_mode,
            seed_color=state.theme_color,
            toggle_mode=toggle_mode,
            set_seed_color=set_theme_color,
        ),
        dependencies=[state.theme_mode, state.theme_color],
    )

    # Callbacks for routing
    def navigate_callback(new_route: str):
        state.navigate(new_route)
        ft.context.page.update()

    route_value = ft.use_memo(
        lambda: RouteContextValue(
            route=state.route,
            navigate=navigate_callback,
        ),
        dependencies=[state.route],
    )

    def on_mounted():
        ft.context.page.title = "AutoMLOps Studio"
        ft.context.page.theme_mode = state.theme_mode
        ft.context.page.theme = ft.Theme(color_scheme_seed=state.theme_color)
        ft.context.page.update()

    ft.on_mounted(on_mounted)

    # Sync theme with page
    def sync_theme():
        ft.context.page.theme_mode = state.theme_mode
        ft.context.page.theme = ft.Theme(color_scheme_seed=state.theme_color)
        ft.context.page.update()

    ft.on_updated(sync_theme, [state.theme_mode, state.theme_color])

    # View selection based on route
    def get_view_content():
        if state.route == "/":
            return DataView()
        elif state.route == "/train":
            return TrainView()
        elif state.route == "/experiments":
            return ExperimentsView()
        elif state.route == "/cv":
            return CVView()
        elif state.route == "/registry":
            return RegistryView()
        else:
            return ft.Text("Page not found", size=30)

    return RouteContext(
        route_value,
        lambda: ThemeContext(
            theme_value,
            lambda: ft.View(
                route="/",
                appbar=AppBar(),
                controls=[
                    ft.Row(
                        expand=True,
                        controls=[
                            ft.Container(
                                content=Navigation(),
                                padding=20,
                                bgcolor="surfacevariant",
                                width=250, # Explicit width for sidebar container
                                border_radius=ft.BorderRadius.only(top_right=20, bottom_right=20),
                            ),
                            ft.VerticalDivider(width=1),
                            ft.Container(
                                content=get_view_content(),
                                expand=True,
                                padding=20,
                            ),
                        ],
                    )
                ],
            ),
        ),
    )

if __name__ == "__main__":
    ft.run(lambda page: page.render_views(lambda: App()))
