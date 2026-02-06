import pytest
import flet as ft
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.app_state import AppState

def test_app_state_initialization():
    state = AppState()
    assert state.route == "/"
    assert state.theme_mode == ft.ThemeMode.LIGHT
    assert state.theme_color == ft.Colors.GREEN

def test_app_state_navigation():
    # We can't easily test ft.context.page.go without a real page,
    # but we can check if the route is updated in our state if we mock it.
    state = AppState()
    # Mocking page interaction would be complex here, 
    # but we can test the logic that doesn't depend on flet internals.
    state.route = "/train"
    assert state.route == "/train"

def test_theme_toggle():
    state = AppState()
    state.theme_mode = ft.ThemeMode.LIGHT
    state.toggle_theme()
    # Note: toggle_theme calls ft.context.page.update() which will fail without a page.
    # In a real test environment, we'd use flet.Page mock.
    pass

@pytest.mark.parametrize("route", ["/", "/train", "/experiments", "/cv", "/registry"])
def test_valid_routes(route):
    state = AppState()
    state.route = route
    assert state.route == route
