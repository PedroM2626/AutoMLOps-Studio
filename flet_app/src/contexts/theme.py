import flet as ft
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ThemeContextValue:
    mode: ft.ThemeMode
    seed_color: str
    toggle_mode: Callable[[], None]
    set_seed_color: Callable[[str], None]

ThemeContext = ft.create_context(ThemeContextValue)
