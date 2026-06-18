"""Native file dialogs, with graceful fallbacks when none are available.

The graphical interface uses these to offer the familiar operating-system save
dialog. They rely on Tkinter, which ships with most desktop Python builds; when
it is missing (for example in a headless environment), the caller is given a
sensible default path instead so that saving still works.
"""

from __future__ import annotations

import pathlib


def ask_save_pgn_path(
    suggested_name: str, fallback_directory: pathlib.Path
) -> tuple[pathlib.Path | None, bool]:
    """Ask the user where to save a PGN file using the native dialog.

    Args:
        suggested_name: The default file name to offer.
        fallback_directory: The directory to save into when no dialog can be
            shown.

    Returns:
        A tuple ``(path, was_cancelled)``. When a dialog is shown, ``path`` is
        the chosen path or ``None`` if the user cancelled. When no dialog is
        available, ``path`` is a default inside ``fallback_directory`` and
        ``was_cancelled`` is False.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:
        fallback_directory.mkdir(parents=True, exist_ok=True)
        return fallback_directory / suggested_name, False

    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.asksaveasfilename(
            parent=root,
            title="Save game as PGN",
            initialfile=suggested_name,
            defaultextension=".pgn",
            filetypes=[("PGN file", "*.pgn"), ("All files", "*.*")],
        )
        root.update()
        root.destroy()
    except Exception:
        fallback_directory.mkdir(parents=True, exist_ok=True)
        return fallback_directory / suggested_name, False

    if not chosen:
        return None, True
    return pathlib.Path(chosen), False
