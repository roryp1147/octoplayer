# Octoplayer

An eerie bioluminescent deep-sea octopus music player for Windows. It floats on your desktop, glows from one solid color to the next and ripples to the beat.

![Octoplayer](https://roryp1147.github.io/img/octo-beat.webp)

## Download

**[Download Octoplayer for Windows (v1.0.0)](https://github.com/roryp1147/octoplayer/releases/latest/download/Octoplayer-1.0.0-win64.zip)** · [Project page](https://roryp1147.github.io/projects/octoplayer.html)

Windows 10/11 (64-bit). No Python required: extract the zip and run `Octoplayer.exe`. The app isn't code-signed, so SmartScreen may warn you; click **More info → Run anyway**.

## Run from source

Requires Windows 10/11 and Python 3.14.

```bat
git clone https://github.com/roryp1147/octoplayer.git
cd octoplayer
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

`run_octoplayer.bat` starts it without a console window. Useful options: `--no-boot`, `--on-top`, `--source auto|local|loopback|synth|none`. Setting `OCTO_MUTE=1` silences everything, and `OCTO_NO_SETTINGS=1` ignores saved settings (they normally live in the registry under `HKCU\Software\Octoplayer`).

## Build the exe (PyInstaller)

```bat
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --clean --onedir --windowed --name Octoplayer ^
  --icon assets\octoplayer.ico --add-data "assets;assets" --paths . ^
  --collect-submodules player --collect-submodules winrt main.py
```

The app ends up in `dist\Octoplayer\`. Zip that folder to share it. You can delete `_internal\PySide6\opengl32sw.dll` to save about 20 MB.

## Developer tools

- `tools/build_layers.py` rebuilds the art layers from the original concept art (not included), and `tools/make_icon.py` regenerates the icon. These need `pip install -r requirements-tools.txt`.

## License

The **code** is MIT licensed, (c) 2026 Rory. The **art and sound assets** in `assets/` are free for personal use only and may not be redistributed or resold. See [LICENSE](LICENSE) (code) and [ASSETS-LICENSE.txt](ASSETS-LICENSE.txt) (assets). Not affiliated with any company. 
