# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files


project = Path(SPECPATH)
datas = [
    (str(project / "assets"), "assets"),
    (str(project / "packaging" / "config.jsonc"), "."),
    (str(project / "words.txt"), "."),
    (str(project / "test_script.txt"), "."),
]
datas += collect_data_files("faster_whisper")

a = Analysis(
    ["web_gui.py"],
    pathex=[str(project)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "av",
        "ctranslate2",
        "faster_whisper",
        "huggingface_hub",
        "onnxruntime",
        "sounddevice",
        "tokenizers",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Stream Censor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Stream Censor",
)
app = BUNDLE(
    coll,
    name="Stream Censor.app",
    icon=None,
    bundle_identifier="com.tomatetz.stream-censor",
    info_plist={
        "CFBundleDisplayName": "Stream Censor",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSMicrophoneUsageDescription": (
            "Stream Censor uses the microphone to recognize and replace "
            "configured words before streaming."
        ),
        "NSHighResolutionCapable": True,
    },
)
