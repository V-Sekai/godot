# Godot Steam®Audio

This is a Godot 4.3 module that adds support for Valve's Steam® Audio SDK/Library for the spatialization of sound in a 3D space. This module provides an integration with the Godot (4.3) game engine and is MIT-licensed ([MODULELICENSE.md](MODULELICENSE.md)), but the Steam® Audio SDK/Library itself is licensed under Valve's Apache 2.0 license, and thus, if you use this module, you are bound by Valve's terms.

You can find the Steam® Audio Github page [here](https://github.com/ValveSoftware/steam-audio). You can also see the latest release here under "C API": [https://valvesoftware.github.io/steam-audio/downloads.html](https://valvesoftware.github.io/steam-audio/downloads.html)

## Building

This is a module, meaning it must be compiled as you build the Godot engine. Work is being done to move this module to the GDExtension framework to avoid the need to recompile the engine.

At this time, this repository only supports Windows builds. Contributions to get Linux and Mac support working would be greatly appreciated.
If you are unfamiliar with compiling Godot from source, please read this guide from the official docs: [Compiling Godot from Source](https://docs.godotengine.org/en/stable/contributing/development/compiling/index.html). To get a Windows build going, clone or copy this repository into the Godot source's modules folder:

```bash
git clone https://github.com/V-Sekai/godot -b vsk-steam-audio-4.3 godot-steam-audio
```

## Sample Project

A sample project with multiple test scenes is available at [https://github.com/V-Sekai/godot_steamaudio_sample_project](https://github.com/V-Sekai/godot_steamaudio_sample_project)

## Current Status

Currently, there is support for rendering 3D ambisonics, volumetric occlusion by geometry, transmission through geometry, and distance attenuation.

## Road Map

1. Real-time Reflections (echoes caused by geometry) support via GPU or CPU ray-tracing
2. Non-static geometry (moving objects) influencing the simulation
3. Exposing controls in the Godot Editor through inspect settings and project settings
4. Pathing baking - offline computation of paths to spatialize sound. This requires a lot of Editor work!
5. More as they come!
