# Interaction System for Godot

This module contains a system for delegating input events such as clicks or VR controller clicks, as 3d raycasts into a 3D scene.

Canvas UI is supported via canvas_plane, so that the 3D clicks (or raycast from 2D mouse clicks) are passed through the Lasso database to determine which UI element is closest to the click.

## Usage

Place this addon at `addons/interaction_system` in your project.

Note that for use with `canvas_plane`, this addon is designed to work together with https://github.com/V-Sekai/canvas_plane on the `interaction_system` branch. Make sure to place the canvas plane addon at `addons/canvas_plane` in your project.

## Work in progress

This addon is not finished yet. If you encounter something broken, file an issue.

## License

MIT License. See [LICENSE](./LICENSE)