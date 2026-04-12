# Contributing -- keychain

Platform-gated module wrapping
[thirdparty/keychain](../../thirdparty/keychain/) for persisting secrets
in OS secure storage (macOS/iOS/visionOS Keychain Services, Windows
Credential Vault, Linux libsecret, Android KeyStore). Only the web
platform is excluded via `can_build` in `config.py`.

## Build

Built automatically when `scons` targets a supported platform:

```bash
scons platform=macos dev_build=yes
```

On unsupported platforms the module is skipped and dependents gate calls
behind `#ifdef MODULE_KEYCHAIN_ENABLED`.

## Companion projects

See [COMPANION_PROJECTS.md](../../COMPANION_PROJECTS.md) for the full
list of sibling modules and vendored projects that must stay in sync.
