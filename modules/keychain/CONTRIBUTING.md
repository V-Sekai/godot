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

### [`modules/multiplayer_fabric_mmog`](../multiplayer_fabric_mmog) -- MMOG layer

The primary consumer. `fabric_mmog_asset.cpp` uses `FabricMMOGKeyStore`
to cache per-asset AES key material between sessions, guarded by
`MODULE_KEYCHAIN_ENABLED`.

### [`modules/multiplayer_fabric`](../multiplayer_fabric) -- zone transport layer

The lower networking layer. No direct dependency on this module, but
shares the same fabric and build tree.

### [`thirdparty/keychain`](../../thirdparty/keychain/) -- vendored library

Upstream: <https://github.com/hrantzsch/keychain>

The cross-platform keychain C++ library. One platform backend
(`keychain_mac.cpp`, `keychain_win.cpp`, `keychain_linux.cpp`, or
`keychain_android.cpp`) is compiled per platform by `SCsub`. The macOS
backend uses the `SecItem*` API shared across all Apple platforms, so
iOS and visionOS compile the same file.

### [`thirdparty/uro`](../../thirdparty/uro) -- asset backend

The key material cached by this module originates from uro's
`/auth/script_key` endpoint. TTL and key size constants must stay in
sync between `FabricMMOGKeyStore` and the uro route.
