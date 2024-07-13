# LEMON fork

This repository is a (shallow) fork of [LEMON](https://lemon.cs.elte.hu/trac/lemon),
the *L*ibrary for *E*fficient *M*odeling and *O*ptimization in *N*etworks.

We are very happy users of LEMON, this repository just contains minor changes for
C++20 compatibility and slightly improved CMake.

## Changes in this fork

- Apply Kevin Tew's [C++20 compat patch](http://lemon.cs.elte.hu/trac/lemon/ticket/631)
- Some additional C++20 compat changes.
- Improve `CMakeLists.txt`:
    - Create a `lemon::lemon` `ALIAS` target
    - Compile as position-independent code (`POSITION_INDEPENDENT_CODE` / `-fPIC`)
    - Allow for interprocedural optimization (`-flto`)

See [README](README) for the original LEMON readme.


## License

Lemon (and the changes in this fork) are available under the terms of the [LICENSE](Boost Software License, Version 1.0)


