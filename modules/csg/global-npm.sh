#!/bin/bash

mkdir "${HOME}/.npm-packages"
npm config set prefix "\${HOME}/.npm-packages"

cat >> ~/.bashrc <<EOL
NPM_PACKAGES="\${HOME}/.npm-packages"

export PATH="\$PATH:$NPM_PACKAGES/bin"

# Preserve MANPATH if you already defined it somewhere in your config.
# Otherwise, fall back to `manpath` so we can inherit from `/etc/manpath`.
export MANPATH="${MANPATH-$(manpath)}:$NPM_PACKAGES/share/man"
EOL



