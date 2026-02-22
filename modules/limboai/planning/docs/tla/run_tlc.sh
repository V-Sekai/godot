#!/bin/bash
# Helper script to run TLC (TLA+ Model Checker) using TLA+ Toolbox's tla2tools.jar

TLA_TOOLBOX_JAR="/Applications/TLA+ Toolbox.app/Contents/Eclipse/tla2tools.jar"

# Try to find Java - check Homebrew first, then system
if [ -f "/opt/homebrew/opt/openjdk/bin/java" ]; then
    JAVA_CMD="/opt/homebrew/opt/openjdk/bin/java"
elif [ -f "/usr/bin/java" ]; then
    JAVA_CMD="/usr/bin/java"
else
    JAVA_CMD="java"
fi

if [ ! -f "$TLA_TOOLBOX_JAR" ]; then
    echo "Error: TLA+ Toolbox jar not found at $TLA_TOOLBOX_JAR"
    exit 1
fi

# Run TLC with the provided arguments
"$JAVA_CMD" -cp "$TLA_TOOLBOX_JAR" tlc2.TLC "$@"
