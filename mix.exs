defmodule Godot.Mixfile do
  use Mix.Project

  def project do
    [
      app: :godot,
      compilers: [:unifex, :bundlex] \
      ++ [:elixir_make] \
      ++ Mix.compilers,
      version: "0.1.0",
      deps: deps(),
      extra_applications: [:logger],
      package: package(),
      description: description(),
   ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps() do
    [
      {:unifex, "~> 0.7.0"},
      {:elixir_make, "~> 0.4", runtime: false},
    ]
  end

  defp description() do
    "Godot as an Elixir library."
  end

  defp package do
    [
      licenses: ["MIT"],
      files: [
        "lib", "LICENSE.txt", "mix.exs", "README.md",
        "c_src", "c_src/*.md", "Makefile", "_build/dev/lib/godot/priv/libgodot.linuxbsd.opt.tools.64.llvm.so"],
      links: %{"GitHub" => "https://github.com/V-Sekai/godot/tree/elixir"}
    ]
  end
end
