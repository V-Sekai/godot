defmodule Uro.MixProject do
  use Mix.Project

  def project do
    [
      app: :uro,
      version: "0.1.0",
      elixir: ">= 1.16.3",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: [] ++ Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps()
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [
      mod: {Uro.Application, []},
      extra_applications: [
        :logger,
        :runtime_tools,
        :email_checker,
        :mnesia,
        :scrivener_ecto,
        :httpoison
      ]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      {:phoenix, "~> 1.7"},
      {:phoenix_pubsub, "~> 2.0"},
      {:phoenix_ecto, "~> 4.4"},
      {:phoenix_live_view, "~> 0.20.3"},
      {:phoenix_view, "~> 2.0"},
      {:ecto_sql, "~> 3.11"},
      {:redix, "~> 0.9.2"},
      {:postgrex, ">= 0.0.0"},
      {:phoenix_html, "~> 3.3"},
      {:cors_plug, "~> 3.0"},
      {:phoenix_live_reload, "~> 1.5", only: :dev},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:recode, "~> 0.7", only: :dev},
      {:gettext, "~> 0.18"},
      {:hackney, "~> 1.17"},
      {:httpoison, "~> 2.0"},
      {:jason, "~> 1.2"},
      {:joken, "~> 2.6"},
      {:bandit, "~> 1.0"},
      {:plug_static_index_html, "~> 1.0"},
      {:comeonin, "~> 5.3.2"},
      {:bcrypt_elixir, "~> 2.3"},
      {:pow, "~> 1.0"},
      {:email_checker, "~> 0.1.4"},
      {:pow_assent, "~> 0.4.18"},
      {:ssl_verify_fun, "~> 1.1.6"},
      {:open_api_spex, "~> 3.18"},
      {:ex_json_schema, "~> 0.7.4"},
      {:remote_ip, "~> 1.0"},
      {:waffle, "~> 1.1"},
      {:waffle_ecto, "~> 0.0.10"},
      {:ecto_commons, "~> 0.3.4"},
      {:swoosh, "~> 1.3"},
      {:hammer, "~> 6.0"},
      {:scrivener_ecto, "~> 2.7"},
      {:ex_marcel, "~> 0.1.0"}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  # For example, to create, migrate and run the seeds file at once:
  #
  #     $ mix ecto.setup
  #
  # See the documentation for `Mix` for more info on aliases.
  defp aliases do
    [
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      "uro.apigen": [
        "openapi.spec.json --spec Uro.OpenAPI.Specification --pretty --vendor-extensions=false ./frontend/src/__generated/openapi.json"
      ],

      # Not required, fixes warning https://github.com/chaskiq/ex-marcel/pull/2
      "patch.exmarcel": fn _args ->
        path = "deps/ex_marcel/lib/magic.ex"

        patched =
          String.replace(
            File.read!(path),
            "ext |> String.slice(1..-1)",
            "ext |> String.slice(1..-1//1)"
          )

        File.write!(path, patched)
        IO.puts("Module 'ex_marcel' patched successfully!")
      end,
      test: ["ecto.create --quiet", "ecto.migrate", "test"]
    ]
  end
end
