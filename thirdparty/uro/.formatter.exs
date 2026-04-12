[
  import_deps: [:ecto, :phoenix, :open_api_spex],
  inputs: [
    "*.{ex,exs}",
    "priv/*/seeds.exs",
    "{config,lib,test}/**/*.{ex,exs,eex}"
  ],
  subdirectories: ["priv/*/migrations"],
  plugins: [Recode.FormatterPlugin]
]
