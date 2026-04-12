import Config
Code.require_file("config/helpers.exs")
Code.ensure_loaded!(Uro.Config.Helpers)
alias Uro.Config.Helpers

config :uro, Uro.Endpoint,
  debug_errors: true,
  code_reloader: true,
  check_origin: false

config :uro, Uro.Mailer, adapter: Swoosh.Adapters.Local

config :open_api_spex, :cache_adapter, OpenApiSpex.Plug.NoneCache

config :logger, :console, format: "[$level] $message\n"
config :logger, level: :debug

config :phoenix, :stacktrace_depth, 20
config :phoenix, :plug_init_mode, :runtime

config :uro, Uro.Repo,
  show_sensitive_data_on_connection_error: true,
  pool_size: 10,
  url: System.get_env("DATABASE_URL"),
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  database: "uro-dev",
  stacktrace: true,
  show_sensitive_data_on_connection_error: true,
  pool_size: 10

redis_url = Helpers.get_env("REDIS_URL", nil)
config :uro, Redix, url: if(redis_url, do: redis_url, else: "redis://localhost:6379")
