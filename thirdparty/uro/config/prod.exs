import Config

# Unused (no static files)
# config :uro, Uro.Endpoint, cache_static_manifest: "priv/static/cache_manifest.json"

config :uro, Uro.Mailer,
  adapter: Swoosh.Adapters.Sendgrid,
  api_key: System.get_env("SENDGRID_API_KEY", "")

# Do not print debug messages in production.
config :logger, level: :info
