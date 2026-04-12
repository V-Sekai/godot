import Config

# Tests run against the Oxide CockroachDB 22.1 fork served by the
# docker-compose.yml `database` service. Bring it up before `mix test`:
#
#   docker compose -f thirdparty/uro/docker-compose.yml up -d database
#
# CRDB's --accept-sql-without-tls mode lets the Postgrex adapter connect
# unencrypted on port 26257. The vsekai user and database are seeded by the
# V-Sekai cockroach image on first boot, and the Ecto.Adapters.SQL.Sandbox
# gives us per-test transactional isolation on top.
config :uro, Uro.Repo,
  show_sensitive_data_on_connection_error: true,
  url: System.get_env("TEST_DATABASE_URL"),
  username: "vsekai",
  password: "vsekai",
  hostname: "127.0.0.1",
  port: 26257,
  database: "vsekai_test",
  stacktrace: true,
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: 10
