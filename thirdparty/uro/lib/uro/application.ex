defmodule Uro.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  def start(_type, _args) do
    children =
      if System.get_env("MINIMAL_START") == "true",
        do: [],
        else: [
          Uro.Repo,
          Uro.Acl,
          Uro.Manifest,
          Uro.Keys,
          Uro.Endpoint,
          Uro.VSekai.ShardJanitor,
          {Redix, {Application.get_env(:uro, Redix)[:url], [name: :redix]}},
          {Phoenix.PubSub, [name: Uro.PubSub, adapter: Phoenix.PubSub.PG2]},
          ExMarcel.TableWrapper,

          # ExMarcel
          {Task, fn -> Uro.Helpers.Validation.init_extra_extensions() end}
        ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: Uro.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  def config_change(changed, _new, removed) do
    Uro.Endpoint.config_change(changed, removed)
    :ok
  end
end
