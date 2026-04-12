# lib/my_app_web/ensure_user_not_locked_plug.ex
defmodule Uro.EnsureUserNotLockedPlug do
  alias Plug.Conn
  alias Pow.Config
  alias Pow.Plug

  @doc false
  @spec init(Config.t()) :: atom()
  def init(config) do
    Config.get(config, :error_handler) || raise_no_error_handler!()
  end

  @doc false
  @spec call(Conn.t(), any()) :: Conn.t()
  def call(conn, handler) do
    conn
    |> Plug.current_user()
    |> locked?()
    |> maybe_halt(conn, handler)
  end

  defp locked?(%{locked_at: locked_at}) when not is_nil(locked_at), do: true
  defp locked?(_user), do: false

  defp maybe_halt(true, conn, handler) do
    conn
    |> Plug.delete()
    |> handler.call(:account_locked)
    |> Conn.halt()
  end

  defp maybe_halt(_any, conn, _handler), do: conn

  @spec raise_no_error_handler!() :: no_return()
  defp raise_no_error_handler!,
    do:
      Config.raise_error(
        "No :error_handler configuration option provided. It's required to set this when using #{inspect(__MODULE__)}."
      )
end
