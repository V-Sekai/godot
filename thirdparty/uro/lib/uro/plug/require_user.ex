defmodule Uro.Plug.RequireUser do
  import Plug.Conn

  @doc false
  def init(options), do: options

  @doc false
  @spec call(Conn.t(), any()) :: Conn.t()
  def call(conn, _opts) do
    conn
    |> Uro.Helpers.User.is_session_user?()
    |> maybe_halt(conn)
  end

  @doc false
  defp maybe_halt(true, conn), do: conn

  @doc false
  defp maybe_halt(false, conn) do
    conn
    |> put_status(403)
    |> send_resp(403, "Unauthorized 403")
    |> halt()
  end
end
