defmodule Uro.Plug.ChooseAuth do
  import Plug.Conn

  def init(default), do: default

  def call(conn, opts) do
    if has_authorization_header?(conn) do
      # Game client
      token = get_req_header(conn, "authorization")

      case token do
        [] ->
          conn

        [auth_header] ->
          assign(conn, :signed_access_token, token)
      end

      Uro.Plug.RequireUser.call(conn, Uro.Plug.RequireUser.init(opts))
    else
      opts = Keyword.merge(opts, error_handler: Uro.FallbackController)
      Pow.Plug.RequireAuthenticated.call(conn, Pow.Plug.RequireAuthenticated.init(opts))
    end
  end

  defp has_authorization_header?(conn) do
    case get_req_header(conn, "authorization") do
      [] -> false
      [_ | _] -> true
    end
  end
end
