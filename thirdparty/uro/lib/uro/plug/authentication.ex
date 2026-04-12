defmodule Uro.Plug.Authentication do
  @moduledoc false

  use Pow.Plug.Base

  alias Plug.Conn
  alias Pow.Config
  alias Pow.Plug
  alias PowPersistentSession.Store.PersistentSessionCache
  alias Uro.Accounts.User

  # The lifetime of the session.
  @session_lifetime :timer.hours(168)

  # Renew the session if it expires in less than this.
  @session_renewal @session_lifetime - :timer.hours(1)

  @impl true
  def fetch(%Conn{} = conn, config) do
    store_config = store_config(config)

    with {:ok, signed_access_token} <- fetch_access_token(conn),
         {:ok, access_token} <- verify_token(conn, signed_access_token, config),
         {user, metadata} <- PersistentSessionCache.get(store_config, access_token),
         expires_in <- metadata[:expires_at] |> DateTime.diff(DateTime.utc_now(), :millisecond),
         conn <-
           conn
           |> Conn.assign(:access_token, signed_access_token)
           |> Conn.assign(:access_token_unsigned, access_token)
           |> Conn.assign(
             :access_token_expires_in,
             expires_in
           ),
         {conn, user} <- maybe_renew(conn, {user, metadata}, config) do
      {conn, user}
    else
      _ -> {conn, nil}
    end
  end

  defp maybe_renew(%Conn{} = conn, {%User{} = user, metadata}, config) do
    expires_in =
      metadata[:expires_at]
      |> DateTime.diff(DateTime.utc_now(), :millisecond)

    with true <- expires_in < @session_renewal,
         {conn, user} <- load_and_create_session(conn, {[id: user.id], metadata}, config) do
      {conn, user}
    else
      _ -> {conn, user}
    end
  end

  @impl true
  def create(%Conn{} = conn, %User{} = user, config) do
    store_config =
      store_config(config)
      |> Keyword.put(:ttl, @session_lifetime)

    access_token = Pow.UUID.generate()
    signed_access_token = sign_token(conn, access_token, config)

    conn =
      conn
      |> Conn.assign(:access_token, signed_access_token)
      |> Conn.assign(:access_token_unsigned, access_token)
      |> Conn.assign(:access_token_expires_in, store_config[:ttl])
      |> Conn.put_resp_cookie(
        "session",
        signed_access_token,
        max_age: store_config[:ttl]
      )
      |> Conn.register_before_send(fn conn ->
        expires_at = DateTime.add(DateTime.utc_now(), store_config[:ttl], :millisecond)

        PersistentSessionCache.put(
          store_config,
          access_token,
          {user,
           [
             expires_at: expires_at
           ]}
        )

        conn
      end)

    {conn, user}
  end

  @impl true
  @spec delete(Conn.t(), Config.t()) :: Conn.t()
  def delete(conn, config) do
    store_config = store_config(config)

    with {:ok, signed_access_token} <- fetch_access_token(conn),
         {:ok, access_token} <- verify_token(conn, signed_access_token, config),
         :ok <- PersistentSessionCache.delete(store_config, access_token),
         conn <- Conn.delete_resp_cookie(conn, "session") do
      conn
    else
      _ -> conn
    end
  end

  defp load_and_create_session(conn, {clauses, _metadata}, config) do
    case Pow.Operations.get_by(clauses, config) do
      nil -> {conn, nil}
      user -> create(conn, user, config)
    end
  end

  defp sign_token(conn, token, config) do
    Plug.sign_token(conn, signing_salt(), token, config)
  end

  defp signing_salt(), do: Atom.to_string(__MODULE__)

  defp fetch_access_token(conn) do
    case Conn.get_req_header(conn, "authorization") do
      ["Bearer" <> " " <> access_token | _] ->
        {:ok, access_token}

      [access_token | _] ->
        {:ok, access_token}

      _ ->
        case conn.cookies do
          %{"session" => access_token} when is_binary(access_token) ->
            {:ok, access_token}

          _ ->
            :error
        end
    end
  end

  defp verify_token(conn, token, config),
    do: Plug.verify_token(conn, signing_salt(), token, config)

  defp store_config(config) do
    backend = Config.get(config, :cache_store_backend)
    [backend: backend, pow_config: config]
  end
end
