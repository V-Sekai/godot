defmodule Uro.Helpers.Auth do
  @moduledoc """
  Session & authentication helper functions.
  """

  require Pow.Phoenix.Router
  require PowEmailConfirmation.Phoenix.Router

  def get_user_privilege_ruleset(user) do
    user
    |> is_map()
    |> case do
      true ->
        user
        |> Uro.Repo.preload([:user_privilege_ruleset])
        |> Map.get(:user_privilege_ruleset)

      false ->
        nil
    end
  end

  def verify_confirmed_or_send_confirmation_email(conn) do
    conn
    |> PowEmailConfirmation.Plug.email_unconfirmed?()
    |> case do
      true ->
        Pow.Plug.current_user(conn)
        |> PowEmailConfirmation.Phoenix.ControllerCallbacks.send_confirmation_email(conn)

        {:error, conn}

      false ->
        {:ok, conn}
    end
  end

  def signed_in?(conn) do
    if Pow.Plug.current_user(conn) do
      true
    end
  end

  def get_current_user(conn) do
    Pow.Plug.current_user(conn)
  end

  def session_username(conn) do
    user = get_current_user(conn)

    if user do
      if user.username do
        user.username
      else
        "[NULL]"
      end
    end
  end

  def session_display_name(conn) do
    user = get_current_user(conn)

    if user do
      if user.display_name do
        user.display_name
      else
        "[NULL]"
      end
    end
  end

  @doc false
  defmacro __using__(_config) do
    quote do
      import unquote(__MODULE__), only: [validate_login: 2]
    end
  end
end
