defmodule Uro.Helpers.User do
  @moduledoc """
  User helper functions, automatically imported by controllers.
  """

  import Uro.Helpers.UUID

  alias Jason.Encoder.Uro.Session
  alias Uro.Accounts
  alias Uro.Accounts.User
  alias Uro.Session

  def user_from_key(conn, "me") do
    current_user(conn)
  end

  def user_from_key(conn, key) do
    {:ok, self} = current_user(conn, optional: true)

    case key do
      _
      when key == self.id or
             key == self.username ->
        self

      _
      when is_uuid(key) ->
        Accounts.get_user!(key)

      _ ->
        Accounts.get_by_username(key)
    end
    |> case do
      %User{} = user ->
        {:ok, user}

      nil ->
        {:error, :not_found, "User not found"}
    end
  end

  def current_user(conn, options \\ [])

  def current_user(%{assigns: %{current_user: nil}}, optional: true),
    do: {:ok, nil}

  def current_user(%{assigns: %{current_user: nil}}, _),
    do: {:error, :invalid_credentials}

  def current_user(%{assigns: %{current_user: self}}, _),
    do: {:ok, self}

  def current_session(conn, options \\ [])

  def current_session(%{assigns: %{current_user: nil}}, optional: true),
    do: {:ok, nil}

  def current_session(%{assigns: %{current_user: nil}}, _),
    do: {:error, :invalid_credentials}

  def current_session(
        %{
          assigns: %{
            current_user: self,
            access_token: access_token,
            access_token_expires_in: expires_in
          }
        },
        _
      ),
      do:
        {:ok,
         %Session{
           access_token: access_token,
           expires_in: expires_in,
           token_type: "Bearer",
           user: self
         }}

  def user_confirmed_email(%User{email_confirmed_at: nil}),
    do: {:error, :forbidden, "Email not confirmed"}

  def user_confirmed_email(%User{} = user), do: {:ok, user}

  def validate_as_yourself(changeset, user, field_key \\ :user_id) do
    Ecto.Changeset.validate_inclusion(changeset, field_key, [user.id],
      message: "must be yourself"
    )
  end

  def is_session_user?(conn) do
    user = user_from_key(conn, "me")

    case user do
      {:ok, _result} -> true
      {:error, _reason} -> false
      _ -> false
    end
  end
end
