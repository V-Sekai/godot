defmodule Uro.FallbackController do
  @moduledoc """
  Translates controller action results into valid `Plug.Conn` responses.

  See `Phoenix.Controller.action_fallback/1` for more details.
  """

  use Uro, :controller

  alias Ecto.Changeset
  alias Uro.Error

  def call(conn, errors) when is_list(errors) do
    json_error(
      conn,
      code: :bad_request,
      properties:
        errors
        |> Enum.map(fn %{path: path} = error ->
          {Enum.join(path, "."), OpenApiSpex.Cast.Error.message(error)}
        end)
        |> Map.new()
    )
  end

  def call(conn, {:error, %Ecto.Changeset{} = changeset}) do
    properties = Changeset.traverse_errors(changeset, &Error.translate/1)

    json_error(
      conn,
      code: :bad_request,
      properties: properties
    )
  end

  # Used by: Pow.Plug.RequireAuthenticated.
  def call(conn, :not_authenticated), do: call(conn, {:error, :invalid_credentials})

  def call(conn, {:error, :invalid_credentials}) do
    json_error(conn, code: :unauthorized, message: "Invalid credentials")
  end

  def call(conn, {:error, :insufficient_permission}) do
    json_error(conn, code: :forbidden, message: "Insufficient permission")
  end

  def call(conn, {:error, :account_locked}) do
    json_error(conn, code: :locked, message: "Account unavailable")
  end

  def call(conn, {:error, status}) when is_atom(status) do
    json_error(conn, code: status)
  end

  def call(conn, {:error, status, message}) when is_atom(status) and is_binary(message) do
    json_error(conn, code: status, message: message)
  end
end
