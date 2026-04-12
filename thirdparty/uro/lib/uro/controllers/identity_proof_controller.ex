defmodule Uro.IdentityProofController do
  use Uro, :controller

  alias Uro.Error

  @spec create(Conn.t(), map()) :: Conn.t()
  def create(conn, %{"identity_proof" => identity_proof_params}) do
    if !Map.has_key?(identity_proof_params, "user_to") do
      conn
      |> put_status(500)
      |> json(%{error: %{status: 500, message: "No recipiant for identity_proof"}})
    else
      user_from = conn.assigns[:current_user]
      user_to = Uro.Accounts.get_user!(Map.get(identity_proof_params, "user_to"))

      if user_to do
        user_from
        |> Uro.UserRelations.create_identity_proof(user_to)
        |> case do
          {:ok, identity_proof} ->
            json(conn, %{id: identity_proof.id})

          {:error, %Ecto.Changeset{} = changeset} ->
            errors = Ecto.Changeset.traverse_errors(changeset, &Error.translate/1)

            conn
            |> put_status(500)
            |> json(%{
              error: %{status: 500, message: "Couldn't create identity_proof", errors: errors}
            })
        end
      else
        conn
        |> put_status(500)
        |> json(%{error: %{status: 500, message: "Recipiant id invalid"}})
      end
    end
  end

  def show(conn, %{"id" => id}) do
    id
    |> Uro.UserRelations.get_identity_proof_as!(conn.assigns[:current_user])
    |> case do
      nil ->
        put_status(
          conn,
          400
        )

      identity_proof ->
        conn
        |> put_status(200)
        |> json(%{data: %{identity_proof: identity_proof}})
    end
  end
end
