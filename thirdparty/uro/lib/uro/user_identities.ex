defmodule Uro.UserIdentities do
  use PowAssent.Ecto.UserIdentities.Context,
    repo: Uro.Repo,
    user: Uro.Accounts.User

  alias Uro.Accounts
  alias Uro.Repo

  def create_user(user_identity_params, user_params, user_id_params) do
    Repo.transaction(fn ->
      with {:ok, user} <-
             pow_assent_create_user(user_identity_params, user_params, user_id_params),
           {:ok, user} <- Accounts.create_user_associations(user) do
        user
      else
        {:error, reason} ->
          Repo.rollback(reason)

        any ->
          Repo.rollback(any)
      end
    end)
  end
end
