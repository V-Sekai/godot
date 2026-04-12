defmodule Uro.Repo.Migrations.CreateIdentityProofs do
  use Ecto.Migration

  def change do
    create table(:identity_proofs, primary_key: false) do
      add :id, :uuid, primary_key: true

      add :user_from_id, references(:users, type: :uuid, on_delete: :delete_all)
      add :user_to_id, references(:users, type: :uuid, on_delete: :delete_all)

      timestamps()
    end
  end
end
