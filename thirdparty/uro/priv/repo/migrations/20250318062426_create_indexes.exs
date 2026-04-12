defmodule Uro.Repo.Migrations.CreateIndexes do
  use Ecto.Migration

  # Warning: Transaction disabled for CockroachDB compatibility.
  # This migration must be manually reverted if it fails.

  @disable_ddl_transaction true
  def change do
    # users
    create unique_index(:users, [:username])
    create unique_index(:users, [:email])

    # user_identities
    create unique_index(:user_identities, [:uid, :provider])

    # identity_proofs
    create index(:identity_proofs, [:user_from_id, :user_to_id])

    # friendships
    create unique_index(:friendships, [:user_id, :friend_id])
  end
end
