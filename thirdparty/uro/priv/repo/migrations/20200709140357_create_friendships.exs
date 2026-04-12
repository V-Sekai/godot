defmodule Uro.Repo.Migrations.CreateFriendships do
  use Ecto.Migration

  def change do
    create table(:friendships, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :user_id, references(:users, on_delete: :delete_all, type: :uuid), null: false
      add :friend_id, references(:users, on_delete: :delete_all, type: :uuid), null: false
      add :accepted_at, :utc_datetime

      timestamps()
    end
  end
end
