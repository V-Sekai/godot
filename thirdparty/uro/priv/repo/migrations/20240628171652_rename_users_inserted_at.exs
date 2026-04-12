defmodule Uro.Repo.Migrations.RenameUsersInsertedAt do
  use Ecto.Migration

  def change do
    rename table(:users), :inserted_at, to: :created_at
  end
end
