defmodule Uro.Repo.Migrations.CreateShards do
  use Ecto.Migration

  def change do
    create table(:shards) do
      add :address, :string
      add :port, :integer
      add :name, :string

      add :current_users, :integer
      add :max_users, :integer

      add :map, :string
      add :user_id, references(:users, type: :uuid, on_delete: :delete_all)

      timestamps()
    end
  end
end
