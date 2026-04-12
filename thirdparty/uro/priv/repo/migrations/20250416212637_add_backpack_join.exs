defmodule Uro.Repo.Migrations.AddBackpackJoin do
  use Ecto.Migration

  def change do
    create table(:backpack_join, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :backpack_id, references(:users, type: :binary_id), null: false

      add :map_id, references(:maps, type: :binary_id)
      add :avatar_id, references(:avatars, type: :binary_id)
      add :prop_id, references(:props, type: :binary_id)
      timestamps()
    end
  end
end
