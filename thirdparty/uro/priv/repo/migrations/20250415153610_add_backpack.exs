defmodule Uro.Repo.Migrations.AddBackpack do
  use Ecto.Migration

  def change do
    create table(:backpacks, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :owner_id, references(:users, type: :binary_id), null: false

      timestamps()
    end
  end
end
