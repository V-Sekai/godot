defmodule Uro.Repo.Migrations.CreateEvents do
  use Ecto.Migration

  def change do
    create table(:events) do
      add :description, :string
      add :name, :string

      add :start_date, :utc_datetime
      add :end_date, :utc_datetime

      timestamps()
    end
  end
end
