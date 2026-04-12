defmodule Uro.Repo.Migrations.CreateUsers do
  use Ecto.Migration

  def change do
    create table(:users, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :email, :string, null: false
      add :password_hash, :string

      add :username, :string
      add :display_name, :string

      add :email_notifications, :bool, default: false

      add :icon, :string
      add :banner, :string
      add :biography, :string
      add :status, :string
      add :status_message, :string

      add :email_confirmed_at, :utc_datetime

      add :locked_at, :utc_datetime

      timestamps()
    end
  end
end
