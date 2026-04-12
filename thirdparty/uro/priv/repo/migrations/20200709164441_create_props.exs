defmodule Uro.Repo.Migrations.CreateProps do
  use Ecto.Migration

  def change do
    create table(:props, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :name, :string
      add :description, :text
      add :user_content_data, :string
      add :uploader_id, references(:users, type: :uuid)
      add :user_content_preview, :string
      add :is_public, :boolean, default: false, null: false

      timestamps()
    end
  end
end
