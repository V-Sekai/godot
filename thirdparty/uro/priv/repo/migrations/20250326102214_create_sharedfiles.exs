defmodule Uro.Repo.Migrations.CreateSharedFiles do
  use Ecto.Migration

  def change do
    create table(:shared_files, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :name, :string
      add :description, :string
      add :shared_content_data, :string
      add :mime_type, :string
      add :file_size, :integer
      add :checksum, :string
      add :uploader_id, references(:users, type: :uuid)
      add :version, :string
      add :tags, {:array, :string}
      add :is_public, :boolean, default: false, null: false

      timestamps(inserted_at: :created_at)
    end
  end
end
