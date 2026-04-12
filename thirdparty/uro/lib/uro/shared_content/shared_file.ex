defmodule Uro.SharedContent.SharedFile do
  import Ecto.Changeset
  use Uro.SharedContent.SharedContent

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  @derive {Phoenix.Param, key: :id}
  schema "shared_files" do
    shared_content_fields()

    timestamps(inserted_at: :created_at)
  end

  @doc false
  def changeset(shared_file, attrs) do
    shared_content_changeset(shared_file, attrs)
  end

  @doc false
  def upload_changeset(shared_file, attrs) do
    shared_content_upload_changeset(shared_file, attrs)
  end
end
