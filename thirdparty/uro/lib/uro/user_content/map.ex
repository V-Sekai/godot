defmodule Uro.UserContent.Map do
  import Ecto.Changeset

  use Uro.UserContent.UserContent

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  @derive {Phoenix.Param, key: :id}
  schema "maps" do
    user_content_fields()

    timestamps()
  end

  @doc false
  def changeset(map, attrs) do
    user_content_changeset(map, attrs)
  end

  @doc false
  def upload_changeset(map, attrs) do
    user_content_upload_changeset(map, attrs)
  end
end
