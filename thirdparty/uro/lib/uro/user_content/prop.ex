defmodule Uro.UserContent.Prop do
  import Ecto.Changeset

  use Uro.UserContent.UserContent

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  @derive {Phoenix.Param, key: :id}
  schema "props" do
    user_content_fields()

    timestamps()
  end

  @doc false
  def changeset(prop, attrs) do
    user_content_changeset(prop, attrs)
  end

  @doc false
  def upload_changeset(prop, attrs) do
    user_content_upload_changeset(prop, attrs)
  end
end
