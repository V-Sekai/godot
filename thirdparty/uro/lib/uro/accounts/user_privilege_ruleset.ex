defmodule Uro.Accounts.UserPrivilegeRuleset do
  @moduledoc """
  A user's privilege ruleset, defining what they can do.
  """

  use Ecto.Schema

  import Ecto.Changeset

  alias OpenApiSpex.Schema

  schema "user_privilege_rulesets" do
    belongs_to(:user, Uro.Accounts.User, foreign_key: :user_id, type: :binary_id)

    field(:is_admin, :boolean, default: false)
    field(:can_upload_avatars, :boolean, default: false)
    field(:can_upload_maps, :boolean, default: false)
    field(:can_upload_props, :boolean, default: false)
    field(:can_upload_shared_files, :boolean, default: false)

    timestamps()
  end

  # Don't show 'can_upload_shared_files' privilege status to clients in json
  @json_schema %Schema{
    title: "UserPrivilegeRuleset",
    description: @moduledoc,
    type: :object,
    required: [
      :is_admin,
      :can_upload_avatars,
      :can_upload_maps,
      :can_upload_props
    ],
    properties: %{
      is_admin: %Schema{type: :boolean},
      can_upload_avatars: %Schema{type: :boolean},
      can_upload_maps: %Schema{type: :boolean},
      can_upload_props: %Schema{type: :boolean}
    }
  }

  def json_schema(), do: @json_schema

  def to_json_schema(%__MODULE__{} = user_privilege_ruleset),
    do: %{
      is_admin: user_privilege_ruleset.is_admin,
      can_upload_avatars: user_privilege_ruleset.can_upload_avatars,
      can_upload_maps: user_privilege_ruleset.can_upload_maps,
      can_upload_props: user_privilege_ruleset.can_upload_props
    }

  def to_json_schema(_), do: nil

  def admin_changeset(user_privilege_ruleset_or_changeset, attrs) do
    cast(user_privilege_ruleset_or_changeset, attrs, [
      :user_id,
      :is_admin,
      :can_upload_avatars,
      :can_upload_maps,
      :can_upload_props,
      :can_upload_shared_files
    ])
  end
end
