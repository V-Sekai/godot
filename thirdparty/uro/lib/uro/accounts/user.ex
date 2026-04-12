defmodule Uro.Accounts.User do
  @moduledoc """
  The User schema.
  """

  use Ecto.Schema

  @pow_config user_id_field: :email,
              password_hash_verify: {&Bcrypt.hash_pwd_salt/1, &Bcrypt.verify_pass/2}

  def pow_config(), do: @pow_config

  use Pow.Ecto.Schema, @pow_config

  use Pow.Extension.Ecto.Schema,
    extensions: [PowResetPassword]

  use PowAssent.Ecto.Schema

  import Ecto.Query, only: [from: 2]
  import Ecto.Changeset
  import Uro.Helpers.Changeset

  alias OpenApiSpex.Schema
  alias Uro.Accounts.UserPrivilegeRuleset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  @derive {Phoenix.Param, key: :id}
  @derive {Inspect,
           only: [
             :id,
             :username,
             :display_name
           ]}

  @user_statuses [
    :online,
    :offline,
    :away,
    :busy,
    :invisible
  ]

  def user_statuses(), do: @user_statuses

  schema "users" do
    field(:username, :string)
    field(:display_name, :string)

    field(:icon, :string)
    field(:banner, :string)

    field(:biography, :string, default: "")

    field(:status, Ecto.Enum,
      values: @user_statuses,
      default: :offline
    )

    field(:status_message, :string)

    field(:email, :string)
    field(:email_confirmed_at, :utc_datetime)
    field(:email_notifications, :boolean)

    has_one(:user_privilege_ruleset, Uro.Accounts.UserPrivilegeRuleset, foreign_key: :user_id)

    has_many(:uploaded_avatars, Uro.UserContent.Avatar, foreign_key: :uploader_id)
    has_many(:uploaded_maps, Uro.UserContent.Map, foreign_key: :uploader_id)
    has_many(:uploaded_props, Uro.UserContent.Prop, foreign_key: :uploader_id)

    # Inventory

    has_many :backpacks, Uro.Inventory.Backpack, foreign_key: :owner_id

    has_many :avatars, through: [:backpacks, :avatars]
    has_many :maps, through: [:backpacks, :maps]
    has_many :props, through: [:backpacks, :props]

    ######

    many_to_many(:friendships, Uro.UserRelations.Friendship,
      join_through: "friendships",
      join_keys: [user_id: :id, friend_id: :id]
    )

    has_many(:hosted_shards, Uro.VSekai.Shard, foreign_key: :user_id)

    has_many(:identity_proofs_from, Uro.UserRelations.IdentityProof, foreign_key: :user_from_id)
    has_many(:identity_proofs_to, Uro.UserRelations.IdentityProof, foreign_key: :user_to_id)

    field(:locked_at, :utc_datetime)

    pow_user_fields()

    timestamps(inserted_at: :created_at)
  end

  @user_status_json_schema %Schema{
    title: "UserStatus",
    type: :string,
    enum: @user_statuses
  }

  def user_status_json_schema(), do: @user_status_json_schema

  @sensitive_json_schema %Schema{
    title: "SensitiveUser",
    description: "A user, with sensitive information.",
    type: :object,
    required: [
      :id,
      :username,
      :display_name,
      :icon,
      :banner,
      :biography,
      :status,
      :status_message,
      :email,
      :email_confirmed_at,
      :email_notifications,
      :user_privilege_ruleset,
      :created_at
    ],
    properties: %{
      id: %Schema{
        title: "UserID",
        type: :string,
        format: :uuid
      },
      username: %Schema{
        title: "Username",
        type: :string,
        minLength: 3,
        maxLength: 16
      },
      display_name: %Schema{
        type: :string
      },
      icon: %Schema{
        type: :string,
        nullable: true
      },
      banner: %Schema{
        type: :string,
        nullable: true
      },
      biography: %Schema{
        type: :string
      },
      status: @user_status_json_schema,
      status_message: %Schema{
        type: :string,
        nullable: true
      },
      email: %Schema{
        type: :string,
        format: :email
      },
      email_confirmed_at: %Schema{
        type: :string,
        format: :"date-time",
        nullable: true
      },
      email_notifications: %Schema{
        type: :boolean
      },
      user_privilege_ruleset: UserPrivilegeRuleset.json_schema(),
      created_at: %Schema{
        type: :string,
        format: :"date-time"
      }
    }
  }

  def sensitive_json_schema(), do: @sensitive_json_schema

  def to_sensitive_json_schema(%__MODULE__{} = user),
    do: %{
      id: user.id,
      username: user.username,
      display_name: user.display_name,
      icon: user.icon,
      banner: user.banner,
      biography: user.biography,
      status: user.status,
      status_message: user.status_message,
      email: user.email,
      email_confirmed_at: user.email_confirmed_at,
      email_notifications: user.email_notifications,
      user_privilege_ruleset: UserPrivilegeRuleset.to_json_schema(user.user_privilege_ruleset),
      created_at: user.created_at
    }

  def to_sensitive_json_schema(_), do: nil

  @limited_json_schema %Schema{
    title: "LimitedUser",
    description: "A user object for public consumption, excluding sensitive information.",
    type: :object,
    required: [
      :id,
      :username,
      :display_name,
      :icon,
      :banner,
      :biography,
      :status,
      :status_message,
      :created_at
    ],
    properties: %{
      id: @sensitive_json_schema.properties[:id],
      username: @sensitive_json_schema.properties[:username],
      display_name: @sensitive_json_schema.properties[:display_name],
      icon: @sensitive_json_schema.properties[:icon],
      banner: @sensitive_json_schema.properties[:banner],
      biography: @sensitive_json_schema.properties[:biography],
      status: @sensitive_json_schema.properties[:status],
      status_message: @sensitive_json_schema.properties[:status_message],
      created_at: @sensitive_json_schema.properties[:created_at]
    }
  }

  def limited_json_schema(), do: @limited_json_schema

  def to_limited_json_schema(nil), do: nil

  def to_limited_json_schema(%__MODULE__{} = user),
    do: %{
      id: user.id,
      username: user.username,
      display_name: user.display_name,
      icon: user.icon,
      banner: user.banner,
      biography: user.biography,
      status: user.status,
      status_message: user.status_message,
      created_at: user.created_at
    }

  @json_schema %Schema{
    title: "User",
    oneOf: [
      @limited_json_schema,
      @sensitive_json_schema
    ]
  }

  def json_schema(), do: @json_schema

  def to_json_schema(nil, _), do: nil
  def to_json_schema(list, conn) when is_list(list), do: Enum.map(list, &to_json_schema(&1, conn))

  def to_json_schema(%__MODULE__{id: user_id} = user, %{assigns: %{current_user: %{id: user_id}}}),
    do: to_sensitive_json_schema(user)

  def to_json_schema(%__MODULE__{} = user, _), do: to_limited_json_schema(user)

  def loose_key_json_schema(),
    do: %Schema{
      title: "LooseUserKey",
      description: """
      A loose representation of a user, can any of the following:

      * The literal `me` string, representing the current user,
      * A user's username,
      * Or their ID.
      """,
      default: "me",
      oneOf: [
        %Schema{type: :string, enum: ["me"]},
        @sensitive_json_schema.properties[:id],
        @sensitive_json_schema.properties[:username]
      ]
    }

  def admin?(%{user_privilege_ruleset: %{is_admin: is_admin}}), do: is_admin
  def admin?(_), do: false

  def lock_changeset(user_or_changeset) do
    changeset = change(user_or_changeset)
    locked_at = DateTime.truncate(DateTime.utc_now(), :second)

    case get_field(changeset, :locked_at) do
      nil -> change(changeset, locked_at: locked_at)
      _any -> add_error(changeset, :locked_at, "already set")
    end
  end

  def unlock_changeset(user_or_changeset) do
    changeset = change(user_or_changeset)

    case get_field(changeset, :locked_at) do
      nil -> add_error(changeset, :locked_at, "already set")
      _any -> change(changeset, locked_at: nil)
    end
  end

  @update_json_schema %Schema{
    type: :object,
    properties: %{
      display_name: @sensitive_json_schema.properties[:display_name],
      username: @sensitive_json_schema.properties[:username],
      biography: @sensitive_json_schema.properties[:biography],
      email_notifications: @sensitive_json_schema.properties[:email_notifications],
      status_message: @sensitive_json_schema.properties[:status_message]
    }
  }

  def update_json_schema(), do: @update_json_schema

  def user_custom_changeset(user_or_changeset, attrs) do
    user_or_changeset
    |> cast(attrs, [
      :display_name,
      :username,
      :biography,
      :email_notifications,
      :status_message
    ])
    |> validate_required([:display_name, :username])
    |> validate_display_name(:display_name)
    |> validate_username(:username)
    |> unique_constraint(:username)
  end

  def changeset(user_or_changeset, attrs) do
    attrs =
      case attrs do
        %{"password" => password} ->
          # Asking the user to confirm their password is an antiqued practice, but Pow expects it.
          # https://estevanmaito.me/blog/killing-the-confirm-password-field-is-not-enough
          Map.put(attrs, "password_confirmation", password)

        _ ->
          attrs
      end

    user_or_changeset
    |> pow_changeset(attrs)
    # |> pow_extension_changeset(attrs)
    |> user_custom_changeset(attrs)
  end

  def user_identity_changeset(user_or_changeset, user_identity, attrs, user_id_attrs) do
    %{"userinfo" => userinfo} = user_identity

    changeset =
      pow_assent_user_identity_changeset(user_or_changeset, user_identity, attrs, user_id_attrs)

    username =
      get_field(changeset, :username) || attrs["username"] || userinfo["preferred_username"]

    unique_constraint(changeset, :username)
    |> put_change(:username, username)
    |> validate_username(:username)
    |> put_change(
      :display_name,
      get_field(changeset, :display_name) || userinfo["name"] || userinfo["preferred_username"] ||
        username
    )
    |> validate_display_name(:display_name)
    |> put_change(
      :icon,
      get_field(changeset, :icon) || userinfo["picture"]
    )
  end

  def admin_changeset(user_or_changeset, attrs) do
    user_or_changeset
    |> pow_user_id_field_changeset(attrs)
    |> pow_password_changeset(attrs)
    |> user_custom_changeset(attrs)
    |> cast(attrs, [:display_name])
  end

  def confirm_email_changeset(changeset) do
    change(
      changeset,
      %{
        email_confirmed_at: DateTime.truncate(DateTime.utc_now(), :second)
      }
    )
  end

  def update_email_changeset(changeset, email, send_confirmation: true) do
    changeset
    |> cast(%{email: email}, [:email])
    |> validate_different(:email)
    |> validate_email(:email)
    |> put_change(:email_confirmed_at, nil)
  end

  def update_email_changeset(changeset, email, send_confirmation: false) do
    changeset
    |> cast(%{email: email}, [:email])
    |> validate_different(:email)
    |> validate_email(:email)
    |> confirm_email_changeset()
  end

  def validate_current_password(changeset, %{password_hash: password_hash}) do
    if(password_hash == nil, do: changeset, else: validate_required(changeset, :current_password))
    |> validate_change(:current_password, fn _, current_password ->
      Pow.Ecto.Schema.Changeset.verify_password(
        %{password_hash: password_hash},
        current_password,
        pow_config()
      )
      |> case do
        true -> []
        false -> [{:current_password, "is invalid"}]
      end
    end)
  end

  def validate_display_name(changeset, field) when is_atom(field) do
    validate_length(
      changeset,
      field,
      min: 3,
      max: 32
    )
  end

  def validate_username(changeset, field) when is_atom(field) do
    case Map.get(changeset.changes, field) do
      nil ->
        changeset

      username ->
        changeset
        |> put_change(field, String.downcase(username))
        |> validate_length(
          field,
          min: 3,
          max: 16
        )
        |> validate_format(
          field,
          ~r/^[a-z0-9_]+$/,
          message: "must contain only lowercase letters, numbers, and underscores"
        )
    end
  end

  def validate_email(changeset, field) when is_atom(field) do
    EctoCommons.EmailValidator.validate_email(changeset, field)
  end

  def admin_search(query, search_term) do
    wildcard_search = "%#{search_term}%"

    from(user in query,
      where: ilike(user.username, ^wildcard_search),
      or_where: ilike(user.display_name, ^wildcard_search),
      or_where: ilike(user.email, ^wildcard_search)
    )
  end
end
