defmodule Uro.VSekai.Shard do
  use Ecto.Schema

  import Ecto.Changeset

  alias OpenApiSpex.Schema
  alias Uro.Accounts.User

  @derive {Jason.Encoder,
           only: [
             :user,
             :address,
             :port,
             :map,
             :name,
             :current_users,
             :max_users
           ]}

  schema "shards" do
    belongs_to(:user, Uro.Accounts.User, foreign_key: :user_id, type: :binary_id)

    field(:address, :string)
    field(:port, :integer)
    field(:map, :string)
    field(:name, :string)

    field(:current_users, :integer, default: 0)
    field(:max_users, :integer, default: 32)

    timestamps()
  end

  @json_schema %Schema{
    title: "Shard",
    type: :object,
    required: [
      :id,
      :username,
      :display_name
    ],
    properties: %{
      user: User.json_schema(),
      address: %Schema{
        type: :string
      },
      port: %Schema{
        type: :integer
      },
      map: %Schema{
        type: :string
      },
      name: %Schema{
        type: :string
      },
      current_users: %Schema{
        type: :integer
      },
      max_users: %Schema{
        type: :integer
      }
    }
  }

  def json_schema, do: @json_schema

  def to_json_schema(%__MODULE__{} = shard) do
    %{
      user: User.to_limited_json_schema(shard.user),
      address: to_string(shard.address),
      port: shard.port,
      map: to_string(shard.map),
      name: to_string(shard.name)
    }
  end

  @doc false
  def changeset(shard, attrs) do
    shard
    |> cast(attrs, [:user_id, :address, :port, :map, :name, :current_users, :max_users])
    |> validate_required([:address, :port, :map, :name])
  end
end
