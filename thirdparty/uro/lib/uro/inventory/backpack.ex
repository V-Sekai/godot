defmodule Uro.Inventory.Backpack do
  alias Uro.Accounts.User
  import Ecto.Changeset
  use Ecto.Schema

  defmodule Join do
    use Ecto.Schema
    import Ecto.Query

    @primary_key {:id, :binary_id, autogenerate: true}
    @foreign_key_type :binary_id
    schema "backpack_join" do
      belongs_to :backpack, Uro.Inventory.Backpack

      belongs_to :map, Uro.UserContent.Map
      belongs_to :prop, Uro.UserContent.Prop
      belongs_to :avatar, Uro.UserContent.Avatar

      timestamps()
    end
  end

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "backpacks" do
    belongs_to :owner, User, foreign_key: :owner_id, type: :binary_id

    many_to_many :maps, Uro.UserContent.Map, join_through: Join
    many_to_many :avatars, Uro.UserContent.Avatar, join_through: Join
    many_to_many :props, Uro.UserContent.Prop, join_through: Join

    timestamps()
  end
end
