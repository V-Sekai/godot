defmodule Uro.UserRelations.Friendship do
  @moduledoc """
  A friendship between two users.
  """

  use Ecto.Schema

  import Ecto.Changeset

  alias OpenApiSpex.Schema
  alias Uro.Accounts.User
  alias Uro.Repo
  alias Uro.UserRelations.Friendship

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  @derive {Phoenix.Param, key: :id}

  @derive {Jason.Encoder,
           only: [
             :status,
             :accepted_at
           ]}

  schema "friendships" do
    belongs_to(:user, User, foreign_key: :user_id)
    belongs_to(:friend, User, foreign_key: :friend_id)

    field(:status, :string, virtual: true)
    field(:accepted_at, :utc_datetime)

    timestamps()
  end

  @friend_status_json_schema %Schema{
    title: "FriendStatus",
    type: :string,
    enum: [
      :none,
      :received,
      :sent,
      :accepted
    ]
  }

  def friend_status_json_schema, do: @friend_status_json_schema

  @json_schema %Schema{
    title: "Friendship",
    type: :object,
    required: [
      :status,
      :accepted_at
    ],
    properties: %{
      status: @friend_status_json_schema,
      accepted_at: %Schema{
        format: "date-time",
        type: :string,
        nullable: true
      }
    }
  }

  def json_schema, do: @json_schema

  @doc false
  def changeset(friendship, attrs) do
    friendship
    |> cast(attrs, [:user_id, :friend_id, :accepted_at])
    |> validate_required([:user_id, :friend_id])
    |> unique_constraint([:user_id, :friend_id],
      error_key: :friend_id
    )
  end

  defp associations(%User{id: self_id}, %User{id: friend_id}) do
    {
      Repo.get_by(Friendship, user_id: self_id, friend_id: friend_id),
      Repo.get_by(Friendship, user_id: friend_id, friend_id: self_id)
    }
  end

  # You can never be friends with yourself.
  def friend_status(%User{id: self_id}, %User{id: self_id}), do: :none

  def friend_status(%User{} = self, %User{} = friend) do
    {a, b} = associations(self, friend)
    friend_status(a, b)
  end

  def friend_status(nil, nil), do: :none
  def friend_status(nil, _), do: :received
  def friend_status(_, nil), do: :sent
  def friend_status(_, _), do: :accepted

  defp put_status(a, b),
    do: Map.put(a || %Friendship{}, :status, friend_status(a, b))

  def get_friendship(%User{} = self, %User{} = friend) do
    {a, b} = associations(self, friend)
    put_status(a, b)
  end

  def add_friend(self, friend) do
    Repo.transaction(fn ->
      {a, b} = associations(self, friend)

      case friend_status(a, b) do
        :none ->
          %Friendship{}
          |> changeset(%{user_id: self.id, friend_id: friend.id})
          |> Repo.insert!()
          |> put_status(b)

        :received ->
          accepted_at = DateTime.truncate(DateTime.utc_now(), :second)

          {:ok, b} =
            b
            |> changeset(%{accepted_at: accepted_at})
            |> Repo.update()

          %Friendship{}
          |> changeset(%{user_id: self.id, friend_id: friend.id, accepted_at: accepted_at})
          |> Repo.insert!()
          |> put_status(b)

        :sent ->
          Repo.rollback(:already_sent)

        :accepted ->
          Repo.rollback(:already_friends)
      end
    end)
  end

  def remove_friend(self, friend) do
    Repo.transaction(fn ->
      {a, b} = associations(self, friend)

      case friend_status(a, b) do
        status when status in [:accepted, :sent, :received] ->
          a && Repo.delete!(a)
          b && Repo.delete!(b)

          put_status(nil, nil)

        _ ->
          Repo.rollback(:not_friends)
      end
    end)
  end
end
