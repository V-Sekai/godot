defmodule Uro.UserRelations do
  @moduledoc """
  The UserRelations context.
  """

  import Ecto.Query, warn: false
  alias Uro.Repo

  alias Uro.UserRelations.Friendship
  alias Uro.UserRelations.IdentityProof

  @doc """
  Returns the list of friendships.

  ## Examples

      iex> list_friendships()
      [%Friendship{}, ...]

  """
  def list_friendships do
    Repo.all(Friendship)
  end

  @doc """
  Gets a single friendship.

  Raises `Ecto.NoResultsError` if the Friendship does not exist.

  ## Examples

      iex> get_friendship!(123)
      %Friendship{}

      iex> get_friendship!(456)
      ** (Ecto.NoResultsError)

  """
  def get_friendship!(id), do: Repo.get!(Friendship, id)

  @doc """
  Creates a friendship.

  ## Examples

      iex> create_friendship(%{field: value})
      {:ok, %Friendship{}}

      iex> create_friendship(%{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def create_friendship(attrs \\ %{}) do
    %Friendship{}
    |> Friendship.changeset(attrs)
    |> Repo.insert()
  end

  @doc """
  Updates a friendship.

  ## Examples

      iex> update_friendship(friendship, %{field: new_value})
      {:ok, %Friendship{}}

      iex> update_friendship(friendship, %{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def update_friendship(%Friendship{} = friendship, attrs) do
    friendship
    |> Friendship.changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Deletes a friendship.

  ## Examples

      iex> delete_friendship(friendship)
      {:ok, %Friendship{}}

      iex> delete_friendship(friendship)
      {:error, %Ecto.Changeset{}}

  """
  def delete_friendship(%Friendship{} = friendship) do
    Repo.delete(friendship)
  end

  @doc """
  Returns an `%Ecto.Changeset{}` for tracking friendship changes.

  ## Examples

      iex> change_friendship(friendship)
      %Ecto.Changeset{source: %Friendship{}}

  """
  def change_friendship(%Friendship{} = friendship) do
    Friendship.changeset(friendship, %{})
  end

  @doc """
  Gets a single identity proof.

  Raises `Ecto.NoResultsError` if the IdentityProof does not exist.

  ## Examples

      iex> get_identity_proof!(123)
      %IdentityProof{}

      iex> get_identity_proof!(456)
      ** (Ecto.NoResultsError)

  """
  def get_identity_proof_as!(id, requester) do
    IdentityProof
    |> Repo.get!(id)
    |> Repo.preload([:user_from, :user_to])
    |> case do
      nil ->
        nil

      identity_proof ->
        if identity_proof.user_from == requester or identity_proof.user_to == requester do
          identity_proof
        else
          nil
        end
    end
  end

  @doc """
  Creates an identity token.

  ## Examples

      iex> create_identity_proof(%{field: value})
      {:ok, %IdentityProof{}}

      iex> create_identity_proof(%{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def create_identity_proof(user_from, user_to) do
    user_from_id = user_from.id
    user_to_id = user_to.id

    %IdentityProof{}
    |> IdentityProof.changeset(%{user_from_id: user_from_id, user_to_id: user_to_id})
    |> Repo.insert()
  end
end
