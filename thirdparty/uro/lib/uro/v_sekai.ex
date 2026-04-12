defmodule Uro.VSekai do
  @moduledoc """
  The VSekai context.
  """

  import Ecto.Query, warn: false
  alias Uro.Repo

  alias Uro.VSekai.Shard

  @doc """
  Returns the time in seconds a shard before is considered stale.
  """
  def shard_freshness_time_in_seconds, do: 30

  @doc """
  Returns the list of shards.

  ## Examples

      iex> list_shards()
      [%Shard{}, ...]

  """
  def list_shards do
    Shard
    |> Repo.all()
    |> Repo.preload(user: [:user])
  end

  @doc """
  Returns a list of shards last modified within the shard freshness time.

  ## Examples

      iex> list_fresh_shards()
      [%Shard{}, ...]

  """
  def list_fresh_shards do
    stale_timestamp =
      DateTime.add(DateTime.utc_now(), -shard_freshness_time_in_seconds(), :second)

    Repo.all(from m in Shard, where: m.updated_at > ^stale_timestamp, preload: [:user])
  end

  @doc """
  Gets a single shard.

  Raises `Ecto.NoResultsError` if the Shard does not exist.

  ## Examples

      iex> get_shard!(123)
      %Shard{}

      iex> get_shard!(456)
      ** (Ecto.NoResultsError)

  """
  def get_shard!(id) do
    Shard
    |> Repo.get!(id)
    |> Repo.preload(user: [:user])
  end

  @doc """
  Creates a shard.

  ## Examples

      iex> create_shard(%{field: value})
      {:ok, %Shard{}}

      iex> create_shard(%{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def create_shard(attrs \\ %{}) do
    shard_data = Map.get(attrs, "shard", %{})
    flattened_attrs = Map.merge(Map.drop(attrs, ["shard"]), shard_data)

    %Shard{}
    |> Shard.changeset(flattened_attrs)
    |> Repo.insert()
  end

  @doc """
  Updates a shard.

  ## Examples

      iex> update_shard(shard, %{field: new_value})
      {:ok, %Shard{}}

      iex> update_shard(shard, %{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def update_shard(%Shard{} = shard, attrs) do
    shard
    |> Shard.changeset(attrs)
    |> Repo.update(force: true)
  end

  @doc """
  Deletes a shard.

  ## Examples

      iex> delete_shard(shard)
      {:ok, %Shard{}}

      iex> delete_shard(shard)
      {:error, %Ecto.Changeset{}}

  """
  def delete_shard(%Shard{} = shard) do
    Repo.delete(shard)
  end

  @doc """
  Returns an `%Ecto.Changeset{}` for tracking shard changes.

  ## Examples

      iex> change_shard(shard)
      %Ecto.Changeset{source: %Shard{}}

  """
  def change_shard(%Shard{} = shard) do
    Shard.changeset(shard, %{})
  end

  def get_shard_by_address(address) when is_nil(address) do
    nil
  end

  def get_shard_by_address(address) do
    Repo.get_by(Shard, address: address)
  end
end
