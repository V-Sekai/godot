defmodule Uro.SharedContent do
  @moduledoc """
  The Content context.
  """

  import Ecto.Query, warn: false
  alias Uro.Repo

  alias Uro.SharedContent.SharedFile

  @doc """
  Returns the list of all shared files.

  ## Examples

      iex> list_shared_files()
      [%SharedFile{}, ...]

  """
  def list_shared_files do
    SharedFile
    |> Repo.all()
    |> Repo.preload([:uploader])
  end

  @doc """
  Returns the list of shared files with pagination
  """
  def list_shared_files_paginated(params) do
    SharedFile
    |> Repo.paginate(params)
  end

  @doc """
  Returns the list of public shared files
  """
  def list_public_shared_files() do
    SharedFile
    |> where(is_public: true)
    |> Repo.all()
    |> Repo.preload([:uploader])
  end

  @doc """
  Returns the list of public shared files with pagination
  """
  def list_public_shared_files_paginated(params) do
    SharedFile
    |> where(is_public: true)
    |> Repo.paginate(params)
  end

  @doc """
  Returns the list of public shared files filtered by tag
  """
  def list_public_shared_files_by_tag(tag) do
    SharedFile
    |> where(is_public: true)
    |> Repo.all()
    |> Repo.preload([:uploader])
    |> Enum.filter(fn shared_file ->
      tag in shared_file.tags
    end)
  end

  @doc """
  Returns the list of public shared files matching all tags in a list
  """
  def list_public_shared_files_by_taglist(tag_list) when is_list(tag_list) do
    SharedFile
    |> where(is_public: true)
    |> Repo.all()
    |> Repo.preload([:uploader])
    |> Enum.filter(fn shared_file ->
      Enum.all?(tag_list, fn tag -> tag in shared_file.tags end)
    end)
  end

  @doc """
  Gets a single shared file.

  Raises `Ecto.NoResultsError` if the SharedFile does not exist.

  ## Examples

      iex> get_shared_file!(123)
      %SharedFile{}

      iex> get_shared_file!(456)
      ** (Ecto.NoResultsError)

  """
  def get_shared_file!(id) do
    SharedFile
    |> Repo.get!(id)
    |> Repo.preload([:uploader])
  end

  @doc """
  Gets a single public shared file.

  Raises `Ecto.NoResultsError` if the SharedFile does not exist or is inaccessible.

  """
  def get_public_shared_file!(id) do
    SharedFile
    |> where(is_public: true)
    |> Repo.get!(id)
    |> Repo.preload([:uploader])
  end

  @doc """
  Creates a shared file.

  ## Examples

      iex> create_shared_file(%{field: value})
      {:ok, %SharedFile{}}

      iex> create_shared_file(%{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def create_shared_file(attrs \\ %{}) do
    shared_content = %SharedFile{}

    Ecto.Multi.new()
    |> Ecto.Multi.insert(:shared_content, SharedFile.changeset(shared_content, attrs))
    |> Ecto.Multi.update(
      :shared_content_with_upload,
      &SharedFile.upload_changeset(&1.shared_content, attrs)
    )
    |> Repo.transaction()
    |> case do
      {:ok, %{shared_content_with_upload: shared_content_with_upload}} ->
        {:ok, shared_content_with_upload}

      {:error, _, reason, _} ->
        {:error, reason}
    end
  end

  @doc """
  Updates a shared file.

  ## Examples

      iex> update_shared_file(shared_file, %{field: new_value})
      {:ok, %SharedFile{}}

      iex> update_shared_file(shared_file, %{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def update_shared_file(%SharedFile{} = shared_file, attrs) do
    shared_file
    |> SharedFile.changeset(attrs)
    |> SharedFile.upload_changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Deletes a shared file.

  ## Examples

      iex> delete_shared_file(shared_file)
      {:ok, %SharedFile{}}

      iex> delete_shared_file(shared_file)
      {:error, %Ecto.Changeset{}}

  """
  def delete_shared_file(%SharedFile{} = shared_file) do
    Repo.delete(shared_file)
  end

  @doc """
  Returns an `%Ecto.Changeset{}` for tracking file changes.

  ## Examples

      iex> change_shared_file(shared_file)
      %Ecto.Changeset{source: %SharedFile{}}

  """
  def change_shared_file(%SharedFile{} = shared_file) do
    SharedFile.changeset(shared_file, %{})
  end
end
