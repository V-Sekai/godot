defmodule Uro.ShardController do
  use Uro, :controller

  alias OpenApiSpex.Schema
  alias Uro.Repo
  alias Uro.VSekai
  alias Uro.VSekai.Shard

  tags(["shards"])

  def ensure_has_address(conn, params) do
    if !Map.has_key?(params, "address") do
      Map.put(params, "address", to_string(:inet_parse.ntoa(conn.remote_ip)))
    else
      params
    end
  end

  def ensure_user_is_current_user_or_nil(conn, params) do
    if Uro.Helpers.Auth.signed_in?(conn) do
      Map.put(params, "user_id", Uro.Helpers.Auth.get_current_user(conn).id)
    else
      Map.put(params, "user_id", nil)
    end
  end

  def can_connection_modify_shard(conn, shard) do
    if shard.user != nil and
         Uro.Helpers.Auth.signed_in?(conn) and
         shard.user == Uro.Helpers.Auth.get_current_user(conn) do
      true
    else
      if shard.user == nil and
           shard.address == to_string(:inet_parse.ntoa(conn.remote_ip)) do
        true
      else
        false
      end
    end
  end

  operation(:index,
    operation_id: "listShards",
    summary: "List Shards",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{
          type: :array,
          items: Shard.json_schema()
        }
      }
    ]
  )

  def index(conn, _params) do
    shards = VSekai.list_fresh_shards()
    shards_json = Enum.map(shards, fn x -> Shard.to_json_schema(x) end)

    conn
    |> put_status(200)
    |> json(%{data: %{shards: shards_json}})
  end

  operation(:create,
    operation_id: "createShard",
    summary: "Create Shard",
    request_body: {
      "",
      "application/json",
      Shard.json_schema()
    },
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{
          type: :object,
          required: [:id],
          properties: %{
            id: %Schema{
              type: :string
            }
          }
        }
      }
    ]
  )

  def create(conn, shard_params) do
    shard_params =
      ensure_has_address(
        conn,
        shard_params
      )

    conn
    |> ensure_user_is_current_user_or_nil(shard_params)
    |> VSekai.create_shard()
    |> case do
      {:ok, shard} ->
        conn
        |> put_status(200)
        |> json(%{data: %{id: to_string(shard.id)}})

      {:error, %Ecto.Changeset{}} ->
        json_error(conn)
    end
  end

  operation(:update,
    operation_id: "updateShard",
    summary: "Update Shard",
    parameters: [
      id: [
        in: :path,
        schema: %Schema{
          type: :string
        }
      ]
    ],
    responses: [
      ok: {
        "",
        "application/json",
        Shard.json_schema()
      }
    ]
  )

  def update(conn, %{"id" => id, "shard" => shard_params}) do
    shard = VSekai.get_shard!(id)

    if can_connection_modify_shard(conn, shard) do
      case VSekai.update_shard(shard, shard_params) do
        {:ok, shard} ->
          conn
          |> put_status(200)
          |> json(%{data: %{id: to_string(shard.id)}})

        {:error, %Ecto.Changeset{}} ->
          json_error(conn)
      end
    else
      json_error(conn)
    end
  end

  def update(conn, %{"id" => id}) do
    update(conn, %{"id" => id, "shard" => %{}})
  end

  operation(:delete,
    operation_id: "deleteShard",
    summary: "Delete Shard",
    parameters: [
      id: [
        in: :path,
        schema: %Schema{
          type: :string
        }
      ]
    ],
    responses: [
      ok: {
        "",
        "application/json",
        success_json_schema()
      }
    ]
  )

  def delete(conn, %{"id" => id}) do
    shard =
      Uro.VSekai.Shard
      |> Repo.get!(id)
      |> Repo.preload(:user)
      |> Repo.preload(user: [:user_privilege_ruleset])

    if can_connection_modify_shard(conn, shard) do
      case VSekai.delete_shard(shard) do
        {:ok, shard} ->
          conn
          |> put_status(200)
          |> json(%{data: %{id: to_string(shard.id)}})

        {:error, %Ecto.Changeset{}} ->
          json_error(conn)
      end
    else
      json_error(conn)
    end
  end
end
