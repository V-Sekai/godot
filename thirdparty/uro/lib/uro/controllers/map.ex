defmodule Uro.MapController do
  use Uro, :controller

  alias OpenApiSpex.Schema
  alias Uro.UserContent

  action_fallback Uro.FallbackController

  tags(["maps"])

  operation(:index,
    operation_id: "listMaps",
    summary: "List Maps",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def index(conn, _params) do
    maps = UserContent.list_public_maps()

    conn
    |> put_status(200)
    |> json(%{
      data: %{
        maps:
          Uro.Helpers.UserContentHelper.get_api_user_content_list(maps, %{
            merge_uploader_id: true
          })
      }
    })
  end

  operation(:indexUploads,
    operation_id: "listMapsUploads",
    summary: "List Maps uploaded by logged in user",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def indexUploads(conn, _params) do
    user = Uro.Helpers.Auth.get_current_user(conn)
    maps = UserContent.list_maps_uploaded_by(user)

    conn
    |> put_status(200)
    |> json(%{
      data: %{
        maps:
          Uro.Helpers.UserContentHelper.get_api_user_content_list(maps, %{
            merge_uploader_id: true
          })
      }
    })
  end

  operation(:show,
    operation_id: "getMap",
    summary: "Get Map",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def show(conn, %{"id" => id}) do
    id
    |> UserContent.get_map!()
    |> case do
      %Uro.UserContent.Map{} = map ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            map:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                map,
                %{merge_uploader_id: true, merge_is_public: true}
              )
          }
        })

      _ ->
        put_status(
          conn,
          400
        )
    end
  end

  operation(:showUpload,
    operation_id: "getMapUpload",
    summary: "Get uploaded Map",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def showUpload(conn, %{"id" => id}) do
    user = Uro.Helpers.Auth.get_current_user(conn)

    case UserContent.get_map_uploaded_by_user!(id, user) do
      %Uro.UserContent.Map{} = map ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            map:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                map,
                %{merge_uploader_id: true, merge_is_public: true}
              )
          }
        })

      _ ->
        put_status(
          conn,
          400
        )
    end
  end

  operation(:create,
    operation_id: "createMap",
    summary: "Create Map",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def create(conn, %{"map" => map_params}) do
    case UserContent.create_map(
           Uro.Helpers.UserContentHelper.get_correct_user_content_params(
             conn,
             map_params,
             "user_content_data",
             "user_content_preview"
           )
         ) do
      {:ok, map} ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            id: to_string(map.id),
            map:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                map,
                %{merge_uploader_id: true}
              )
          }
        })

      {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
        {:error, changeset}
    end
  end

  operation(:update,
    operation_id: "updateMap",
    summary: "Update Map",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def update(conn, %{"id" => id, "map" => map_params}) do
    user = Uro.Helpers.Auth.get_current_user(conn)
    map = UserContent.get_map_uploaded_by_user!(id, user)

    case UserContent.update_map(map, map_params) do
      {:ok, map} ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            id: to_string(map.id),
            map:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                map,
                %{merge_uploader_id: true}
              )
          }
        })

      {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
        {:error, changeset}
    end
  end

  operation(:delete,
    operation_id: "deleteMap",
    summary: "Delete Map",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def delete(conn, %{"id" => id}) do
    user = Uro.Helpers.Auth.get_current_user(conn)

    case UserContent.get_map_uploaded_by_user!(id, user) do
      %Uro.UserContent.Map{} = map ->
        case UserContent.delete_map(map) do
          {:ok, _map} ->
            conn
            |> put_status(200)
            |> json(%{data: %{}})

          {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
            {:error, changeset}
        end

      _ ->
        conn
        |> put_status(200)
        |> json(%{data: %{}})
    end
  end
end
