defmodule Uro.PropController do
  use Uro, :controller

  alias OpenApiSpex.Schema
  alias Uro.UserContent

  action_fallback Uro.FallbackController

  tags(["props"])

  operation(:index,
    operation_id: "listProps",
    summary: "List Props",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def index(conn, _params) do
    props = UserContent.list_public_props()

    conn
    |> put_status(200)
    |> json(%{
      data: %{
        props:
          Uro.Helpers.UserContentHelper.get_api_user_content_list(props, %{
            merge_uploader_id: true
          })
      }
    })
  end

  operation(:indexUploads,
    operation_id: "listPropsUploads",
    summary: "List Props uploaded by logged in user",
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
    props = UserContent.list_props_uploaded_by(user)

    conn
    |> put_status(200)
    |> json(%{
      data: %{
        props:
          Uro.Helpers.UserContentHelper.get_api_user_content_list(props, %{
            merge_uploader_id: true
          })
      }
    })
  end

  operation(:show,
    operation_id: "getProp",
    summary: "Get Prop",
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
    |> UserContent.get_prop!()
    |> case do
      %Uro.UserContent.Prop{} = prop ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            prop:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                prop,
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
    operation_id: "getPropUpload",
    summary: "Get uploaded Prop",
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

    case UserContent.get_prop_uploaded_by_user!(id, user) do
      %Uro.UserContent.Prop{} = prop ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            prop:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                prop,
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
    operation_id: "createProp",
    summary: "Create Prop",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def create(conn, %{"prop" => prop_params}) do
    case UserContent.create_prop(
           Uro.Helpers.UserContentHelper.get_correct_user_content_params(
             conn,
             prop_params,
             "user_content_data",
             "user_content_preview"
           )
         ) do
      {:ok, prop} ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            id: to_string(prop.id),
            prop:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                prop,
                %{merge_uploader_id: true}
              )
          }
        })

      {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
        {:error, changeset}
    end
  end

  operation(:update,
    operation_id: "updateProp",
    summary: "Update Prop",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def update(conn, %{"id" => id, "prop" => prop_params}) do
    user = Uro.Helpers.Auth.get_current_user(conn)
    prop = UserContent.get_prop_uploaded_by_user!(id, user)

    case UserContent.update_prop(prop, prop_params) do
      {:ok, prop} ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            id: to_string(prop.id),
            prop:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                prop,
                %{merge_uploader_id: true}
              )
          }
        })

      {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
        {:error, changeset}
    end
  end

  operation(:delete,
    operation_id: "deleteProp",
    summary: "Delete Prop",
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

    case UserContent.get_prop_uploaded_by_user!(id, user) do
      %Uro.UserContent.Prop{} = prop ->
        case UserContent.delete_prop(prop) do
          {:ok, _prop} ->
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
