defmodule Uro.AvatarController do
  use Uro, :controller

  alias OpenApiSpex.Schema
  alias Uro.UserContent

  action_fallback Uro.FallbackController

  tags(["avatars"])

  operation(:index,
    operation_id: "listAvatars",
    summary: "List Avatars",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def index(conn, _params) do
    avatars = UserContent.list_public_avatars()

    conn
    |> put_status(200)
    |> json(%{
      data: %{
        avatars:
          Uro.Helpers.UserContentHelper.get_api_user_content_list(avatars, %{
            merge_uploader_id: true
          })
      }
    })
  end

  operation(:indexUploads,
    operation_id: "listAvatarsUploads",
    summary: "List Avatars uploaded by logged in user",
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
    avatars = UserContent.list_avatars_uploaded_by(user)

    conn
    |> put_status(200)
    |> json(%{
      data: %{
        avatars:
          Uro.Helpers.UserContentHelper.get_api_user_content_list(avatars, %{
            merge_uploader_id: true
          })
      }
    })
  end

  operation(:show,
    operation_id: "getAvatar",
    summary: "Get Avatar",
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
    |> UserContent.get_avatar!()
    |> case do
      %Uro.UserContent.Avatar{} = avatar ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            avatar:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                avatar,
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
    operation_id: "getAvatarUpload",
    summary: "Get uploaded Avatar",
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

    case UserContent.get_avatar_uploaded_by_user!(id, user) do
      %Uro.UserContent.Avatar{} = avatar ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            avatar:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                avatar,
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
    operation_id: "createAvatar",
    summary: "Create Avatar",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def create(conn, %{"avatar" => avatar_params}) do
    case UserContent.create_avatar(
           Uro.Helpers.UserContentHelper.get_correct_user_content_params(
             conn,
             avatar_params,
             "user_content_data",
             "user_content_preview"
           )
         ) do
      {:ok, avatar} ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            id: to_string(avatar.id),
            avatar:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                avatar,
                %{merge_uploader_id: true}
              )
          }
        })

      {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
        {:error, changeset}
    end
  end

  operation(:update,
    operation_id: "updateAvatar",
    summary: "Update Avatar",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{}
      }
    ]
  )

  def update(conn, %{"id" => id, "avatar" => avatar_params}) do
    user = Uro.Helpers.Auth.get_current_user(conn)
    avatar = UserContent.get_avatar_uploaded_by_user!(id, user)

    case UserContent.update_avatar(avatar, avatar_params) do
      {:ok, avatar} ->
        conn
        |> put_status(200)
        |> json(%{
          data: %{
            id: to_string(avatar.id),
            avatar:
              Uro.Helpers.UserContentHelper.get_api_user_content(
                avatar,
                %{merge_uploader_id: true}
              )
          }
        })

      {:error, %Ecto.Changeset{changes: _changes, errors: _errors} = changeset} ->
        {:error, changeset}
    end
  end

  operation(:delete,
    operation_id: "deleteAvatar",
    summary: "Delete Avatar",
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

    case UserContent.get_avatar_uploaded_by_user!(id, user) do
      %Uro.UserContent.Avatar{} = avatar ->
        case UserContent.delete_avatar(avatar) do
          {:ok, _avatar} ->
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
