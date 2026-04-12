defmodule Uro.FriendController do
  use Uro, :controller

  alias Uro.Accounts.User
  alias Uro.UserRelations.Friendship

  action_fallback(Uro.FallbackController)

  tags(["users"])

  operation(:create,
    operation_id: "friend",
    summary: "Friend",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    responses: [
      ok: {
        "",
        "application/json",
        Friendship.json_schema()
      }
    ]
  )

  def create(conn, %{"user_id" => friend_id}) do
    with {:ok, self} <- current_user(conn),
         {:ok, friend} <- user_from_key(conn, friend_id),
         :ok <-
           if(self.id === friend.id,
             do: {:error, :bad_request},
             else: :ok
           ),
         {:ok, friendship} <- Friendship.add_friend(self, friend) do
      conn
      |> put_status(:ok)
      |> json(friendship)
    else
      {:error, :already_friends} ->
        json_error(conn, code: :conflict, message: "Already friends")

      {:error, :already_sent} ->
        json_error(conn, code: :conflict, message: "Friend request already sent")

      any ->
        any
    end
  end

  operation(:delete,
    operation_id: "unfriend",
    summary: "Unfriend",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    responses: [
      ok: {
        "",
        "application/json",
        Friendship.json_schema()
      }
    ]
  )

  def delete(conn, %{"user_id" => friend_id}) do
    with {:ok, self} <- current_user(conn),
         {:ok, friend} <- user_from_key(conn, friend_id),
         :ok <-
           if(self.id === friend.id,
             do: {:error, :bad_request},
             else: :ok
           ),
         {:ok, friendship} <- Friendship.remove_friend(self, friend) do
      conn
      |> put_status(:ok)
      |> json(friendship)
    else
      {:error, :not_friends} ->
        json_error(conn, code: :bad_request)

      any ->
        any
    end
  end

  operation(:show,
    operation_id: "friendStatus",
    summary: "Get Friend Status",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    responses: [
      ok: {
        "",
        "application/json",
        Friendship.json_schema()
      }
    ]
  )

  def show(conn, %{"user_id" => user_id}) do
    with {:ok, self} <- current_user(conn),
         {:ok, friend} <- user_from_key(conn, user_id),
         friendship <-
           Friendship.get_friendship(self, friend) do
      conn
      |> put_status(:ok)
      |> json(friendship)
    end
  end
end
