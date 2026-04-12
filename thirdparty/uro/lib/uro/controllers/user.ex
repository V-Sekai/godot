defmodule Uro.UserController do
  @moduledoc false

  use Uro, :controller

  import Ecto.Changeset
  import Uro.Helpers.User
  import Uro.Helpers.Changeset

  alias OpenApiSpex.Schema
  alias Uro.Accounts
  alias Uro.Accounts.User
  alias Uro.Accounts.UserPrivilegeRuleset
  alias Uro.Repo
  alias Uro.Session
  alias Uro.Turnstile

  action_fallback(Uro.FallbackController)

  tags(["users"])
  security(default_security())

  def user_not_found(conn), do: json_error(conn, code: :not_found, message: "User not found")

  operation(:show,
    operation_id: "getUser",
    summary: "Get User",
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
        User.json_schema()
      },
      not_found: {
        "User not found",
        "application/json",
        error_json_schema()
      }
    ]
  )

  def show(conn, %{"user_id" => id}) do
    with {:ok, user} <- user_from_key(conn, id) do
      conn
      |> put_status(:ok)
      |> json(User.to_json_schema(user, conn))
    end
  end

  operation(:showCurrent,
    operation_id: "getUserCurrent",
    summary: "Get Current User",
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
        User.json_schema()
      },
      not_found: {
        "User not found",
        "application/json",
        error_json_schema()
      }
    ]
  )

  def showCurrent(conn, _params) do
    with {:ok, user} <- user_from_key(conn, "me") do
      ruleset = UserPrivilegeRuleset.to_json_schema(user.user_privilege_ruleset)

      conn
      |> put_status(:ok)
      |> json(%{
        data: %{
          access_token: conn.assigns[:access_token],
          renewal_token: conn.assigns[:access_token],
          user: User.to_json_schema(user, conn),
          user_privilege_ruleset: ruleset
        }
      })
    end
  end

  operation(:index,
    operation_id: "listUsers",
    summary: "List Users",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{
          type: :array,
          items: User.json_schema()
        }
      }
    ]
  )

  def index(conn, _) do
    json(
      conn,
      User.to_json_schema(Accounts.list_users_admin(), conn)
    )
  end

  operation(:create,
    operation_id: "signup",
    summary: "Create an Account",
    request_body:
      {"", "application/json",
       %Schema{
         type: :object,
         required: [
           :display_name,
           :username,
           :email,
           :password,
           :captcha
         ],
         properties: %{
           display_name: User.sensitive_json_schema().properties.display_name,
           username: User.sensitive_json_schema().properties.username,
           email: User.sensitive_json_schema().properties.email,
           password: %Schema{type: :string},
           captcha: %Schema{
             type: :string
           }
         }
       }},
    responses: [
      ok: {
        "",
        "application/json",
        Session.json_schema()
      },
      unprocessable_entity: {
        "",
        "application/json",
        error_json_schema()
      }
    ]
  )

  def create(conn, params) do
    Repo.transaction(fn ->
      with {:ok, _} <- Turnstile.verify_captcha(conn),
           {:ok, user} <- Accounts.create(params),
           :ok <- Accounts.send_confirmation_email(user),
           conn <- Pow.Plug.create(conn, user) do
        {user, conn}
      else
        any ->
          Repo.rollback(any)
      end
    end)
    |> case do
      {:ok, {_, conn}} ->
        {:ok, session} = current_session(conn)

        json(
          conn,
          Session.to_json_schema(session)
        )

      {:error, conn} ->
        conn
    end
  end

  operation(:createClient,
    operation_id: "signupClient",
    summary: "Create an Account from Game client request",
    responses: [
      ok: {
        "",
        "application/json",
        Session.json_game_client_schema()
      },
      unprocessable_entity: {
        "",
        "application/json",
        error_json_schema()
      }
    ]
  )

  def createClient(conn, %{"user" => user_params, "apiKey" => api_key}) do
    create_params = %{
      "username" => Map.get(user_params, "username"),
      "display_name" => Map.get(user_params, "username"),
      "email" => Map.get(user_params, "email"),
      "password" => Map.get(user_params, "password")
    }

    Repo.transaction(fn ->
      with :ok <-
             (fn ->
                if api_key == System.get_env("SIGNUP_API_KEY") do
                  :ok
                end
              end).(),
           {:ok, user} <- Accounts.create(create_params),
           :ok <- Accounts.send_confirmation_email(user),
           conn <- Pow.Plug.create(conn, user) do
        {user, conn}
      else
        any ->
          Repo.rollback(any)
      end
    end)
    |> case do
      {:ok, {_, conn}} ->
        {:ok, session} = current_session(conn)

        json(
          conn,
          Session.to_json_game_client_schema(session)
        )

      {:error, conn} ->
        conn
    end
  end

  operation(:resend_confirmation_email,
    operation_id: "resendConfirmationEmail",
    summary: "Resend Confirmation Email",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    responses: [
      accepted: {
        "",
        "application/json",
        success_json_schema()
      }
    ]
  )

  def resend_confirmation_email(conn, %{"user_id" => user_id}) do
    with {:ok, current_user} <- current_user(conn),
         {:ok, user} <- user_from_key(conn, user_id),
         true <-
           User.admin?(current_user) or user.id == current_user.id ||
             {
               :error,
               :unauthorized,
               "You do not have permission to resend this user's confirmation email"
             },
         :ok <- Accounts.send_confirmation_email(user) do
      conn
      |> put_status(:accepted)
      |> json(%{
        message: "Confirmation email sent"
      })
    end
  end

  operation(:update_email,
    operation_id: "updateEmail",
    summary: "Update Email Address",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    request_body:
      {"", "application/json",
       %Schema{
         type: :object,
         required: [
           :email
         ],
         properties: %{
           email: %Schema{
             type: :string,
             format: "email",
             description: "The new email address."
           },
           current_password: %Schema{
             type: :string,
             description: "The user's current password."
           },
           send_confirmation: %Schema{
             type: :boolean,
             default: true,
             description: "Whether to send a confirmation email."
           }
         }
       }},
    responses: [
      ok: {
        "",
        "application/json",
        User.json_schema()
      }
    ]
  )

  def update_email(conn, %{"user_id" => user_id} = params) do
    with {:ok, current_user} <- current_user(conn),
         {:ok, user} <- user_from_key(conn, user_id),
         {:ok,
          %{
            email: email,
            send_confirmation: send_confirmation
          }} <-
           cast(
             {%{},
              %{
                user_id: :string,
                email: :string,
                current_password: :string,
                send_confirmation: :boolean
              }},
             Map.merge(params, %{
               "user_id" => user.id,
               "password_hash" => user.password_hash
             }),
             [
               :user_id,
               :email,
               :current_password,
               :send_confirmation
             ]
           )
           |> validate_required([:email])
           |> put_default(:send_confirmation, true)
           |> then(fn changeset ->
             if User.admin?(current_user) do
               changeset
             else
               changeset
               |> validate_as_yourself(current_user)
               |> validate_inclusion(:send_confirmation, [true], message: "must be true")
               |> User.validate_current_password(user)
             end
           end)
           |> apply_action(nil),
         {:ok, user} <- Accounts.update_email(user, email, send_confirmation: send_confirmation) do
      json(conn, User.to_json_schema(user, conn))
    end
  end

  operation(:confirm_email,
    operation_id: "confirmEmail",
    summary: "Confirm Email Address",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    request_body:
      {"", "application/json",
       %Schema{
         type: :object,
         required: [
           :token
         ],
         properties: %{
           token: %Schema{
             type: :string,
             description: "The confirmation token."
           }
         }
       }},
    responses: [
      ok: {
        "",
        "application/json",
        User.json_schema()
      }
    ]
  )

  def confirm_email(conn, %{"user_id" => user_id, "token" => token}) do
    with {:ok, user} <- user_from_key(conn, user_id),
         {:ok, user} <- Accounts.confirm_email(user, token) do
      json(conn, User.to_json_schema(user, conn))
    end
  end

  def confirm_email(_, _), do: {:error, :bad_request}

  operation(:update,
    operation_id: "updateUser",
    summary: "Update User",
    parameters: [
      user_id: [
        in: :path,
        schema: User.loose_key_json_schema()
      ]
    ],
    request_body: {"", "application/json", User.update_json_schema()},
    responses: [
      ok: {
        "",
        "application/json",
        User.json_schema()
      }
    ]
  )

  def update(
        conn,
        %{
          "user_id" => user_id
        } = params
      ) do
    with {:ok, self} <- current_user(conn),
         {:ok, self} <- user_confirmed_email(self),
         {:ok, user} <- user_from_key(conn, user_id),
         true <-
           User.admin?(self) or user.id == self.id ||
             {
               :error,
               :unauthorized,
               "You do not have permission to update this user"
             },
         {:ok, user} <- Accounts.update_user(user, params) do
      conn
      |> put_status(:ok)
      |> json(User.to_json_schema(user, conn))
    end
  end
end
