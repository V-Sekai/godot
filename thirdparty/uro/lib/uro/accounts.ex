defmodule Uro.Accounts do
  @moduledoc """
  The Accounts context.
  """

  use Pow.Ecto.Context,
    repo: Uro.Repo,
    user: Uro.Accounts.User

  import Ecto.Query, warn: false
  import Uro.Helpers.UUID

  alias Uro.Accounts.User
  alias Uro.Accounts.UserPrivilegeRuleset
  alias Uro.EmailConfirmationToken
  alias Uro.Repo

  @user_associated_schemas [:user_privilege_ruleset]

  def get_by(params) do
    params
    |> pow_get_by()
    |> Repo.preload(@user_associated_schemas)
  end

  def create(params) do
    Repo.transaction(fn ->
      with {:ok, user} <- pow_create(params),
           {:ok, user} <- create_user_associations(user) do
        user
      else
        {:error, reason} ->
          Repo.rollback(reason)

        any ->
          Repo.rollback(any)
      end
    end)
  end

  def create_user_associations(user) do
    with {:ok, user_privilege_ruleset} <-
           Ecto.build_assoc(user, :user_privilege_ruleset) |> Repo.insert() do
      {:ok, Map.put(user, :user_privilege_ruleset, user_privilege_ruleset)}
    end
  end

  def get_by_username(username) when is_nil(username) do
    nil
  end

  def get_by_username(username) do
    User
    |> Repo.get_by(username: username)
    |> Repo.preload(@user_associated_schemas)
  end

  def get_by_email(email) when is_nil(email) do
    nil
  end

  def get_by_email(email) do
    User
    |> Repo.get_by(email: email)
    |> Repo.preload(@user_associated_schemas)
  end

  def get_by_username_or_email(username_or_email) when is_nil(username_or_email) do
    nil
  end

  def get_by_username_or_email(username_or_email) do
    User
    |> where(username: ^username_or_email)
    |> or_where(email: ^username_or_email)
    |> Repo.one()
    |> Repo.preload(@user_associated_schemas)
  end

  def list_users_admin do
    User
    |> Repo.all()
    |> Repo.preload(@user_associated_schemas)
  end

  def list_users_admin(params) do
    search_term = get_in(params, ["query"])

    User
    |> User.admin_search(search_term)
    |> Repo.all()
    |> Repo.preload(@user_associated_schemas)
  end

  def get_user!(id) when is_uuid(id) do
    User
    |> Repo.get!(id)
    |> Repo.preload(@user_associated_schemas)
  end

  def get_user!(_), do: nil

  # def create_user_privilege_ruleset_for_user(user, attrs \\ %{}) do
  #   user
  #   |> Ecto.build_assoc(:user_privilege_ruleset, attrs)
  #   |> Repo.insert()
  # end
  # def create_associated_entries_for_user(user) do
  #   create_user_privilege_ruleset_for_user(user)
  #   user
  # end

  # def create_user(conn, attrs) do
  #   conn
  #   |> Pow.Plug.create_user(attrs)
  #   |> case do
  #     {:ok, user, _} ->
  #       create_associated_entries_for_user(user)
  #       {:ok, user, conn}
  #     {:error, changeset, _} ->
  #       {:error, changeset}
  #   end
  # end

  def send_confirmation_email(%{email_confirmed_at: nil} = user) do
    {:ok, confirmation_token, _} = EmailConfirmationToken.new(user)
    confirmed_user = %{user | email_confirmed_at: DateTime.utc_now()}

    {:ok, _} =
      Uro.Mailer.confirmation_email(confirmation_token)
      |> Uro.Mailer.deliver_to(confirmed_user)

    :ok
  end

  def send_confirmation_email(_), do: :ok

  def confirm_email(user, token) do
    with {:ok, user} <- EmailConfirmationToken.confirm(user, token),
         %User{} = user <-
           user
           |> User.confirm_email_changeset()
           |> Repo.update() do
      {:ok, user}
    end
  end

  def update_email(%User{} = user, email, send_confirmation: send_confirmation) do
    user
    |> User.update_email_changeset(email, send_confirmation: send_confirmation)
    |> Repo.update()
    |> case do
      {:ok, user} ->
        if send_confirmation do
          send_confirmation_email(user)
        end

        {:ok, user}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  def update_user(%User{} = user, attrs) do
    user
    |> User.user_custom_changeset(attrs)
    |> Repo.update()
  end

  def update_user_as_admin(%User{} = user, attrs) do
    Ecto.Multi.new()
    |> Ecto.Multi.update(:user, User.admin_changeset(user, attrs))
    |> Ecto.Multi.update(
      :user_privilege_ruleset,
      &UserPrivilegeRuleset.admin_changeset(
        &1.user.user_privilege_ruleset,
        attrs["user_privilege_ruleset"]
      )
    )
    |> Repo.transaction()
    |> case do
      {:ok, %{user: user}} ->
        {:ok, user}

      {:error, _, reason, _} ->
        {:error, reason}
    end
  end

  def delete_user(%User{} = user) do
    Repo.delete(user)
  end

  def update_current_user(conn, attrs) do
    Pow.Plug.update_user(conn, attrs)
  end

  def delete_current_user(conn) do
    Pow.Plug.delete_user(conn)
  end

  def change_user(%User{} = user) do
    User.changeset(user, %{})
  end

  @spec lock(map()) :: {:ok, map()} | {:error, map()}
  def lock(user) do
    user
    |> User.lock_changeset()
    |> Repo.update()
  end

  @spec unlock(map()) :: {:ok, map()} | {:error, map()}
  def unlock(user) do
    user
    |> User.unlock_changeset()
    |> Repo.update()
  end
end
