# Script for populating the database. You can run it as:
#
#     mix run priv/repo/test_seeds.exs
#
# Inside the script, you can read and write to any of your
# repositories directly:
#
#     Uro.Repo.insert!(%Uro.SomeSchema{})
#
# We recommend using the bang functions (`insert!`, `update!`
# and so on) as they will fail if something goes wrong.

alias Uro.Repo
alias Uro.Accounts.User
alias Uro.Accounts.UserPrivilegeRuleset

current_time = DateTime.utc_now()

# Start a single transaction for all database seed operations
Repo.transaction(fn ->
  # Upsert normal user and their privileges
  normal_user =
    User
    |> Repo.get_by(email: "user@example.com")
    |> case do
      nil ->
        %User{}
        |> User.admin_changeset(%{
          email: "user@example.com",
          username: "regularuser",
          display_name: "Regular User",
          email_notifications: true,
          password: "userpassword",
          password_confirmation: "userpassword",
          email_confirmed_at: current_time
        })
        |> User.confirm_email_changeset()
        |> Repo.insert!()
      user ->
        user
        |> User.confirm_email_changeset()
        |> Repo.update!()
    end

  # Ensure normal user privileges exist
  normal_user_privileges_params = %{user_id: normal_user.id}
  UserPrivilegeRuleset
  |> Repo.get_by(user_id: normal_user.id)
  |> case do
    nil ->
      UserPrivilegeRuleset.admin_changeset(%UserPrivilegeRuleset{}, normal_user_privileges_params)
      |> Repo.insert!()
  end

  # Upsert admin user and their privileges
  admin_user =
    User
    |> Repo.get_by(email: "admin@example.com")
    |> case do
      nil ->
        %User{}
        |> User.admin_changeset(%{
          email: "admin@example.com",
          username: "adminuser",
          display_name: "Admin User",
          email_notifications: true,
          password: "adminpassword",
          password_confirmation: "adminpassword",
          email_confirmed_at: current_time
        })
        |> User.confirm_email_changeset()
        |> Repo.insert!()
      user ->
        user
        |> User.confirm_email_changeset()
        |> Repo.update!()
    end

  # Ensure admin user privileges exist with additional permissions
  admin_privileges_params = %{
    user_id: admin_user.id,
    is_admin: true,
    can_upload_avatars: true,
    can_upload_maps: true,
    can_upload_props: true,
    can_upload_shared_files: true
  }
  UserPrivilegeRuleset
  |> Repo.get_by(user_id: admin_user.id)
  |> case do
    nil ->
      %UserPrivilegeRuleset{}
      |> UserPrivilegeRuleset.admin_changeset(admin_privileges_params)
      |> Repo.insert!()
  end
end)
