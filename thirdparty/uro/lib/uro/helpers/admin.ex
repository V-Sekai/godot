defmodule Uro.Helpers.Admin do
  @moduledoc """
  Admin helper functions, automatically imported by controllers.
  """

  alias Uro.Accounts.User
  alias Uro.Helpers.Auth

  def is_session_admin?(conn) do
    user = Auth.get_current_user(conn)
    User.admin?(user)
  end
end
