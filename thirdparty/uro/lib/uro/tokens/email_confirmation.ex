defmodule Uro.EmailConfirmationToken do
  @moduledoc """
  Email confirmation token generation and verification.
  """

  use Joken.Config

  alias Uro.Accounts.User
  alias Uro.DefaultToken

  def token_config() do
    DefaultToken.audience_claim(DefaultToken.token_config(), "email_confirmation")
  end

  def new(%User{} = user) do
    generate_and_sign(%{
      "sub" => user.id,
      "email" => user.email
    })
  end

  def confirm(
        %User{
          id: user_id,
          email: email
        } = user,
        token
      )
      when is_binary(token) do
    case verify_and_validate(token) do
      {:ok,
       %{
         "sub" => ^user_id,
         "email" => ^email
       }} ->
        {:ok, user}

      _ ->
        {:error, :invalid_token}
    end
  end
end
