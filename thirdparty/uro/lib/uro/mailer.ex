defmodule Uro.Mailer do
  use Swoosh.Mailer, otp_app: :uro

  import Swoosh.Email
  require Logger

  alias Uro.Accounts.User

  defp get_adapter() do
    if Application.get_env(:uro, Uro.Mailer)[:adapter] == Swoosh.Adapters.Sendgrid do
      case System.get_env("SENDGRID_API_KEY") do
        nil ->
          Logger.warning("SENDGRID_API_KEY not found. Falling back to mail Logger")
          Swoosh.Adapters.Logger

        "" ->
          Logger.warning("SENDGRID_API_KEY is empty. Falling back to mail Logger")
          Swoosh.Adapters.Logger

        _ ->
          Application.get_env(:uro, Uro.Mailer)[:adapter]
      end
    else
      Application.get_env(:uro, Uro.Mailer)[:adapter]
    end
  end

  def create_email(subject: subject, text: text, html: html) do
    %Swoosh.Email{}
    |> from({"V-Sekai", System.get_env("NOREPLY_SENDER") || "noreply@vsekai.com"})
    |> subject(subject)
    |> html_body(html)
    |> text_body(text)
  end

  def deliver_to(_, %User{email_confirmed_at: nil}) do
    {:error, :email_not_confirmed}
  end

  def deliver_to(%Swoosh.Email{} = email, %User{display_name: display_name, email: email_address})
      when is_binary(email_address) do
    adapter = get_adapter()

    if adapter == Swoosh.Adapters.Logger do
      Logger.warning("Emails are not sent when using Swoosh.Adapters.Logger.")
      {:ok, %{}}
    else
      email
      |> to({display_name, email_address})
      |> deliver(adapter: adapter)
    end
  end

  def confirmation_email(confirmation_token) when is_binary(confirmation_token) do
    confirmation_url =
      "#{Application.get_env(:uro, :frontend_url)}confirm-email/#{confirmation_token}"

    create_email(
      subject: "Confirm your email address",
      text: """
      Please confirm your email by clicking on the link below:
      #{confirmation_url}
      """,
      html: """
      <p>Please confirm your email by clicking on the link below:</p>
      <a href="#{confirmation_url}">Confirm email</a>
      """
    )
  end
end
