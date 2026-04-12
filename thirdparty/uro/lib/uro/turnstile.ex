defmodule Uro.Turnstile do
  @moduledoc """
  https://developers.cloudflare.com/turnstile/get-started/server-side-validation/
  """

  alias Plug.Conn

  @verification_url "https://challenges.cloudflare.com/turnstile/v0/siteverify"

  def config(), do: Application.get_env(:uro, Uro.Turnstile)
  def config(key), do: Keyword.get(config(), key)

  # https://developers.cloudflare.com/turnstile/get-started/server-side-validation/#error-codes
  @verify_captcha_errors [
    {:missing_input_secret, "The secret parameter was not passed."},
    {:invalid_input_secret, "The secret parameter was invalid or did not exist."},
    {:missing_input_response, "The response parameter (token) was not passed."},
    {:invalid_input_response,
     "The response parameter (token) is invalid or has expired. Most of the time, this means a fake token has been used. If the error persists, contact customer support."},
    {:invalid_widget_id,
     "The widget ID extracted from the parsed site secret key was invalid or did not exist."},
    {:invalid_parsed_secret, "The secret extracted from the parsed site secret key was invalid."},
    {:bad_request, "The request was rejected because it was malformed."},
    {:timeout_or_duplicate,
     "The response parameter (token) has already been validated before. This means that the token was issued five minutes ago and is no longer valid, or it was already redeemed."},
    {:internal_error,
     "An internal error happened while validating the response. The request can be retried."}
  ]

  for {error, message} <- @verify_captcha_errors do
    def error_message(unquote(error)), do: unquote(message)
  end

  def verify_captcha(_, options \\ [])

  def verify_captcha(
        %Conn{
          remote_ip: remote_ip,
          params: params,
          assigns: %{request_id: request_id}
        } = conn,
        options
      ) do
    verify_captcha(params[Keyword.get(options, :field_key, "captcha")],
      remoteip: Tuple.to_list(remote_ip) |> Enum.join("."),
      idempotency_key: request_id
    )
    |> case do
      {:ok, _} ->
        {:ok, conn}

      {:error, _} ->
        {:error, :bad_request, "Captcha failed, try again"}
    end
  end

  def verify_captcha(token, options) when is_binary(token) do
    with {:ok, %HTTPoison.Response{body: body}} <-
           HTTPoison.post(
             @verification_url,
             Jason.encode!(
               Enum.into(options, %{
                 secret: config(:secret_key),
                 response: token
               })
             ),
             [{"content-type", "application/json"}]
           ),
         {:ok, %{"success" => true} = body} <- Jason.decode(body) do
      {:ok, body}
    else
      {:ok, %{"error-codes" => [head]}} ->
        {:error, head |> String.replace("-", "_") |> String.to_atom()}

      _ ->
        {:error, :internal_error}
    end
  end

  def verify_captcha(_, _), do: {:error, :missing_input_response}
end
