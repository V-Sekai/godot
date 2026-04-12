defmodule Uro.DefaultToken do
  @moduledoc """
  Token generation and verification.
  """

  use Joken.Config

  @iss "vsekai"

  def audience(), do: @iss

  def audience(pathname) do
    "#{@iss}.#{pathname}"
  end

  def audience_claim(config, pathname) do
    value = audience(pathname)
    add_claim(config, "aud", fn -> value end, &(&1 == value))
  end

  def token_config() do
    Joken.Config.default_claims()
    |> add_claim("iss", fn -> @iss end, &(&1 == @iss))
    |> add_claim("aud", fn -> audience() end, &(&1 == audience()))
  end
end
