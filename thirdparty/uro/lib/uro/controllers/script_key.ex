defmodule Uro.ScriptKeyController do
  @moduledoc """
  POST /auth/script_key — returns the AES-128-GCM `{key, iv, ttl}` for
  an asset uuid so the client can decrypt its encrypted chunks. See
  CONCEPT_MMOG.md for the wire format and the post-MVP JWT gating plan.
  """

  use Uro, :controller

  def show(conn, %{"uuid" => asset_uuid}) do
    case Uro.Keys.fetch(asset_uuid) do
      {:ok, %{key: key, iv: iv, ttl: ttl}} ->
        json(conn, %{key: key, iv: iv, ttl: ttl})

      :error ->
        conn |> put_status(:not_found) |> json(%{error: "script key not found"})
    end
  end
end
