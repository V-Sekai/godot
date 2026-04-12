defmodule Uro.ManifestController do
  @moduledoc """
  POST /storage/:id/manifest — returns every chunk the client needs to
  reassemble an asset in one round trip. See CONCEPT_MMOG.md for the
  wire format and why this exists (avoids N+1 chunk discovery).
  """

  use Uro, :controller

  def show(conn, %{"id" => asset_id}) do
    case Uro.Manifest.fetch(asset_id) do
      {:ok, chunks} -> json(conn, %{chunks: chunks})
      :error -> conn |> put_status(:not_found) |> json(%{error: "manifest not found"})
    end
  end
end
