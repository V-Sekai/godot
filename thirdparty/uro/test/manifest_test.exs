defmodule Uro.ManifestTest do
  use ExUnit.Case, async: false

  import Plug.Conn
  import Phoenix.ConnTest

  @endpoint Uro.Endpoint

  setup do
    :ok =
      Uro.Manifest.put("asset:fixture", [
        %{id: String.duplicate("a", 64), start: 0, size: 1024},
        %{id: String.duplicate("b", 64), start: 1024, size: 2048}
      ])

    on_exit(fn -> Uro.Manifest.delete("asset:fixture") end)
    :ok
  end

  defp manifest(id) do
    build_conn()
    |> put_req_header("content-type", "application/json")
    |> post("/storage/#{id}/manifest", "{}")
  end

  test "POST /storage/:id/manifest returns the seeded chunk list" do
    assert json_response(manifest("asset:fixture"), 200) == %{
             "chunks" => [
               %{"id" => String.duplicate("a", 64), "start" => 0, "size" => 1024},
               %{"id" => String.duplicate("b", 64), "start" => 1024, "size" => 2048}
             ]
           }
  end

  test "POST /storage/:id/manifest returns 404 for an unknown asset id" do
    assert json_response(manifest("asset:does-not-exist"), 404) ==
             %{"error" => "manifest not found"}
  end
end
