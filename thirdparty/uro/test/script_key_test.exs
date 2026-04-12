defmodule Uro.ScriptKeyTest do
  use ExUnit.Case, async: false

  import Plug.Conn
  import Phoenix.ConnTest

  @endpoint Uro.Endpoint

  @fixture_uuid "asset:script-key-fixture"
  @fixture_key Base.encode64(:binary.copy(<<0xAA>>, 16))
  @fixture_iv Base.encode64(:binary.copy(<<0xBB>>, 12))
  @fixture_ttl 86_400

  setup do
    :ok =
      Uro.Keys.put(@fixture_uuid, %{
        key: @fixture_key,
        iv: @fixture_iv,
        ttl: @fixture_ttl
      })

    on_exit(fn -> Uro.Keys.delete(@fixture_uuid) end)
    :ok
  end

  defp script_key(uuid) do
    build_conn()
    |> put_req_header("content-type", "application/json")
    |> post("/auth/script_key", Jason.encode!(%{uuid: uuid}))
  end

  test "POST /auth/script_key returns the seeded key material" do
    assert json_response(script_key(@fixture_uuid), 200) == %{
             "key" => @fixture_key,
             "iv" => @fixture_iv,
             "ttl" => @fixture_ttl
           }
  end

  test "POST /auth/script_key returns 404 for an unknown asset uuid" do
    assert json_response(script_key("asset:does-not-exist"), 404) ==
             %{"error" => "script key not found"}
  end
end
