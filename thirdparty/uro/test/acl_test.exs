defmodule Uro.AclTest do
  # No Repo needed: Uro.Acl is a pure in-memory GenServer, and ExUnit.Case
  # keeps the test isolated from the Postgrex sandbox so the existing
  # backpack_test bug can't contaminate us.
  use ExUnit.Case, async: false

  import Plug.Conn
  import Phoenix.ConnTest

  @endpoint Uro.Endpoint

  setup do
    # Seed a single (object, relation, subject) tuple for each test and
    # tear it down afterwards so tests don't leak state between runs.
    :ok = Uro.Acl.put({"asset:1", "viewer", "user:1"})
    on_exit(fn -> Uro.Acl.delete({"asset:1", "viewer", "user:1"}) end)
    :ok
  end

  defp check(params) do
    build_conn()
    |> put_req_header("content-type", "application/json")
    |> post("/acl/check", Jason.encode!(params))
  end

  test "POST /acl/check returns allowed: true for a seeded tuple" do
    assert json_response(check(%{object: "asset:1", relation: "viewer", subject: "user:1"}), 200) ==
             %{"allowed" => true}
  end

  test "POST /acl/check returns allowed: false when the tuple is not in the store" do
    assert json_response(check(%{object: "asset:1", relation: "viewer", subject: "user:99"}), 200) ==
             %{"allowed" => false}
  end

  test "POST /acl/check resolves a relation transitively through group membership" do
    # (group:vip, member, user:2) + (asset:2, viewer, group:vip) should mean
    # user:2 inherits the viewer relation on asset:2 via the vip group edge.
    :ok = Uro.Acl.put({"group:vip", "member", "user:2"})
    :ok = Uro.Acl.put({"asset:2", "viewer", "group:vip"})

    on_exit(fn ->
      Uro.Acl.delete({"group:vip", "member", "user:2"})
      Uro.Acl.delete({"asset:2", "viewer", "group:vip"})
    end)

    assert json_response(check(%{object: "asset:2", relation: "viewer", subject: "user:2"}), 200) ==
             %{"allowed" => true}
  end
end
