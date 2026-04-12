defmodule Uro.AclController do
  @moduledoc """
  ReBAC relation check endpoint. Resolves a `(object, relation, subject)`
  tuple against `Uro.Acl`. All token-issuance and upload-gating code paths
  in uro call this endpoint instead of consulting flat privilege flags.
  """

  use Uro, :controller

  def check(conn, %{"object" => o, "relation" => r, "subject" => s}) do
    json(conn, %{allowed: Uro.Acl.check(o, r, s)})
  end
end
