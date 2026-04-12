defmodule Uro.RepoCase do
  use ExUnit.CaseTemplate

  using do
    quote do
      alias Uro.Repo

      import Ecto
      import Ecto.Query
      import Uro.RepoCase
    end
  end

  setup tags do
    pid = Ecto.Adapters.SQL.Sandbox.start_owner!(Uro.Repo, shared: not tags[:async])
    on_exit(fn -> Ecto.Adapters.SQL.Sandbox.stop_owner(pid) end)
    :ok
  end
end
