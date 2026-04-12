defmodule Uro.Acl do
  @moduledoc """
  In-memory ReBAC relation store.

  Holds `{object, relation, subject}` tuples where every component is an
  opaque string (`"asset:123"`, `"viewer"`, `"user:456"`). The MVP check
  is a direct-edge lookup; transitive resolution across `member` edges is
  added in a later slice.
  """

  use GenServer

  @type tuple3 :: {String.t(), String.t(), String.t()}

  # ── Client ──────────────────────────────────────────────────────────

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @spec put(tuple3) :: :ok
  def put({o, r, s}), do: GenServer.call(__MODULE__, {:put, {o, r, s}})

  @spec delete(tuple3) :: :ok
  def delete({o, r, s}), do: GenServer.call(__MODULE__, {:delete, {o, r, s}})

  @spec check(String.t(), String.t(), String.t()) :: boolean
  def check(object, relation, subject),
    do: GenServer.call(__MODULE__, {:check, object, relation, subject})

  # ── Server ──────────────────────────────────────────────────────────

  @impl true
  def init(_), do: {:ok, MapSet.new()}

  @impl true
  def handle_call({:put, tuple}, _from, state), do: {:reply, :ok, MapSet.put(state, tuple)}

  def handle_call({:delete, tuple}, _from, state),
    do: {:reply, :ok, MapSet.delete(state, tuple)}

  def handle_call({:check, o, r, s}, _from, state),
    do: {:reply, resolve(state, o, r, s), state}

  # Direct edge, or a one-hop rewrite through a `member` group edge:
  # `(o, r, g) + (g, member, s)` ⇒ `(o, r, s)`. Matches the ReBAC convention
  # in CONCEPT_MMOG.md — group membership propagates viewer/uploader/etc.
  # without writing a per-user tuple. Only a single hop is resolved today;
  # multi-hop graphs land when the first real use case needs them.
  defp resolve(state, o, r, s) do
    MapSet.member?(state, {o, r, s}) or
      Enum.any?(state, fn
        {^o, ^r, group} -> MapSet.member?(state, {group, "member", s})
        _ -> false
      end)
  end
end
