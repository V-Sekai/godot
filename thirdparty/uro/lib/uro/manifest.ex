defmodule Uro.Manifest do
  @moduledoc """
  In-memory asset manifest store.

  Maps an asset id to the list of chunks (`%{id, start, size}`) the client
  should fetch to reassemble the asset. A single POST to
  `/storage/:id/manifest` returns the whole list, avoiding the N+1
  pattern where the client discovers dependencies one fetch at a time.

  The production path will parse a stored `.caibx`/`.caidx` index on demand
  and populate this table (or replace it with a DB-backed lookup). The
  MVP keeps it in-process so tests don't need a fixture file layout.
  """

  use GenServer

  @type chunk :: %{id: String.t(), start: non_neg_integer, size: pos_integer}

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @spec put(String.t(), [chunk]) :: :ok
  def put(asset_id, chunks), do: GenServer.call(__MODULE__, {:put, asset_id, chunks})

  @spec delete(String.t()) :: :ok
  def delete(asset_id), do: GenServer.call(__MODULE__, {:delete, asset_id})

  @spec fetch(String.t()) :: {:ok, [chunk]} | :error
  def fetch(asset_id), do: GenServer.call(__MODULE__, {:fetch, asset_id})

  @impl true
  def init(_), do: {:ok, %{}}

  @impl true
  def handle_call({:put, id, chunks}, _from, state),
    do: {:reply, :ok, Map.put(state, id, chunks)}

  def handle_call({:delete, id}, _from, state),
    do: {:reply, :ok, Map.delete(state, id)}

  def handle_call({:fetch, id}, _from, state),
    do: {:reply, Map.fetch(state, id), state}
end
