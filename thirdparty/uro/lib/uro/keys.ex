defmodule Uro.Keys do
  @moduledoc """
  In-memory asset key store.

  Maps an asset uuid to its AES-128-GCM `{key, iv, ttl}` triple used by
  the encrypted-chunk delivery pipeline. `key` and `iv` are stored and
  returned as base64 strings on the wire so the client can decode them
  with `PackedByteArray.from_base64`. The production path will derive
  keys through KMS and gate `/auth/script_key` behind the JWT
  authentication pipeline; the MVP keeps the lookup in-process so
  tests don't need a DB or KMS fixture.
  """

  use GenServer

  @type key_material :: %{key: String.t(), iv: String.t(), ttl: pos_integer}

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @spec put(String.t(), key_material) :: :ok
  def put(asset_uuid, material), do: GenServer.call(__MODULE__, {:put, asset_uuid, material})

  @spec delete(String.t()) :: :ok
  def delete(asset_uuid), do: GenServer.call(__MODULE__, {:delete, asset_uuid})

  @spec fetch(String.t()) :: {:ok, key_material} | :error
  def fetch(asset_uuid), do: GenServer.call(__MODULE__, {:fetch, asset_uuid})

  @impl true
  def init(_), do: {:ok, %{}}

  @impl true
  def handle_call({:put, uuid, material}, _from, state),
    do: {:reply, :ok, Map.put(state, uuid, material)}

  def handle_call({:delete, uuid}, _from, state),
    do: {:reply, :ok, Map.delete(state, uuid)}

  def handle_call({:fetch, uuid}, _from, state),
    do: {:reply, Map.fetch(state, uuid), state}
end
