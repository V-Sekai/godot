defmodule Uro.VSekai.ShardJanitor do
  use GenServer
  alias Uro.Repo
  import Ecto.Query, only: [from: 2]

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{})
  end

  def init(state) do
    schedule_cleanup()
    {:ok, state}
  end

  def handle_info(:cleanup, state) do
    cleanup_stale_shards()
    schedule_cleanup()
    {:noreply, state}
  end

  defp cleanup_stale_shards() do
    stale_shard_cutoff = Application.get_env(:uro, :stale_shard_cutoff)

    query =
      from s in "shards",
        where:
          s.updated_at >
            from_now(^stale_shard_cutoff[:amount], ^stale_shard_cutoff[:calendar_type]),
        select: s.id

    Repo.delete_all(query)
  end

  defp schedule_cleanup() do
    # every 3 days
    # 3 * 24 * 60 * 60 * 1000)
    Process.send_after(self(), :cleanup, Application.get_env(:uro, :stale_shard_interval))
  end
end
