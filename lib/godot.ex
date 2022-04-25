defmodule GodotState do
  defstruct pid: nil, last_tick: :os.system_time(:millisecond)
end

defmodule Godot.Loop do
  @godot_timeout 2_000
  @godot_frame 100
  use GenServer
  require Unifex.CNode

  @impl true
  def init(args) do
    {:ok, pid} = Unifex.CNode.start_link(:godot_elixir)
    Unifex.CNode.call(pid, :init, [["godot"]] ++ args, @godot_timeout)
    state = %GodotState{pid: pid, last_tick: :os.system_time(:millisecond)}
    {:ok, state}
  end

  @impl true
  def terminate(_reason, state) do
    Unifex.CNode.stop(state.pid)
    state = %{state | last_tick: :os.system_time(:millisecond)}      
    {:noreply, state}
  end

  @impl true
  def handle_call({:call, args}, _from, state) do
    res = Unifex.CNode.call(state.pid, :call, args, @godot_timeout)
    state = %{state | last_tick: :os.system_time(:millisecond)}
    {:reply, [res], state}
  end
end

defmodule Godot do
  use Unifex.Loader
  def start_link(args) do
     GenServer.start_link(Godot.Loop, args)
  end
end
