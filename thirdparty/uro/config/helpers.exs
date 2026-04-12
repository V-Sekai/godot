defmodule Uro.Config.Helpers do
  @compile_phase? System.get_env("COMPILE_PHASE") != "false"

  def get_env(key, example) do
    case @compile_phase? do
      true ->
        example

      false ->
        System.get_env(key) ||
          raise """
          Environment variable #{key} is required but not set.
          """
    end
  end
end
