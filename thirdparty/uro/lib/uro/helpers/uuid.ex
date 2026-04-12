defmodule Uro.Helpers.UUID do
  @moduledoc """
  UUID helper functions.
  """

  defmacro is_uuid(value) do
    quote do
      is_binary(unquote(value)) and byte_size(unquote(value)) == 36 and
        binary_part(unquote(value), 8, 1) == "-" and binary_part(unquote(value), 13, 1) == "-" and
        binary_part(unquote(value), 18, 1) == "-" and binary_part(unquote(value), 23, 1) == "-"
    end
  end
end
