# defmodule Uro.JSONSchema do
#   @moduledoc """
#   JSON Schema helpers.
#   """
#
#   defmacro __using__(_) do
#     quote do
#       require OpenApiSpex
#       import Uro.JSONSchema
#
#       alias OpenApiSpex.Schema
#
#       def shape(), do: schema()
#
#       def shape(key) when is_atom(key) do
#         schema()
#         |> Map.get(:properties)
#         |> Map.get(key)
#       end
#     end
#   end
# end
#
