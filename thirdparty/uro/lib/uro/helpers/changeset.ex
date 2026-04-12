defmodule Uro.Helpers.Changeset do
  @moduledoc """
  Changeset helper functions.
  """

  import Ecto.Changeset
  alias Ecto.Changeset

  def put_default(%Changeset{} = changeset, key, value_fun)
      when is_atom(key) and is_function(value_fun) do
    case get_field(changeset, key) do
      nil -> put_change(changeset, key, value_fun.())
      _ -> changeset
    end
  end

  def put_default(%Changeset{} = changeset, key, value) when is_atom(key) do
    case get_field(changeset, key) do
      nil -> put_change(changeset, key, value)
      _ -> changeset
    end
  end

  def validate_different(changeset, field_key, options \\ []) do
    new_value = get_field(changeset, field_key)

    if new_value == nil or get_field(changeset, field_key) == new_value do
      add_error(changeset, field_key, Keyword.get(options, :message, "must be different"))
    else
      changeset
    end
  end
end
