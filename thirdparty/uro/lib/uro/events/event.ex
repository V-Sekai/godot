defmodule Uro.Events.Event do
  use Ecto.Schema
  import Ecto.Changeset

  schema "events" do
    field :description, :string
    field :name, :string

    field :start_date, :utc_datetime
    field :end_date, :utc_datetime

    timestamps()
  end

  @doc false
  def changeset(event, attrs) do
    event
    |> cast(attrs, [:name, :description, :start_date, :end_date])
    |> validate_required([:name, :description, :start_date, :end_date])
  end
end
