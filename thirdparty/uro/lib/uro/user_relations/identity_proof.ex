defmodule Uro.UserRelations.IdentityProof do
  @derive {Jason.Encoder, only: [:user_from, :user_to]}
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  @derive {Phoenix.Param, key: :id}
  schema "identity_proofs" do
    belongs_to :user_from, Uro.Accounts.User, foreign_key: :user_from_id, type: :binary_id
    belongs_to :user_to, Uro.Accounts.User, foreign_key: :user_to_id, type: :binary_id

    timestamps()
  end

  @doc false
  def changeset(identity_proof, attrs) do
    cast(identity_proof, attrs, [:user_from_id, :user_to_id])
  end
end
