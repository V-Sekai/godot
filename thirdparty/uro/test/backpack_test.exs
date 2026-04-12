defmodule Uro.BackpackTest do
  alias Uro.Inventory.Backpack
  use Uro.RepoCase
  import Ecto.Changeset

  alias Uro.Accounts.User

  alias Uro.UserContent.Avatar
  alias Uro.UserContent.Map
  alias Uro.UserContent.Prop

  test "backpack relations works as expected" do
    user = Repo.insert!(%User{} |> cast(%{email: "test@test.test"}, [:email]))

    backpack =
      Repo.get(User, user.id)
      |> build_assoc(:backpacks)
      |> Repo.insert!()
      |> Repo.preload([:avatars, :maps, :props])

    avatar = %Avatar{} |> change() |> Repo.insert!()
    map = %Map{} |> change() |> Repo.insert!()
    prop = %Prop{} |> change() |> Repo.insert!()

    backpack |> change() |> put_assoc(:avatars, [avatar | backpack.avatars]) |> Repo.update!()
    backpack |> change() |> put_assoc(:maps, [map | backpack.maps]) |> Repo.update!()
    backpack |> change() |> put_assoc(:props, [prop | backpack.props]) |> Repo.update!()

    user = Repo.preload(user, [:avatars, :maps, :props])

    assert List.first(user.avatars)
    assert List.first(user.maps)
    assert List.first(user.props)
  end
end
