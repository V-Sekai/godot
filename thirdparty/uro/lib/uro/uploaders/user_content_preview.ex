defmodule Uro.Uploaders.UserContentPreview do
  use Waffle.Definition
  use Waffle.Ecto.Definition
  alias Uro.Helpers.Validation

  @versions [:original, :thumb]
  @extension_whitelist ~w(.jpg .jpeg .gif .png)

  # Override the bucket on a per definition basis:
  # def bucket do
  #   :custom_bucket_name
  # end

  # Whitelist file extensions:
  def validate({file, _}) do
    file_extension = file.file_name |> Path.extname() |> String.downcase()

    with true <- Enum.member?(@extension_whitelist, file_extension),
         true <- Validation.check_magic_number(file),
         do: true,
         else: (_ -> false)
  end

  # Override the persisted filenames:
  def filename(version, {_file, scope}) do
    "#{scope.id}_preview_#{version}"
  end

  # Override the storage directory:
  def storage_dir(_version, {_file, _scope}) do
    "uploads/"
  end

  def default_url(version, _scope) do
    "/images/user_content/default_#{version}.png"
  end
end
