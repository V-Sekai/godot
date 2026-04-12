defmodule Uro.SharedContent.SharedContent do
  alias Ecto.Changeset

  @doc false
  defmacro __using__(_config) do
    quote do
      use Ecto.Schema
      use Waffle.Ecto.Schema
      alias Uro.Helpers

      @derive {Jason.Encoder,
               only: [
                 :name,
                 :description,
                 :uploader_id,
                 :is_public,
                 :shared_content_data,
                 :mime_type,
                 :checksum,
                 :file_size,
                 :version,
                 :tags
               ]}
      import unquote(__MODULE__), only: [shared_content_fields: 0]

      @json_schema %OpenApiSpex.Schema{
        title: "SharedContent",
        type: :object,
        properties: %{
          id: %OpenApiSpex.Schema{type: :string, format: :uuid, description: "File ID"},
          name: %OpenApiSpex.Schema{type: :string, description: "File name"},
          description: %OpenApiSpex.Schema{type: :string, description: "File description"},
          uploader_id: %OpenApiSpex.Schema{
            type: :string,
            format: :uuid,
            description: "Uploader ID"
          },
          is_public: %OpenApiSpex.Schema{type: :boolean, description: "File is public"},
          shared_content_data: %OpenApiSpex.Schema{
            type: :string,
            description: "Shared content data"
          },
          mime_type: %OpenApiSpex.Schema{type: :string, description: "File MIME type"},
          checksum: %OpenApiSpex.Schema{type: :string, description: "File SHA256 checksum"},
          file_size: %OpenApiSpex.Schema{type: :integer, description: "File size in bytes"},
          version: %OpenApiSpex.Schema{type: :string, description: "File version"},
          tags: %OpenApiSpex.Schema{
            type: :array,
            items: %OpenApiSpex.Schema{type: :string},
            description: "An array of tags"
          }
        },
        required: [
          :id,
          :name,
          :description,
          :uploader_id,
          :is_public,
          :shared_content_data,
          :checksum,
          :file_size,
          :version,
          :tags
        ]
      }

      def json_schema(), do: @json_schema

      @spec shared_content_changeset(Ecto.Schema.t() | Changeset.t(), map()) :: Changeset.t()
      def shared_content_changeset(changeset, attrs) do
        changeset
        |> cast(attrs, [
          :name,
          :description,
          :uploader_id,
          :is_public,
          :checksum,
          :file_size,
          :version,
          :tags
        ])
        |> foreign_key_constraint(:uploader_id)
      end

      def shared_content_upload_changeset(changeset, attrs) do
        changeset
        |> cast_attachments(attrs, [:shared_content_data])
        |> validate_required([:name, :uploader_id, :shared_content_data])
        |> populate_file_data(attrs)
      end

      defp populate_file_data(changeset, attrs) do
        case Map.get(attrs, "shared_content_data") do
          %Plug.Upload{path: path} ->
            file_info = File.stat!(path)
            mime = ExMarcel.MimeType.for({:path, path}, name: Path.basename(path))

            IO.puts("ok put mime #{mime}")

            changeset
            |> put_change(:mime_type, mime)
            |> put_change(:file_size, file_info.size)
            |> put_change(:checksum, Helpers.Validation.generate_file_sha256(path))

          _ ->
            changeset
        end
      end
    end
  end

  @doc false
  defmacro shared_content_fields() do
    quote do
      field :name, :string
      field :description, :string
      field :is_public, :boolean
      field :mime_type, :string
      field :checksum, :string
      field :file_size, :integer
      field :version, :string
      field :tags, {:array, :string}
      field :shared_content_data, Uro.Uploaders.SharedContentData.Type
      belongs_to :uploader, Uro.Accounts.User, foreign_key: :uploader_id, type: :binary_id
    end
  end
end
