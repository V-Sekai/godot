defmodule Uro.Helpers.SharedContentHelper do
  @doc false
  def merge_map_with_base_shared_content(map, shared_content) do
    Map.merge(map, %{
      id: to_string(shared_content.id),
      name: to_string(shared_content.name),
      description: to_string(shared_content.description),
      shared_content_data:
        to_string(
          Uro.Uploaders.SharedContentData.url(
            {shared_content.shared_content_data, shared_content}
          )
        ),
      mime_type: to_string(shared_content.mime_type),
      file_size: shared_content.file_size,
      checksum: to_string(shared_content.checksum),
      version: to_string(shared_content.version),
      tags: shared_content.tags
    })
  end

  @doc false
  def get_api_shared_content(shared_content, config) do
    map = merge_map_with_base_shared_content(%{}, shared_content)

    map =
      if Map.get(config, :merge_is_public, false) == true,
        do: Map.merge(map, %{is_public: shared_content.is_public}),
        else: map

    map =
      if Map.get(config, :merge_inserted_at, false) == true,
        do: Map.merge(map, %{inserted_at: shared_content.inserted_at}),
        else: map

    map =
      if Map.get(config, :merge_updated_at, false) == true,
        do: Map.merge(map, %{updated_at: shared_content.updated_at}),
        else: map

    map =
      if Map.get(config, :merge_uploader_id, false) == true,
        do: Map.merge(map, %{uploader_id: shared_content.uploader_id}),
        else: map

    map
  end

  @doc false
  def get_api_shared_content_list(shared_content_list, config) do
    Enum.map(shared_content_list, fn x -> get_api_shared_content(x, config) end)
  end

  def get_correct_shared_content_params(
        conn,
        shared_content_params,
        shared_content_data_filename_param
      ) do
    shared_content_data = Map.get(shared_content_params, shared_content_data_filename_param)

    %{
      "name" => Map.get(shared_content_params, "name", ""),
      "description" => Map.get(shared_content_params, "description", ""),
      "shared_content_data" => shared_content_data,
      "is_public" => Map.get(shared_content_params, "is_public", false),
      "uploader_id" => conn.assigns[:current_user].id,
      "version" => Map.get(shared_content_params, "version", ""),
      "tags" => Map.get(shared_content_params, "tags", [])
    }
  end

  @doc false
  defp has_upload_shared_files_field?(%{can_upload_shared_files: true}) do
    true
  end

  @doc false
  defp has_upload_shared_files_field?(_) do
    false
  end

  @doc false
  def has_shared_file_upload_permission?(user) do
    user
    |> Uro.Helpers.Auth.get_user_privilege_ruleset()
    |> has_upload_shared_files_field?()
  end

  @doc false
  def session_has_shared_file_upload_permission?(conn) do
    has_shared_file_upload_permission?(conn.assigns[:current_user])
  end
end
