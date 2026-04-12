defmodule Uro.Helpers.UserContentHelper do
  @doc false
  def merge_map_with_base_user_content(map, user_content) do
    Map.merge(map, %{
      id: to_string(user_content.id),
      name: to_string(user_content.name),
      description: to_string(user_content.description),
      user_content_data:
        to_string(
          Uro.Uploaders.UserContentData.url({user_content.user_content_data, user_content})
        ),
      user_content_preview:
        to_string(
          Uro.Uploaders.UserContentPreview.url({user_content.user_content_preview, user_content})
        )
    })
  end

  @doc false
  def get_api_user_content(user_content, config) do
    map = merge_map_with_base_user_content(%{}, user_content)

    map =
      if Map.get(config, :merge_is_public, false) == true,
        do: Map.merge(map, %{is_public: user_content.is_public}),
        else: map

    map =
      if Map.get(config, :merge_inserted_at, false) == true,
        do: Map.merge(map, %{inserted_at: user_content.inserted_at}),
        else: map

    map =
      if Map.get(config, :merge_updated_at, false) == true,
        do: Map.merge(map, %{updated_at: user_content.updated_at}),
        else: map

    map =
      if Map.get(config, :merge_uploader_id, false) == true,
        do: Map.merge(map, %{uploader_id: user_content.uploader_id}),
        else: map

    map
  end

  @doc false
  def get_api_user_content_list(user_content_list, config) do
    Enum.map(user_content_list, fn x -> get_api_user_content(x, config) end)
  end

  @doc false
  defp has_upload_avatars_field?(%{can_upload_avatars: true}) do
    true
  end

  @doc false
  defp has_upload_avatars_field?(_) do
    false
  end

  @doc false
  defp has_upload_maps_field?(%{can_upload_maps: true}) do
    true
  end

  @doc false
  defp has_upload_maps_field?(_) do
    false
  end

  @doc false
  defp has_upload_props_field?(%{can_upload_props: true}) do
    true
  end

  @doc false
  defp has_upload_props_field?(_) do
    false
  end

  @doc false
  def has_avatar_upload_permission?(user) do
    user
    |> Uro.Helpers.Auth.get_user_privilege_ruleset()
    |> has_upload_avatars_field?()
  end

  @doc false
  def has_map_upload_permission?(user) do
    user
    |> Uro.Helpers.Auth.get_user_privilege_ruleset()
    |> has_upload_maps_field?()
  end

  @doc false
  def has_prop_upload_permission?(user) do
    user
    |> Uro.Helpers.Auth.get_user_privilege_ruleset()
    |> has_upload_props_field?()
  end

  @spec session_has_avatar_upload_permission?(
          atom
          | %{:assigns => nil | maybe_improper_list | map, optional(any) => any}
        ) :: boolean
  @doc false
  def session_has_avatar_upload_permission?(conn) do
    has_avatar_upload_permission?(conn.assigns[:current_user])
  end

  @doc false
  def session_has_map_upload_permission?(conn) do
    has_map_upload_permission?(conn.assigns[:current_user])
  end

  @doc false
  def session_has_prop_upload_permission?(conn) do
    has_prop_upload_permission?(conn.assigns[:current_user])
  end

  def get_correct_user_content_params(
        conn,
        user_content_params,
        user_content_data_filename_param,
        user_content_data_preview_param
      ) do
    user_content_data = Map.get(user_content_params, user_content_data_filename_param)
    user_content_preview = Map.get(user_content_params, user_content_data_preview_param)

    %{
      "name" => Map.get(user_content_params, "name", ""),
      "description" => Map.get(user_content_params, "description", ""),
      "user_content_data" => user_content_data,
      "user_content_preview" => user_content_preview,
      "is_public" => Map.get(user_content_params, "is_public", false),
      "uploader_id" => conn.assigns[:current_user].id
    }
  end
end
