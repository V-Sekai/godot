defmodule Uro.Helpers.Validation do
  @moduledoc """
  Helper module to validate files.
  """

  require Logger

  @magic_numbers %{
    ".glb" => <<0x67, 0x6C, 0x54, 0x46>>,
    ".vrm" => <<0x67, 0x6C, 0x54, 0x46>>,
    ".scn" => <<0x52, 0x53, 0x43, 0x43>>
  }

  def init_extra_extensions() do
    ExMarcel.Magic.add("application/vnd.godot.scn",
      extensions: ["scn"],
      magic: [[0, "\x52\x53\x43\x43"]],
      parents: []
    )

    ExMarcel.Magic.add("model/gltf-binary",
      extensions: ["glb", "vrm"],
      magic: [[0, "\x67\x6C\x54\x46"]],
      parents: []
    )
  end

  # Weak check, ensures magic matches extension mime for some file types.
  @spec check_magic_number(%{file_name: String.t(), path: String.t()}) :: boolean
  def check_magic_number(%{file_name: file_name, path: path}) do
    file_extension = file_name |> Path.extname() |> String.downcase()
    magic_number = Map.get(@magic_numbers, file_extension)

    if magic_number != nil do
      expected_length = byte_size(magic_number)

      case :file.open(path, [:read, :binary]) do
        {:ok, file_handle} ->
          result =
            with {:ok, file_content} <- :file.read(file_handle, expected_length),
                 true <- byte_size(file_content) >= expected_length,
                 true <- :binary.part(file_content, 0, expected_length) == magic_number,
                 do: true,
                 else: (_ -> false)

          :file.close(file_handle)
          result

        {:error, reason} ->
          Logger.error("Error opening file: #{reason}")
          false
      end

      # Skip check if ext not in custom magic numbers
    else
      true
    end
  end

  def generate_file_sha256(file_path) do
    # 4KB chunks
    file_stream = File.stream!(file_path, [], 4096)

    hash =
      Enum.reduce(file_stream, :crypto.hash_init(:sha256), fn chunk, acc ->
        :crypto.hash_update(acc, chunk)
      end)

    :crypto.hash_final(hash)
    |> Base.encode16(case: :lower)
  end
end
