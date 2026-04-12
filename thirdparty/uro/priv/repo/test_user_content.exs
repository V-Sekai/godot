# Script for populating the database. You can run it as:
#
#     mix run priv/repo/test_user_content.exs
#
# Script must be run from repository root

alias Uro.UserContent
alias Uro.Repo

# Create upload database entries with user "adminuser"

user = Repo.get_by(Uro.Accounts.User, username: "adminuser")
uploader = user.id

process_file = fn (path, content_type) -> 
  file = %Plug.Upload{
    path: path,
    filename: Path.basename(path),
    content_type: content_type
  }
  file
end


avatar1 = UserContent.create_avatar(%{
      name: "TestAvatar1",
      description: "First test avatar",
      user_content_data: process_file.("priv/repo/test_content/test_avatar1.scn", "application/octet-stream"),
      uploader_id: uploader,
      user_content_preview: process_file.("priv/repo/test_content/test_image.jpg", "image/jpeg"),
      is_public: true
})

scene1 = UserContent.create_map(%{
      name: "TestScene1",
      description: "First test scene",
      user_content_data: process_file.("priv/repo/test_content/test_scene1.scn", "application/octet-stream"),
      uploader_id: uploader,
      user_content_preview: process_file.("priv/repo/test_content/test_image.jpg", "image/jpeg"),
      is_public: true
})

scene2 = UserContent.create_map(%{
      name: "TestScene2",
      description: "Second test scene",
      user_content_data: process_file.("priv/repo/test_content/test_scene2.scn", "application/octet-stream"),
      uploader_id: uploader,
      user_content_preview: process_file.("priv/repo/test_content/test_image.jpg", "image/jpeg"),
      is_public: true
})
