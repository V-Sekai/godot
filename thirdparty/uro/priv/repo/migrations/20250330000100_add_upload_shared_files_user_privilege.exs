defmodule Uro.Repo.Migrations.AddUploadSharedFilesUserPrivilege do
  use Ecto.Migration

  def change do
    alter table(:user_privilege_rulesets) do
      add :can_upload_shared_files, :boolean, default: false, null: false
    end
  end
end
