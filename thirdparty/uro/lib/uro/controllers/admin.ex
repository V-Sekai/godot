defmodule Uro.AdminController do
  use Uro, :controller

  alias OpenApiSpex.Schema

  tags(["admin"])

  operation(:status,
    operation_id: "getAdminStatus",
    summary: "Get Admin status",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{
          type: :object,
          properties: %{
            status: %Schema{
              type: :object,
              properties: %{
                is_admin: %Schema{
                  type: :string,
                  enum: ["true", "false"]
                }
              }
            }
          }
        }
      }
    ]
  )

  def status(conn, _params) do
    json(conn, %{status: %{is_admin: "true"}})
  end
end
