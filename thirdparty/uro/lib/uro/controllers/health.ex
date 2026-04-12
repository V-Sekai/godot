defmodule Uro.HealthController do
  use Uro, :controller

  alias OpenApiSpex.Schema

  tags(["system"])

  operation(:index,
    operation_id: "health",
    summary: "Health",
    responses: [
      ok: {
        "",
        "application/json",
        %Schema{
          type: :object,
          properties: %{
            services: %Schema{
              type: :object,
              properties: %{
                uro: %Schema{
                  type: :string,
                  enum: ["healthy", "unhealthy"]
                }
              }
            }
          }
        }
      }
    ]
  )

  def index(conn, _params) do
    json(conn, %{services: %{uro: "healthy"}})
  end
end
