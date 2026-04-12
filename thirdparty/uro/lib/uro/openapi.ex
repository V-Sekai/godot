defmodule Uro.OpenAPI.Specification do
  @moduledoc """
  OpenAPI spec for the Uro API.
  """

  alias OpenApiSpex.Components
  alias OpenApiSpex.Info
  alias OpenApiSpex.OpenApi
  alias OpenApiSpex.Paths
  alias OpenApiSpex.SecurityScheme
  alias OpenApiSpex.Server
  alias Uro.Endpoint
  alias Uro.Router

  @behaviour OpenApi

  @impl OpenApi
  def spec() do
    %OpenApi{
      info: %Info{
        title: "Uro, by V-Sekai",
        version: "1"
      },
      servers: [
        %Server{
          url: Endpoint.public_url()
        }
      ],
      paths: Paths.from_router(Router),
      components: %Components{
        securitySchemes: %{
          "bearer" => %SecurityScheme{
            type: "http",
            scheme: "bearer"
          },
          "cookie" => %SecurityScheme{
            type: "apiKey",
            in: "cookie",
            name: "session"
          }
        }
      }
    }
    |> OpenApiSpex.resolve_schema_modules()
  end
end
