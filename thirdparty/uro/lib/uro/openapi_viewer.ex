defmodule Uro.OpenAPI.Viewer do
  @moduledoc """
  A Plug that renders the OpenAPI spec as an interactive API documentation page.
  """

  @behaviour Plug

  @impl Plug
  def init(opts) when is_list(opts) do
    Map.new(opts)
  end

  @impl Plug
  def call(conn, _) do
    {spec, _} = OpenApiSpex.Plug.PutApiSpec.get_spec_and_operation_lookup(conn)

    conn
    |> Plug.Conn.put_resp_content_type("text/html")
    |> Plug.Conn.send_resp(200, """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>API Documentation · #{spec.info.title}</title>
        <script src="https://unpkg.com/@stoplight/elements/web-components.min.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/@stoplight/elements/styles.min.css">
      </head>
      <body style="height: 100svh;">
        <elements-api
          layout="responsive"
          router="hash"
          tryItCredentialsPolicy="include"
        />
        <script>
          document.querySelector("elements-api").apiDescriptionDocument = #{Jason.encode!(spec)};
        </script>
      </body>
    </html>
    """)
  end
end
