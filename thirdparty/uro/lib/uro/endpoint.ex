defmodule Uro.Endpoint do
  use Phoenix.Endpoint, otp_app: :uro

  def public_url(pathname \\ "") do
    URI.to_string(Application.fetch_env!(:uro, :url)) <> pathname
  end

  if Mix.env() == :dev do
    plug(Phoenix.CodeReloader)
  end

  plug(Plug.Static,
    at: "/uploads",
    from: Path.expand("./uploads"),
    gzip: false
  )

  plug(Plug.RequestId, assign_as: :request_id)
  plug(Plug.Telemetry, event_prefix: [:phoenix, :endpoint])

  # Max upload size, 200mb
  plug(Plug.Parsers,
    parsers: [:urlencoded, {:multipart, length: 200_000_000}, :json],
    query_string_length: 1_000_000,
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()
  )

  plug(Plug.MethodOverride)
  plug(Plug.Head)

  plug(Plug.Session,
    store: :cookie,
    key: "session",
    signing_salt: "5DeFKvbM"
  )

  plug(CORSPlug)
  plug(Uro.Router)
end
