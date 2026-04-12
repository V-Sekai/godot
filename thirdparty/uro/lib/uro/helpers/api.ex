defmodule Uro.Helpers.API do
  @moduledoc """
  API helper functions.
  """

  import Plug.Conn, only: [put_status: 2]
  import Phoenix.Controller, only: [json: 2]

  alias OpenApiSpex.Schema

  @default_security [%{"cookie" => []}, %{"bearer" => []}]

  @spec default_security() :: [map()]
  def default_security(), do: @default_security

  def json_error(conn, options \\ []) do
    code =
      options
      |> Keyword.get(:code, :bad_request)
      |> Plug.Conn.Status.code()

    conn
    |> put_status(code)
    |> json(
      Enum.into(options, %{
        message: Keyword.get(options, :message, Plug.Conn.Status.reason_phrase(code)),
        code: Plug.Conn.Status.reason_atom(code)
      })
    )
  end

  @success_json_schema %Schema{
    title: "Success",
    description: "A representation of an generic success response.",
    type: :object,
    required: [:message],
    properties: %{
      message: %Schema{
        type: :string,
        description: "A human-readable message describing the response."
      }
    }
  }

  def success_json_schema, do: @success_json_schema

  @doc """
    A list of HTTP status codes and their corresponding reason phrases.
    These were taken from the `Plug.Conn.Status` module since they are
    not exposed in the public API.
  """
  @statuses %{
    400 => "Bad Request",
    401 => "Unauthorized",
    402 => "Payment Required",
    403 => "Forbidden",
    404 => "Not Found",
    405 => "Method Not Allowed",
    406 => "Not Acceptable",
    407 => "Proxy Authentication Required",
    408 => "Request Timeout",
    409 => "Conflict",
    410 => "Gone",
    411 => "Length Required",
    412 => "Precondition Failed",
    413 => "Request Entity Too Large",
    414 => "Request-URI Too Long",
    415 => "Unsupported Media Type",
    416 => "Requested Range Not Satisfiable",
    417 => "Expectation Failed",
    418 => "I'm a teapot",
    421 => "Misdirected Request",
    422 => "Unprocessable Entity",
    423 => "Locked",
    424 => "Failed Dependency",
    425 => "Too Early",
    426 => "Upgrade Required",
    428 => "Precondition Required",
    429 => "Too Many Requests",
    431 => "Request Header Fields Too Large",
    451 => "Unavailable For Legal Reasons",
    500 => "Internal Server Error",
    501 => "Not Implemented",
    502 => "Bad Gateway",
    503 => "Service Unavailable",
    504 => "Gateway Timeout",
    505 => "HTTP Version Not Supported",
    506 => "Variant Also Negotiates",
    507 => "Insufficient Storage",
    508 => "Loop Detected",
    510 => "Not Extended",
    511 => "Network Authentication Required"
  }

  @status_atoms @statuses
                |> Map.values()
                |> Enum.map(fn reason_phrase ->
                  reason_phrase
                  |> String.downcase()
                  |> String.replace("'", "")
                  |> String.replace(~r/[^a-z0-9]/, "_")
                  |> String.to_atom()
                end)

  def status_atoms(), do: @status_atoms

  @error_json_schema %Schema{
    title: "Error",
    type: :object,
    required: [:message, :status],
    properties: %{
      message: %Schema{
        type: :string,
        description: "A human-readable message describing the error.",
        example: "Bad request"
      },
      code: %Schema{
        type: :integer,
        example: "bad_request",
        enum: @status_atoms
      },
      properties: %Schema{
        type: :object,
        description: "Property specific errors, in response to invalid requests.",
        example: %{
          username: ["should be at least 3 character(s)"],
          password: ["was already taken"]
        },
        additionalProperties: %Schema{
          type: :array,
          items: %Schema{
            type: :string
          }
        }
      }
    }
  }

  def error_json_schema, do: @error_json_schema

  @doc false
  defmacro __using__(_config) do
    quote do
      import unquote(__MODULE__)

      alias Uro.Helpers.API.Success
    end
  end
end
