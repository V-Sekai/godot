module Godot

interface [CNode]

state_type "MyState"

spec init(state, arguments :: [string]) :: {:ok :: label, state, code :: int}
                                  | {:fail :: label, state, result :: string}
spec call(state,
    method :: string, \
    type_1 :: atom, arg_1 :: payload, \
    type_2 :: atom, arg_2 :: payload, \
    type_3 :: atom, arg_3 :: payload, \
    type_4 :: atom, arg_4 :: payload, \
    type_5 :: atom, arg_5 :: payload)  :: {:ok :: label, state, :string :: label, result_string :: string}
                                  # | {:ok :: label, state, :float :: label, result_string :: float}
                                  | {:ok :: label, state, :int :: label, result_string :: int}
                                  | {:ok :: label, state, :bool ::label, result_string :: bool}