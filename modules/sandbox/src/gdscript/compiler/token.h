/**************************************************************************/
/*  token.h                                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once
#include <string>
#include <variant>

namespace gdscript {

enum class TokenType {
	// Literals
	IDENTIFIER,
	INTEGER,
	FLOAT,
	STRING,

	// Keywords
	FUNC,
	VAR,
	CONST,
	RETURN,
	IF,
	ELSE,
	ELIF,
	FOR,
	IN,
	WHILE,
	BREAK,
	CONTINUE,
	PASS,
	EXTENDS,
	TRUE,
	FALSE,
	NULL_VAL,

	// Operators
	PLUS, // +
	MINUS, // -
	MULTIPLY, // *
	DIVIDE, // /
	MODULO, // %
	ASSIGN, // =
	PLUS_ASSIGN, // +=
	MINUS_ASSIGN, // -=
	MULTIPLY_ASSIGN, // *=
	DIVIDE_ASSIGN, // /=
	MODULO_ASSIGN, // %=
	EQUAL, // ==
	NOT_EQUAL, // !=
	LESS, // <
	LESS_EQUAL, // <=
	GREATER, // >
	GREATER_EQUAL, // >=
	AND, // and
	OR, // or
	NOT, // not

	// Delimiters
	LPAREN, // (
	RPAREN, // )
	LBRACKET, // [
	RBRACKET, // ]
	LBRACE, // {
	RBRACE, // }
	COLON, // :
	COMMA, // ,
	DOT, // .
	AT, // @
	NEWLINE,
	INDENT,
	DEDENT,

	// Special
	EOF_TOKEN,
	INVALID
};

struct Token {
	TokenType type;
	std::string lexeme;
	std::variant<int64_t, double, std::string> value;
	int line;
	int column;

	Token() : type(TokenType::INVALID), line(0), column(0) {}
	Token(TokenType t, std::string lex, int l, int c) : type(t), lexeme(std::move(lex)), line(l), column(c) {}

	bool is_type(TokenType t) const { return type == t; }
	bool is_one_of(TokenType t1, TokenType t2) const { return type == t1 || type == t2; }

	template <typename... Types>
	bool is_one_of(TokenType first, Types... rest) const {
		return type == first || is_one_of(rest...);
	}

	std::string to_string() const;
};

const char *token_type_name(TokenType type);

} // namespace gdscript
